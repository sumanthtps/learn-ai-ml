---
id: project-fraud-detection
title: "Project 1 · Real-Time Fraud Detection API"
sidebar_label: "🔴 Project 1 · Fraud Detection"
sidebar_position: 21
tags: [project, fraud, real-time, redis, background-tasks, monitoring, advanced]
---

# Project 1 · Real-Time Fraud Detection API

> **Industry-Grade Project** — Build a complete fraud detection system from scratch using every concept in this series.

---

## Project Overview and Learning Goals

This project integrates every concept you've learned into one cohesive system. By the end, you'll understand how all the pieces fit together in a real production application.

**What you'll build:**
A real-time transaction scoring API used by a fintech company. Every time a user swipes their credit card, the mobile app calls this API to decide: approve, review, or block the transaction.

**Requirements that mirror industry reality:**
- Score a transaction in under 50ms
- Support 5,000 transactions per second
- Never miss a fraud alert — even if the ML model is temporarily down
- Give fraud analysts a real-time dashboard to review flagged transactions
- Automatically retrain the model weekly on new labeled data

**Concepts from this series applied:**
- Pydantic for transaction validation (Doc 05)
- POST endpoint for scoring (Doc 06)
- Async database for audit logging (Doc 15)
- JWT auth + RBAC for analysts vs API clients (Doc 16)
- Celery for scheduled weekly retraining (Doc 17)
- WebSockets for real-time analyst alerts (Doc 18)
- Redis caching for low-risk transactions (Doc 20)

---

## Understanding the Architecture Before Coding

Before writing a single line of code, understand the data flow:

```
┌─────────────────┐
│  Mobile App     │  (credit card swipe)
│  / POS Terminal │
└────────┬────────┘
         │ POST /transactions/score
         ▼
┌─────────────────────────────────────────────┐
│              FastAPI Gateway                 │
│                                             │
│  Step 1: Validate transaction data          │
│          (Pydantic rejects bad data)        │
│                                             │
│  Step 2: Check Redis cache                  │
│          (Was this same card+merchant       │
│           combination approved recently?)   │
│                                             │
│  Step 3: Build feature vector               │
│          (How many transactions in 1h?)     │
│          (How many countries in 24h?)       │
│                                             │
│  Step 4: Score with ML model                │
│          (fraud_probability = 0.87)        │
│                                             │
│  Step 5: Apply rule engine                  │
│          (Is this a known fraud pattern?)   │
│                                             │
│  Step 6: Return decision (< 50ms)          │
│          {"decision": "block"}              │
│                                             │
│  Step 7: Background tasks (non-blocking):  │
│    - Log to PostgreSQL audit table          │
│    - If risky → broadcast via WebSocket     │
│      to fraud analyst dashboard             │
└─────────────────────────────────────────────┘
```

The key insight: **Steps 1-6 happen synchronously before the response is sent** (must be fast). **Step 7 happens after the response** using BackgroundTasks (can take longer).

---

## Step 1: Transaction Schema — Validating Real Financial Data

The schema does more than validate types. It encodes business rules:

```python title="app/schemas/transaction.py"
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal
from datetime import datetime
from enum import Enum

class MerchantCategory(str, Enum):
    """
    Standard merchant category codes (MCCs) mapped to risk categories.
    Gambling and crypto are high-risk categories — the model weighs these heavily.
    """
    retail = "retail"
    food = "food"
    travel = "travel"
    online = "online"
    atm = "atm"
    gambling = "gambling"     # high risk
    crypto = "crypto"         # high risk

class TransactionInput(BaseModel):
    """
    Represents one credit card transaction attempt.
    
    Why such detailed validation?
    - Missing or malformed data corrupts the ML model's input
    - Early validation prevents bad data from entering the audit log
    - Attackers may send malformed requests to probe for vulnerabilities
    """
    transaction_id: str = Field(
        min_length=10, max_length=50,
        description="Unique transaction identifier from payment processor"
    )
    account_id: str = Field(
        description="The cardholder's account ID"
    )
    amount: float = Field(
        gt=0,
        le=1_000_000,  # no single transaction over $1M
        description="Transaction amount in USD"
    )
    merchant_id: str
    merchant_category: MerchantCategory
    merchant_country: str = Field(
        min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code (e.g., 'US', 'IN')"
    )
    card_present: bool = Field(
        description="True if physical card was used (chip/tap), False if online"
    )
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime
    
    @model_validator(mode="after")
    def validate_online_transaction(self) -> "TransactionInput":
        """
        Cross-field business rule:
        Online transactions (card not present) should have an IP address.
        This doesn't block the transaction — the ML model will handle the risk.
        We just log it for analysis.
        """
        if not self.card_present and not self.ip_address:
            # Log but don't reject — fraud detection needs to see these too
            pass
        return self


class FraudScoreResponse(BaseModel):
    """
    What the API returns after scoring a transaction.
    
    The mobile app uses 'decision' to show the user:
    - 'approve': proceed with transaction
    - 'review': proceed but flag for human review (may call user to verify)
    - 'block': decline the transaction
    """
    transaction_id: str
    fraud_score: float = Field(
        ge=0.0, le=1.0,
        description="0 = definitely legitimate, 1 = definitely fraud"
    )
    decision: Literal["approve", "review", "block"]
    triggered_rules: list[str] = Field(
        description="List of rule IDs that triggered (empty = model only)"
    )
    model_version: str
    processing_time_ms: float
```

---

## Step 2: Feature Engineering — The Secret Sauce

The ML features that distinguish fraud detection from a basic classifier are **velocity features** — behavioral patterns over time. A card used 50 times in one hour is suspicious even if each individual transaction looks normal.

```python title="app/services/feature_engineering.py"
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from models.transaction import Transaction

class FeatureEngineer:
    """
    Computes features from the current transaction + historical database records.
    
    Why do we need the database?
    The ML model needs context: not just "this transaction is $5000"
    but "this account has never before spent more than $200 in a single day".
    That context comes from historical transactions.
    
    The "velocity" features (counts and sums over time windows) are what
    make ML-based fraud detection powerful. Simple rule systems check static
    thresholds. ML can learn complex patterns across all features simultaneously.
    """
    
    async def compute_features(
        self,
        txn: TransactionInput,
        db: AsyncSession,
    ) -> dict:
        """
        Build a feature dictionary from the transaction + its history.
        Returns features in the exact format the model expects.
        """
        features = {}
        
        # ─── Static Features (from current transaction only) ──────
        # These don't require a database lookup
        features["amount"] = txn.amount
        features["amount_log"] = round(__import__("math").log1p(txn.amount), 4)
        # log(1 + amount) compresses the range — $1 vs $10 differs more than $10000 vs $10001
        
        features["is_card_present"] = int(txn.card_present)
        features["is_online"] = int(txn.merchant_category == MerchantCategory.online)
        features["is_atm"] = int(txn.merchant_category == MerchantCategory.atm)
        features["is_gambling"] = int(txn.merchant_category == MerchantCategory.gambling)
        features["is_crypto"] = int(txn.merchant_category == MerchantCategory.crypto)
        features["is_foreign"] = int(txn.merchant_country != "US")
        
        # Time-based features (patterns differ by time of day/week)
        features["hour_of_day"] = txn.timestamp.hour
        features["day_of_week"] = txn.timestamp.weekday()
        features["is_weekend"] = int(txn.timestamp.weekday() >= 5)
        features["is_night"] = int(
            txn.timestamp.hour < 6 or txn.timestamp.hour >= 22
        )
        
        # ─── Velocity Features (require database queries) ─────────
        # We compute these for three time windows: 1h, 24h, 7 days
        # These are expensive but critical for fraud detection
        
        windows = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
        }
        
        for window_name, window_delta in windows.items():
            since = txn.timestamp - window_delta
            
            # Single query that computes multiple aggregates at once
            # (one DB round-trip instead of four)
            result = await db.execute(
                select(
                    func.count().label("txn_count"),
                    func.sum(Transaction.amount).label("txn_total"),
                    func.count(func.distinct(Transaction.merchant_id)).label("unique_merchants"),
                    func.count(func.distinct(Transaction.merchant_country)).label("unique_countries"),
                ).where(
                    Transaction.account_id == txn.account_id,
                    Transaction.timestamp >= since,
                    Transaction.timestamp < txn.timestamp,  # don't count current txn
                )
            )
            row = result.first()
            
            features[f"txn_count_{window_name}"] = row.txn_count or 0
            features[f"txn_total_{window_name}"] = float(row.txn_total or 0)
            features[f"unique_merchants_{window_name}"] = row.unique_merchants or 0
            features[f"unique_countries_{window_name}"] = row.unique_countries or 0
        
        # ─── Derived Features (computed from velocity features) ───
        # Amount relative to recent average spend
        avg_24h = (
            features["txn_total_24h"] / max(features["txn_count_24h"], 1)
        )
        features["amount_vs_24h_avg"] = (
            txn.amount / (avg_24h + 1)  # +1 prevents division by zero
        )
        
        # First time using this merchant in 7 days (new merchant = higher risk)
        features["is_new_merchant_7d"] = int(features["unique_merchants_7d"] == 0)
        
        return features
```

---

## Step 3: The Fraud Model Service

```python title="app/services/fraud_model.py"
import joblib
import pandas as pd
import time
import uuid
from schemas.transaction import TransactionInput, FraudScoreResponse

class FraudModelService:
    """
    Wraps the loaded ML model with:
    1. Feature ordering (ML models are picky about column order)
    2. Decision thresholds (when does probability become "block"?)
    3. Rule engine (hard-coded patterns that always trigger regardless of ML score)
    4. Logging for model monitoring
    """
    
    # Decision thresholds (tuned based on business requirements)
    # Higher BLOCK_THRESHOLD = fewer blocks but more fraud slips through
    # Lower BLOCK_THRESHOLD = more blocks but more false positives (frustrated customers)
    REVIEW_THRESHOLD = 0.40
    BLOCK_THRESHOLD = 0.75
    
    def __init__(self):
        self.model = None
        self.model_version = "not_loaded"
        self.feature_columns: list[str] = []
    
    def load(self, model_path: str):
        """Load model artifact (should be called once at startup)."""
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.model_version = artifact["version"]
        self.feature_columns = artifact["feature_columns"]
    
    def score(
        self,
        features: dict,
        transaction_id: str,
    ) -> FraudScoreResponse:
        """Score one transaction. Returns decision and metadata."""
        start_time = time.perf_counter()
        
        # Ensure feature order matches training
        df = pd.DataFrame([features])[self.feature_columns]
        
        # Get probability of fraud (class 1)
        fraud_prob = float(self.model.predict_proba(df)[0][1])
        
        # Apply rule engine AFTER model
        # Rules can override model to block known fraud patterns immediately
        triggered_rules = self._apply_rules(features)
        if triggered_rules:
            # Any triggered rule forces fraud probability up
            fraud_prob = max(fraud_prob, 0.90)
        
        decision = self._make_decision(fraud_prob)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return FraudScoreResponse(
            transaction_id=transaction_id,
            fraud_score=round(fraud_prob, 4),
            decision=decision,
            triggered_rules=triggered_rules,
            model_version=self.model_version,
            processing_time_ms=round(elapsed_ms, 2),
        )
    
    def _make_decision(self, fraud_prob: float) -> str:
        """Convert probability to business decision."""
        if fraud_prob >= self.BLOCK_THRESHOLD:
            return "block"
        elif fraud_prob >= self.REVIEW_THRESHOLD:
            return "review"
        return "approve"
    
    def _apply_rules(self, features: dict) -> list[str]:
        """
        Hard-coded rules that catch known fraud patterns.
        Rules are applied ON TOP of the model score.
        
        Why rules in addition to ML?
        1. Rules are instant and explainable ("blocked because VELOCITY_EXCEEDED")
        2. New fraud patterns can be added as rules immediately
           (retraining the model takes time)
        3. Regulators sometimes require explainable decisions
        """
        triggered = []
        
        if features.get("txn_count_1h", 0) > 20:
            triggered.append("HIGH_VELOCITY_1H")
        
        if features.get("unique_countries_24h", 0) >= 3:
            triggered.append("MULTI_COUNTRY_24H")
        
        if features.get("is_gambling") and features.get("amount", 0) > 5000:
            triggered.append("HIGH_VALUE_GAMBLING")
        
        if features.get("is_crypto") and features.get("is_night"):
            triggered.append("NIGHT_CRYPTO_PURCHASE")
        
        if features.get("amount_vs_24h_avg", 0) > 10:
            # This transaction is 10× the recent average spend
            triggered.append("UNUSUAL_AMOUNT_SPIKE")
        
        return triggered

# Global singleton
fraud_model = FraudModelService()
```

---

## Step 4: The Scoring Endpoint — Pulling Everything Together

```python title="app/routers/transactions.py"
import uuid
import time
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.ws_manager import manager
from schemas.transaction import TransactionInput, FraudScoreResponse
from services.fraud_model import fraud_model
from services.feature_engineering import FeatureEngineer
from services.cache import prediction_cache

router = APIRouter(prefix="/transactions", tags=["transactions"])
feature_engineer = FeatureEngineer()

async def log_to_database(txn: TransactionInput, score: FraudScoreResponse, db: AsyncSession):
    """
    Log every transaction to PostgreSQL for audit trail.
    Runs in the background AFTER the HTTP response is sent.
    
    Why background? The client doesn't need to wait for the DB write.
    The response time should be < 50ms. DB writes take 5-20ms.
    Running this in the background keeps response time low.
    """
    from models.transaction import TransactionRecord
    record = TransactionRecord(
        transaction_id=txn.transaction_id,
        account_id=txn.account_id,
        amount=txn.amount,
        merchant_id=txn.merchant_id,
        merchant_category=txn.merchant_category.value,
        merchant_country=txn.merchant_country,
        card_present=txn.card_present,
        timestamp=txn.timestamp,
        fraud_score=score.fraud_score,
        decision=score.decision,
        triggered_rules=score.triggered_rules,
        model_version=score.model_version,
    )
    db.add(record)
    await db.commit()

async def notify_analysts(score: FraudScoreResponse):
    """
    Broadcast high-risk transactions to fraud analyst dashboards.
    Analysts have WebSocket connections open — they see alerts instantly.
    """
    if score.decision in ("review", "block"):
        await manager.broadcast("fraud-analysts", {
            "type": "fraud_alert",
            "transaction_id": score.transaction_id,
            "fraud_score": score.fraud_score,
            "decision": score.decision,
            "triggered_rules": score.triggered_rules,
        })

@router.post("/score", response_model=FraudScoreResponse)
async def score_transaction(
    txn: TransactionInput,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Score a transaction for fraud risk.
    
    Response time target: < 50ms (P99)
    
    This endpoint is designed for high throughput:
    - Non-blocking async throughout
    - DB writes happen after response
    - Redis cache for repeated patterns
    """
    start = time.perf_counter()
    
    # ─── 1. Check cache for this card+merchant combination ───────
    cache_key_data = {
        "account": txn.account_id,
        "merchant": txn.merchant_id,
        "amount_bucket": round(txn.amount / 10) * 10,  # round to nearest $10
        "category": txn.merchant_category.value,
    }
    cached_score = await prediction_cache.get(cache_key_data)
    if cached_score and cached_score.get("decision") == "approve":
        # Only cache approved transactions — block/review decisions must be fresh
        cached_score["transaction_id"] = txn.transaction_id
        cached_score["processing_time_ms"] = (time.perf_counter() - start) * 1000
        return FraudScoreResponse(**cached_score)
    
    # ─── 2. Build features from transaction + history ─────────────
    features = await feature_engineer.compute_features(txn, db)
    
    # ─── 3. Score with ML model + rules ──────────────────────────
    score = fraud_model.score(features, txn.transaction_id)
    
    # ─── 4. Cache approved low-risk transactions ──────────────────
    if score.decision == "approve" and score.fraud_score < 0.2:
        await prediction_cache.set(cache_key_data, score.model_dump(), ttl=300)
    
    # ─── 5. Background tasks (after response is sent) ────────────
    background_tasks.add_task(log_to_database, txn, score, db)
    background_tasks.add_task(notify_analysts, score)
    
    score.processing_time_ms = round((time.perf_counter() - start) * 1000, 2)
    return score
```

---

## Step 5: Real-Time Analyst Dashboard via WebSocket

```python title="app/routers/analysts.py"
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from core.ws_manager import manager
from core.security import decode_access_token

router = APIRouter(tags=["analyst-dashboard"])

@router.websocket("/ws/fraud-dashboard")
async def fraud_dashboard(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
):
    """
    WebSocket endpoint for the fraud analyst dashboard.
    
    Authentication: JWT token passed as query parameter.
    (Headers aren't supported in WebSocket upgrade requests by most browsers.)
    
    Connected analysts receive real-time alerts whenever a transaction is
    scored as 'review' or 'block'. They can acknowledge alerts via this
    same connection.
    """
    # Authenticate before accepting the WebSocket connection
    try:
        payload = decode_access_token(token)
        username = payload.get("sub")
        role = payload.get("role")
        if role not in ("admin", "data_scientist", "analyst"):
            await websocket.close(code=4003, reason="Insufficient permissions")
            return
    except ValueError:
        await websocket.close(code=4001, reason="Invalid authentication token")
        return
    
    # Add this analyst to the broadcast room
    await manager.connect(websocket, room="fraud-analysts")
    
    try:
        # Confirm connection
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected to fraud alert stream as {username}",
            "room": "fraud-analysts"
        })
        
        # Keep connection alive, handle analyst messages
        while True:
            msg = await websocket.receive_json()
            
            if msg.get("type") == "acknowledge":
                # Analyst acknowledged a fraud alert
                txn_id = msg.get("transaction_id")
                await mark_transaction_reviewed(txn_id, reviewed_by=username)
                await websocket.send_json({
                    "type": "acknowledged",
                    "transaction_id": txn_id
                })
            
            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "fraud-analysts")
```

---

## Step 6: Weekly Retraining with Celery Beat

```python title="app/worker/tasks/retrain.py"
from worker.celery_app import celery_app
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "weekly-fraud-model-retrain": {
        "task": "worker.tasks.retrain.retrain_fraud_model",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
    }
}

@celery_app.task(name="worker.tasks.retrain.retrain_fraud_model", bind=True)
def retrain_fraud_model(self):
    """
    Weekly automated retraining on new labeled transactions.
    
    Why weekly? Fraud patterns evolve constantly. A model trained 6 months
    ago may miss new fraud techniques. Automated retraining keeps the model
    current without manual intervention.
    
    The retraining uses analyst-reviewed transactions where the ground truth
    (was it actually fraud?) has been confirmed.
    """
    from sqlalchemy import create_engine
    import pandas as pd
    import joblib, uuid
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    
    self.update_state(state="RUNNING", meta={"stage": "fetching_data"})
    
    engine = create_engine(settings.sync_database_url)
    
    # Only use analyst-reviewed transactions as ground truth
    df = pd.read_sql("""
        SELECT * FROM transactions
        WHERE timestamp >= NOW() - INTERVAL '30 days'
          AND analyst_reviewed = true
          AND is_confirmed_fraud IS NOT NULL
    """, engine)
    
    if len(df) < 500:
        return {"status": "skipped", "reason": "Insufficient labeled data"}
    
    self.update_state(state="RUNNING", meta={"stage": "training", "rows": len(df)})
    
    feature_cols = [
        "amount", "amount_log", "is_card_present", "is_online",
        "is_gambling", "is_crypto", "is_foreign", "hour_of_day",
        "txn_count_1h", "txn_count_24h", "unique_countries_24h",
        "amount_vs_24h_avg", "is_new_merchant_7d", "is_night",
    ]
    
    X = df[feature_cols]
    y = df["is_confirmed_fraud"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    test_score = model.score(X_test, y_test)
    
    version = uuid.uuid4().hex[:8]
    artifact = {"model": model, "version": version, "feature_columns": feature_cols}
    path = f"artifacts/fraud_model_{version}.pkl"
    joblib.dump(artifact, path)
    
    return {
        "status": "completed",
        "version": version,
        "test_accuracy": round(test_score, 4),
        "training_samples": len(X_train),
        "path": path,
    }
```

---

## What This Project Teaches You

| Skill | Where You Used It |
|-------|------------------|
| Pydantic cross-field validation | `@model_validator` on TransactionInput |
| Async database queries | Velocity feature computation in FeatureEngineer |
| Background tasks | `background_tasks.add_task(log_to_database)` |
| Redis caching | Cache approved low-risk transactions for 5 minutes |
| WebSocket broadcasting | Real-time analyst alerts via `manager.broadcast()` |
| Celery scheduled tasks | Weekly automated model retraining |
| Hybrid ML + rules | FraudModelService combines model score + rule engine |
| RBAC | Only analysts/admins can access the WebSocket dashboard |
| Performance design | < 50ms response despite DB queries + ML inference |

---

## Testing the System

```bash
# Start all services
docker compose up -d

# Run migrations
docker compose exec api alembic upgrade head

# Score a transaction (should be fast: < 50ms)
curl -X POST http://localhost:8000/transactions/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-2024-001",
    "account_id": "ACC-12345",
    "amount": 5000,
    "merchant_id": "MER-CASINO-01",
    "merchant_category": "gambling",
    "merchant_country": "US",
    "card_present": false,
    "timestamp": "2024-01-15T02:30:00Z"
  }'

# Expected: {"decision": "block", "fraud_score": 0.93, "triggered_rules": ["HIGH_VALUE_GAMBLING", "NIGHT_CRYPTO_PURCHASE"]}

# Connect an analyst to the WebSocket dashboard:
# Use a WebSocket client (Postman, wscat, or browser JS)
# wscat -c "ws://localhost:8000/ws/fraud-dashboard?token=<analyst_jwt>"
# → Any scored "block" or "review" transactions appear here in real time
```
