---
id: production-patterns
title: "14 · Production Patterns (Beyond the Videos)"
sidebar_label: "14 · Production Patterns ⚡"
sidebar_position: 14
tags: [production, testing, auth, async, rate-limiting, cicd, security, database]
---

# Production Patterns — What Every ML Engineer Needs Daily

> **Bonus doc** — Topics not covered in the video series but essential for real-world FastAPI ML APIs.

---

## 1. Testing Your FastAPI App

FastAPI includes `TestClient` — a synchronous test client based on `httpx`.

### Unit Testing Endpoints

```python title="tests/test_predict.py"
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestPredictEndpoint:
    def test_valid_prediction(self):
        response = client.post("/predict", json={
            "age": 30, "sex": "male", "bmi": 27.5,
            "children": 2, "smoker": "no", "region": "southeast"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["low", "medium", "high"]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_missing_field_returns_422(self):
        response = client.post("/predict", json={
            "age": 30, "sex": "male"
            # missing bmi, children, smoker, region
        })
        assert response.status_code == 422

    def test_invalid_type_returns_422(self):
        response = client.post("/predict", json={
            "age": "thirty",   # should be int
            "sex": "male", "bmi": 27.5,
            "children": 2, "smoker": "no", "region": "southeast"
        })
        assert response.status_code == 422

    def test_out_of_range_returns_422(self):
        response = client.post("/predict", json={
            "age": 200,   # age > 100
            "sex": "male", "bmi": 27.5,
            "children": 2, "smoker": "no", "region": "southeast"
        })
        assert response.status_code == 422

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
```

### Mocking the ML Model in Tests

```python title="tests/conftest.py"
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

@pytest.fixture(autouse=True)
def mock_model():
    """Replace the real ML model with a mock in all tests."""
    mock = MagicMock()
    mock.predict.return_value = np.array(["medium"])
    mock.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
    
    with patch("main.model_store", {"model": mock}):
        yield mock

# Now tests run instantly without loading the real model
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=html

# Run specific test
pytest tests/test_predict.py::TestPredictEndpoint::test_valid_prediction -v
```

---

## 2. Authentication & Authorization

### API Key Authentication

```python title="core/security.py"
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from core.config import get_settings

settings = get_settings()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return api_key
```

```python title="routers/predict.py"
from core.security import require_api_key
from fastapi import Depends

@router.post("/predict")
def predict(
    data: InsuranceInput,
    _: str = Depends(require_api_key)   # all requests must have valid API key
):
    ...
```

### JWT Authentication (OAuth2)

```python
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY = "your-256-bit-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
        return {"username": username}
    except JWTError:
        raise credentials_exception

# Login endpoint
@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Validate user credentials from DB
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# Protected endpoint
@app.post("/predict")
async def predict(
    data: InsuranceInput,
    current_user: dict = Depends(get_current_user)
):
    ...
```

---

## 3. Database Integration with SQLAlchemy

```python title="database/models.py"
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone

class Base(DeclarativeBase):
    pass

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    city = Column(String(100))
    weight = Column(Float)
    height = Column(Float)
    smoker = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer)
    prediction = Column(String(20))
    confidence = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
```

```python title="database/session.py"
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

```python title="routers/patients.py"
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import Depends
from database.session import get_db
from database.models import Patient

@router.get("/patients/{patient_id}")
async def get_patient(
    patient_id: int,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient
```

---

## 4. Rate Limiting

```bash
pip install slowapi
```

```python title="main.py"
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")       # 10 predictions per IP per minute
async def predict(request: Request, data: InsuranceInput):
    ...

@app.post("/batch-predict")
@limiter.limit("2/minute")        # stricter limit for expensive operation
async def batch_predict(request: Request, batch: BatchInput):
    ...
```

---

## 5. Caching Predictions with Redis

```python
import redis.asyncio as redis
import json
import hashlib
from core.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to Redis
    app.state.redis = await redis.from_url(settings.redis_url)
    # Load model
    model_store["model"] = joblib.load(settings.model_path)
    yield
    await app.state.redis.close()

def make_cache_key(data: InsuranceInput) -> str:
    """Create a deterministic cache key from the input."""
    serialized = json.dumps(data.model_dump(), sort_keys=True)
    return f"predict:{hashlib.md5(serialized.encode()).hexdigest()}"

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: InsuranceInput, request: Request):
    cache_key = make_cache_key(data)
    redis_client = request.app.state.redis
    
    # Check cache first
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Run inference
    result = run_inference(data)
    
    # Cache for 1 hour
    await redis_client.setex(cache_key, 3600, json.dumps(result.model_dump()))
    
    return result
```

---

## 6. Async Inference — Handling CPU-Bound ML Models

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

# Use ProcessPoolExecutor for CPU-bound ML inference
# (avoids Python GIL on multi-core machines)
process_pool = ProcessPoolExecutor(max_workers=4)

def _sync_predict(features: dict) -> dict:
    """Runs in a separate process — bypasses GIL."""
    import joblib
    import pandas as pd
    model = joblib.load("artifacts/model.pkl")   # each process loads its own model
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    prob = float(max(model.predict_proba(df)[0]))
    return {"prediction": pred, "confidence": prob}

@app.post("/predict")
async def predict(data: InsuranceInput):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        process_pool,
        _sync_predict,
        data.model_dump()
    )
    return result
```

---

## 7. Model A/B Testing

```python
import random

model_store = {
    "model_v1": joblib.load("artifacts/model_v1.pkl"),
    "model_v2": joblib.load("artifacts/model_v2.pkl"),
}

class ABTestConfig(BaseModel):
    v1_traffic_pct: float = 0.8   # 80% to v1, 20% to v2

ab_config = ABTestConfig()

@app.post("/predict")
def predict(data: InsuranceInput):
    # Route to model based on traffic split
    use_v2 = random.random() > ab_config.v1_traffic_pct
    model_key = "model_v2" if use_v2 else "model_v1"
    model = model_store[model_key]
    
    df = pd.DataFrame([data.model_dump()])
    pred = model.predict(df)[0]
    
    # Log which model was used for analysis
    logger.info({"model_used": model_key, "prediction": pred})
    
    return PredictionOutput(
        prediction=pred,
        model_version=model_key
    )
```

---

## 8. WebSockets for Streaming Predictions

```python
from fastapi import WebSocket

@app.websocket("/ws/predictions")
async def prediction_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Run inference
            result = run_inference(InsuranceInput(**data))
            
            # Stream result back
            await websocket.send_json(result.model_dump())
    
    except Exception:
        await websocket.close()
```

---

## 9. OpenTelemetry Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# Instrument all FastAPI routes automatically
FastAPIInstrumentor.instrument_app(app)

# Manual spans for ML inference
tracer = trace.get_tracer(__name__)

@app.post("/predict")
def predict(data: InsuranceInput):
    with tracer.start_as_current_span("ml_inference") as span:
        span.set_attribute("model.version", settings.model_version)
        span.set_attribute("input.smoker", data.smoker)
        
        result = run_inference(data)
        
        span.set_attribute("output.prediction", result.prediction)
        return result
```

---

## 10. Graceful Shutdown

```python
import signal
import sys

def handle_shutdown(signum, frame):
    """Clean up resources before shutdown."""
    logger.info("Shutdown signal received, cleaning up...")
    # Close DB connections, flush logs, etc.
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)
```

Or using lifespan:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_store["model"] = joblib.load(settings.model_path)
    db_pool = await create_db_pool()
    yield
    # Shutdown — always runs even on crash
    await db_pool.close()
    model_store.clear()
    logger.info("Application shut down cleanly")
```

---

## Production Checklist

Before going live, verify these:

```
Security
□ API keys / JWT authentication enabled
□ Non-root user inside Docker container
□ Secrets stored in env vars or secrets manager (not code)
□ HTTPS only (TLS termination at load balancer or Nginx)
□ CORS configured — allow only trusted origins
□ Rate limiting enabled

Reliability
□ /health (liveness) endpoint responds < 100ms
□ /ready (readiness) endpoint checks all dependencies
□ Docker HEALTHCHECK configured
□ Auto-restart policy (--restart unless-stopped)
□ Model loaded at startup (not per-request)
□ Graceful shutdown handler

Observability
□ Structured JSON logging (not print statements)
□ Request ID on every log line
□ Latency logged per endpoint
□ Prediction audit log (inputs + outputs stored)
□ Metrics exposed for Prometheus (/metrics)

Performance
□ Uvicorn workers = 2 × CPU cores + 1
□ Connection pooling for database
□ Redis caching for repeated inputs
□ Response compression (gzip) for large payloads

Testing
□ Unit tests for all endpoints
□ Integration tests with real model
□ Load tests (locust / k6) before launch
□ Validation error tests for all edge cases
```

---

## Quick Reference: Most-Used FastAPI Patterns

```python
# Dependency injection
@app.get("/items")
def list_items(db: Session = Depends(get_db), user = Depends(get_current_user)):
    ...

# Optional auth
@app.get("/public")
def public(user = Depends(get_optional_user)):
    ...

# Router with prefix + auth
router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

# Background task
@app.post("/predict")
def predict(data: Input, bg: BackgroundTasks):
    bg.add_task(log_prediction, data)
    return run_inference(data)

# Custom response headers
from fastapi import Response
@app.post("/predict")
def predict(data: Input, response: Response):
    response.headers["X-Model-Version"] = "1.0"
    return run_inference(data)

# File upload
from fastapi import File, UploadFile
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    ...

# Streaming response
from fastapi.responses import StreamingResponse
@app.get("/stream")
async def stream():
    async def generator():
        for chunk in data:
            yield chunk
    return StreamingResponse(generator(), media_type="application/json")
```
