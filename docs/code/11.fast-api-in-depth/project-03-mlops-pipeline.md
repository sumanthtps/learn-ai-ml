---
id: project-mlops-pipeline
title: "Project 3 · MLOps Pipeline Management API"
sidebar_label: "🟢 Project 3 · MLOps Pipeline"
sidebar_position: 23
tags: [project, mlops, pipeline, model-registry, ab-testing, canary, monitoring, advanced]
---

# Project 3 · MLOps Pipeline Management API

> **Industry-Grade Project** — The complete ML platform backend: experiment tracking, model registry, A/B testing, canary deployments, and drift detection.

---

## What is MLOps and Why Does It Need an API?

MLOps (Machine Learning Operations) is the practice of reliably and efficiently deploying and maintaining ML models in production.

The problem without MLOps:
- Data scientists train models in notebooks
- DevOps engineers don't know what model is running in production
- Nobody knows when the model was last updated
- A new model breaks production — nobody can roll back
- The model drifts silently for weeks before anyone notices

**This project builds the control plane** — the set of APIs that orchestrate the entire ML lifecycle:

```
Data Scientist                  MLOps Platform API              Production
──────────────                  ──────────────────              ──────────
 
POST /experiments/run ─────────► Schedule training job
                                 Monitor progress
                                 Record metrics
                                 
POST /models/register ─────────► Store model artifact
                                 Track version history
                                 
POST /models/{id}/promote ─────► Stage → Champion
                                 Retire old champion
                                 
POST /ab-tests ────────────────► Create A/B test
GET  /ab-tests/{id}/results ───► Statistical analysis
                                 Declare winner
                                 
POST /canary/{model}/start ────►                         5% of traffic
POST /canary/{model}/ramp  ────►                         25% of traffic
POST /canary/{model}/promote ──►                         100% of traffic
                                 
GET  /drift/latest ────────────► PSI check on features
                                 Alert if drift detected
```

---

## Step 1: Experiment Tracking

### What is an Experiment?

An **experiment** is one training run. You record:
- Which dataset was used
- What hyperparameters were set
- What metrics resulted
- Where the artifact (model file) was saved

This creates a full audit trail: "Which model is in production? Version abc123. How well did it perform? 87% accuracy. What data was it trained on? `/data/q4_2024.csv`. Who trained it? Ravi Kumar, on December 15th."

```python title="app/models/experiment.py"
from sqlalchemy import String, Integer, JSON, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from enum import Enum
from .base import Base, TimestampMixin

class ExperimentStatus(str, Enum):
    """
    The lifecycle of a training run:
    queued → running → completed (success)
                    → failed (error)
    cancelled: user cancelled before completion
    """
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(String(500), default="")
    dataset_path: Mapped[str] = mapped_column(String(500))
    hyperparams: Mapped[dict] = mapped_column(JSON, default={})   # {n_estimators: 200, ...}
    metrics: Mapped[dict] = mapped_column(JSON, default={})       # {accuracy: 0.87, f1: 0.85}
    tags: Mapped[dict] = mapped_column(JSON, default={})          # {team: "fraud", env: "prod"}
    status: Mapped[ExperimentStatus] = mapped_column(
        SAEnum(ExperimentStatus), default=ExperimentStatus.queued
    )
    celery_task_id: Mapped[str] = mapped_column(String(50), nullable=True)
    created_by: Mapped[str] = mapped_column(String(100))

class ModelVersion(Base, TimestampMixin):
    """
    Represents one version of a trained model.
    Each experiment can produce one model version.
    
    The stage lifecycle:
    staging → champion (production)
    staging → retired (archived, not used)
    champion → retired (when a new champion is promoted)
    """
    __tablename__ = "model_versions"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))        # "fraud_detector"
    version: Mapped[str] = mapped_column(String(50))      # "v1.2.3" or hash
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(500))
    framework: Mapped[str] = mapped_column(String(50))    # "sklearn", "pytorch"
    metrics: Mapped[dict] = mapped_column(JSON)
    feature_schema: Mapped[dict] = mapped_column(JSON)    # expected input format
    stage: Mapped[str] = mapped_column(String(20), default="staging")
    promoted_by: Mapped[str] = mapped_column(String(100), nullable=True)
```

```python title="app/routers/experiments.py"
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from worker.tasks.training import train_model_task

router = APIRouter(prefix="/experiments", tags=["experiments"])

class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    dataset_path: str
    hyperparams: dict = {}
    tags: dict = {}

@router.post("", status_code=202)
async def run_experiment(
    data: ExperimentCreate,
    db=Depends(get_db),
    current_user=Depends(require_ds_or_admin),
):
    """
    Start a training experiment.
    Returns 202 Accepted immediately — training runs in background.
    
    Why 202 and not 201?
    202 = "I received your request and will process it asynchronously"
    201 = "I created a resource synchronously"
    Training takes minutes/hours, so 202 is correct.
    """
    # Record the experiment in the database first
    experiment = Experiment(
        name=data.name,
        description=data.description,
        dataset_path=data.dataset_path,
        hyperparams=data.hyperparams,
        tags=data.tags,
        status=ExperimentStatus.queued,
        created_by=current_user.username,
    )
    db.add(experiment)
    await db.flush()  # get the ID without committing
    
    # Submit to Celery worker
    task = train_model_task.apply_async(
        kwargs={
            "experiment_id": experiment.id,
            "dataset_path": data.dataset_path,
            "hyperparams": data.hyperparams,
        },
        queue="training",
    )
    
    experiment.celery_task_id = task.id
    experiment.status = ExperimentStatus.running
    await db.commit()
    
    return {
        "experiment_id": experiment.id,
        "job_id": task.id,
        "status": "queued",
        "poll_url": f"/jobs/{task.id}"
    }

@router.get("/{exp_id}/compare")
async def compare_experiments(
    baseline_id: int,
    challenger_id: int,
    db=Depends(get_db),
):
    """Compare metrics between two experiments to decide which to promote."""
    baseline = await db.get(Experiment, baseline_id)
    challenger = await db.get(Experiment, challenger_id)
    
    if not baseline or not challenger:
        raise HTTPException(404, "Experiment not found")
    
    comparison = {}
    for metric in set(list(baseline.metrics.keys()) + list(challenger.metrics.keys())):
        base_val = baseline.metrics.get(metric)
        chal_val = challenger.metrics.get(metric)
        
        if base_val and chal_val:
            delta = round(chal_val - base_val, 4)
            comparison[metric] = {
                "baseline": base_val,
                "challenger": chal_val,
                "delta": delta,
                "improved": delta > 0,  # assumes higher = better
            }
    
    return {
        "baseline": {"id": baseline_id, "name": baseline.name},
        "challenger": {"id": challenger_id, "name": challenger.name},
        "comparison": comparison,
    }
```

---

## Step 2: Model Registry — Tracking All Model Versions

```python title="app/routers/registry.py"
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/models", tags=["model-registry"])

@router.get("/{model_name}/champion")
async def get_production_model(model_name: str, db=Depends(get_db)):
    """
    Get the model currently deployed in production.
    This is the endpoint your inference service calls at startup
    to know which model to load.
    """
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.name == model_name,
            ModelVersion.stage == "champion",
        )
    )
    champion = result.scalar_one_or_none()
    if not champion:
        raise HTTPException(404, f"No champion model for '{model_name}'")
    return champion

@router.post("/{model_id}/promote")
async def promote_model(
    model_id: int,
    target_stage: str,
    db=Depends(get_db),
    current_user=Depends(require_admin),
):
    """
    Promote a model to a new stage.
    
    When promoting to 'champion':
    - The current champion is automatically retired
    - Only admins can do this (it affects production)
    
    Stage transitions:
    staging → champion → retired
    staging → retired (skip production)
    """
    model = await db.get(ModelVersion, model_id)
    if not model:
        raise HTTPException(404, "Model version not found")
    
    if target_stage == "champion":
        # First, retire the current champion
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.name == model.name,
                ModelVersion.stage == "champion",
            )
        )
        current_champion = result.scalar_one_or_none()
        if current_champion:
            current_champion.stage = "retired"
            
    model.stage = target_stage
    model.promoted_by = current_user.username
    await db.commit()
    
    return {
        "message": f"Model {model_id} promoted to {target_stage}",
        "promoted_by": current_user.username
    }
```

---

## Step 3: A/B Testing — Statistically Rigorous Model Comparison

A/B testing lets you compare two model versions on real production traffic and use statistics to determine which is actually better.

```python title="app/services/ab_test_service.py"
import hashlib
import numpy as np

class ABTestService:
    """
    Manages A/B tests between model versions.
    
    How A/B testing works:
    1. You have model A (control, current production)
    2. You want to test model B (treatment, new model)
    3. You route X% of traffic to B, (100-X)% stays with A
    4. Collect outcome metrics (accuracy, revenue, click-through, etc.)
    5. Run statistical test to determine if B is actually better
    """
    
    def assign_variant(self, test_id: int, request_id: str, traffic_split: float) -> str:
        """
        Deterministically assign a request to control or treatment.
        
        Why deterministic? The same request_id always gets the same variant.
        This ensures:
        - User always sees the same model in the same session
        - Results are reproducible and auditable
        - No randomness → consistent user experience
        
        How: MD5 hash of (test_id + request_id) → map to 0-1 range
        If the value < traffic_split → treatment, else → control
        """
        hash_input = f"{test_id}:{request_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        fraction = (hash_value % 10000) / 10000.0  # 0.0000 to 0.9999
        
        return "treatment" if fraction < traffic_split else "control"
    
    async def analyze_results(self, test_id: int) -> dict:
        """
        Run Welch's t-test to determine if there's a statistically significant
        difference between control and treatment outcomes.
        
        Why Welch's t-test?
        - Compares means of two groups
        - Doesn't assume equal variance (robust)
        - Standard choice when comparing model performance metrics
        
        p-value interpretation:
        - p < 0.05: 95% confidence that the difference is real, not random
        - p < 0.01: 99% confidence
        - p >= 0.05: can't conclude which is better (need more data)
        """
        from scipy import stats
        import redis.asyncio as aioredis
        
        redis = aioredis.from_url(settings.redis_url)
        
        # Retrieve stored outcomes from Redis
        control_raw = await redis.lrange(f"ab:{test_id}:control:outcomes", 0, -1)
        treatment_raw = await redis.lrange(f"ab:{test_id}:treatment:outcomes", 0, -1)
        
        control = np.array([float(v) for v in control_raw])
        treatment = np.array([float(v) for v in treatment_raw])
        
        # Need minimum sample size for reliable statistics
        MIN_SAMPLE = 100
        if len(control) < MIN_SAMPLE or len(treatment) < MIN_SAMPLE:
            return {
                "status": "insufficient_data",
                "n_control": len(control),
                "n_treatment": len(treatment),
                "needed": MIN_SAMPLE,
                "message": f"Need at least {MIN_SAMPLE} samples per variant"
            }
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        control_mean = float(control.mean())
        treatment_mean = float(treatment.mean())
        relative_lift = (treatment_mean - control_mean) / abs(control_mean) * 100
        
        is_significant = p_value < 0.05
        winner = None
        if is_significant:
            winner = "treatment" if treatment_mean > control_mean else "control"
        
        return {
            "status": "analyzed",
            "n_control": len(control),
            "n_treatment": len(treatment),
            "control_mean": round(control_mean, 4),
            "treatment_mean": round(treatment_mean, 4),
            "relative_lift_pct": round(relative_lift, 2),
            "p_value": round(float(p_value), 4),
            "is_statistically_significant": is_significant,
            "confidence_level": f"{(1 - p_value) * 100:.1f}%",
            "winner": winner,
            "recommendation": (
                f"Promote treatment model (p={p_value:.4f} < 0.05, lift={relative_lift:.1f}%)"
                if winner == "treatment"
                else "Keep control model" if winner == "control"
                else "Continue collecting data"
            )
        }
```

---

## Step 4: Canary Deployments — Gradual Traffic Shifting

A **canary deployment** gradually shifts traffic from the old model to the new one. You start with 5%, watch for errors, increase to 25%, check again, then 100%. If anything goes wrong, roll back instantly.

```python title="app/services/canary_service.py"
import hashlib
import redis.asyncio as aioredis

class CanaryService:
    """
    Controls how much traffic goes to the canary (new) model vs champion (old).
    
    Configuration is stored in Redis so all API instances share the same state
    and changes take effect immediately without redeployment.
    
    A typical canary rollout:
    Day 1:  5% canary (test with small traffic, watch error rates)
    Day 2: 25% canary (larger sample, still mostly on stable model)
    Day 3: 50% canary (half and half, compare metrics)
    Day 4: 100% canary = promote to champion
    
    Rollback at any point: set canary_pct = 0
    """
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
    
    async def set_canary(self, model_name: str, canary_version: str, canary_pct: float):
        """Configure which version is the canary and how much traffic it gets."""
        redis = aioredis.from_url(self.redis_url)
        await redis.hset(f"canary:{model_name}", mapping={
            "canary_version": canary_version,
            "canary_pct": str(canary_pct),
        })
    
    async def get_active_version(self, model_name: str, request_id: str) -> str:
        """
        Determine which model version should handle this request.
        Returns "champion" or the canary version string.
        """
        redis = aioredis.from_url(self.redis_url)
        config = await redis.hgetall(f"canary:{model_name}")
        
        if not config:
            return "champion"  # no canary configured
        
        canary_pct = float(config.get(b"canary_pct", 0))
        
        if canary_pct <= 0:
            return "champion"
        
        # Deterministic routing based on request_id
        hash_val = int(hashlib.md5(
            f"canary:{model_name}:{request_id}".encode()
        ).hexdigest(), 16)
        fraction = (hash_val % 10000) / 10000.0
        
        if fraction < canary_pct:
            return config[b"canary_version"].decode()
        return "champion"
    
    async def rollback(self, model_name: str):
        """Emergency rollback: immediately send all traffic back to champion."""
        await self.set_canary(model_name, "", 0.0)


canary_service = CanaryService(settings.redis_url)
```

```python title="app/routers/canary.py"
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from services.canary_service import canary_service

router = APIRouter(prefix="/canary", tags=["canary-deployments"])

class CanaryConfig(BaseModel):
    canary_version: str
    canary_pct: float = Field(ge=0.0, le=1.0, description="0.05 = 5% of traffic")

@router.post("/{model_name}/start")
async def start_canary(
    model_name: str,
    config: CanaryConfig,
    _=Depends(require_admin),
):
    """Start a canary deployment — route initial small % to new model."""
    await canary_service.set_canary(model_name, config.canary_version, config.canary_pct)
    return {
        "message": f"Canary deployment started",
        "model": model_name,
        "canary_version": config.canary_version,
        "traffic_pct": f"{config.canary_pct * 100:.0f}%"
    }

@router.post("/{model_name}/ramp")
async def ramp_canary(
    model_name: str,
    new_pct: float = Field(ge=0.0, le=1.0),
    _=Depends(require_admin),
):
    """Increase (or decrease) the canary traffic percentage."""
    redis = aioredis.from_url(settings.redis_url)
    config = await redis.hgetall(f"canary:{model_name}")
    if not config:
        raise HTTPException(404, "No canary deployment found for this model")
    
    canary_version = config[b"canary_version"].decode()
    await canary_service.set_canary(model_name, canary_version, new_pct)
    return {"message": f"Traffic ramped to {new_pct*100:.0f}%"}

@router.post("/{model_name}/rollback")
async def rollback_canary(model_name: str, _=Depends(require_admin)):
    """Emergency rollback — all traffic back to champion immediately."""
    await canary_service.rollback(model_name)
    return {"message": f"Rolled back. 100% of traffic on champion model."}
```

---

## Step 5: Data Drift Detection

**Data drift** occurs when the distribution of incoming data changes from what the model was trained on. A model trained on normal data will perform poorly when production data looks different.

```python title="app/worker/tasks/drift_detection.py"
import numpy as np
from worker.celery_app import celery_app
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "daily-drift-check": {
        "task": "worker.tasks.drift_detection.check_drift",
        "schedule": crontab(hour=6, minute=0),  # 6 AM daily
    }
}

def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    PSI (Population Stability Index) measures how much a distribution has shifted.
    
    Interpretation:
    PSI < 0.10:  No significant shift (stable)
    PSI 0.10-0.25: Moderate shift (investigate)
    PSI > 0.25:  Significant shift (model likely degraded, retrain!)
    
    How it works:
    1. Divide the reference distribution into bins (e.g., 10 buckets)
    2. Count what fraction of each distribution falls in each bucket
    3. PSI = sum over bins of (current% - reference%) × ln(current% / reference%)
    
    Higher PSI = more drift from reference distribution.
    """
    # Create bins from reference distribution
    ref_counts, bin_edges = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to percentages, add small epsilon to avoid log(0)
    ref_pct = (ref_counts / len(reference)) + 1e-8
    cur_pct = (cur_counts / len(current)) + 1e-8
    
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)

@celery_app.task(name="worker.tasks.drift_detection.check_drift")
def check_drift():
    """
    Compare the distribution of today's prediction inputs
    to the training data baseline.
    
    If PSI > 0.25 on any key feature, alert the team.
    """
    from sqlalchemy import create_engine
    import pandas as pd
    
    engine = create_engine(settings.sync_database_url)
    
    # Load training data baseline (computed once when model was trained)
    reference_df = pd.read_csv(settings.reference_data_path)
    
    # Load today's production inputs
    current_df = pd.read_sql("""
        SELECT age, bmi, amount, children
        FROM prediction_inputs
        WHERE created_at >= NOW() - INTERVAL '1 day'
    """, engine)
    
    if len(current_df) < 50:
        return {"status": "skipped", "reason": "insufficient_data"}
    
    features_to_check = ["age", "bmi"]  # key numeric features
    drift_results = {}
    has_significant_drift = False
    
    for feature in features_to_check:
        if feature in reference_df.columns and feature in current_df.columns:
            psi = population_stability_index(
                reference_df[feature].dropna().values,
                current_df[feature].dropna().values,
            )
            
            level = "stable" if psi < 0.10 else "moderate" if psi < 0.25 else "significant"
            drift_results[feature] = {"psi": psi, "level": level}
            
            if psi > 0.25:
                has_significant_drift = True
    
    if has_significant_drift:
        # Alert team (send email, Slack message, PagerDuty, etc.)
        alert_team_about_drift(drift_results)
    
    return {
        "status": "completed",
        "has_drift": has_significant_drift,
        "feature_drift": drift_results,
    }
```

---

## Complete API Reference

```
Experiments
  POST /experiments              → Start training run (202 Accepted)
  GET  /experiments              → List all runs with status
  GET  /experiments/{id}/status  → Detailed status + metrics
  GET  /experiments/compare?baseline_id=1&challenger_id=2 → Compare metrics

Model Registry
  GET  /models/{name}/versions   → All versions + stages
  POST /models/{name}/versions   → Register new version (after training)
  POST /models/{id}/promote      → Change stage (staging→champion)
  GET  /models/{name}/champion   → Get production model

A/B Testing
  POST /ab-tests                 → Create test (control vs treatment)
  POST /ab-tests/{id}/outcome    → Record one prediction outcome
  GET  /ab-tests/{id}/results    → Statistical significance analysis

Canary Deployments
  POST /canary/{model}/start     → Start at X% traffic
  POST /canary/{model}/ramp      → Increase/decrease traffic %
  POST /canary/{model}/promote   → Graduate canary to champion
  POST /canary/{model}/rollback  → Emergency: back to champion

Drift Monitoring
  GET  /drift/latest             → Latest drift check results
  POST /drift/trigger            → Manual drift check
  GET  /drift/history            → Historical PSI scores
```

---

## Key Learnings From This Project

| MLOps Concept | How It's Implemented |
|---------------|---------------------|
| Experiment tracking | `Experiment` ORM, Celery training tasks with status updates |
| Model registry | `ModelVersion` ORM with stage lifecycle (staging→champion→retired) |
| Reproducibility | Every experiment records dataset path, hyperparams, metrics |
| A/B testing | Consistent hashing for deterministic variant assignment |
| Statistical significance | Welch's t-test with p-value and confidence reporting |
| Canary deployments | Redis-stored traffic split, changed without redeployment |
| Data drift detection | PSI (Population Stability Index) on key features |
| Automated retraining | Celery Beat scheduled task, triggered on schedule or drift |
| Least privilege | Admins promote models; data scientists can train but not promote |
| Auditability | All promotions record who did it and when |

---

## The Bigger Picture

This project represents **MLOps Level 3** — the highest maturity level in the MLOps maturity model:

```
Level 0 (Manual): Notebooks → manually shared PKL files
Level 1 (API serving): FastAPI + Docker (Series videos 1-12)
Level 2 (Automated serving): The fraud detection project (Project 1)
Level 3 (Full MLOps): This project — experiment tracking, registry, A/B, canary, drift
```

Most companies operate between Level 1 and Level 2. Reaching Level 3 is what enables truly reliable, continuously improving ML systems.
