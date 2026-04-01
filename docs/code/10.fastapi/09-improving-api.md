---
id: improving-api
title: "09 · Improving the FastAPI API"
sidebar_label: "09 · Improving the API"
sidebar_position: 9
tags: [logging, error-handling, middleware, best-practices, production, fastapi]
---

# Improving the FastAPI API — Production Best Practices

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=M17qwKnmG38) · **Series:** FastAPI for ML – CampusX

---

## Why "Improve" the Basic API?

The insurance premium prediction API from Video 8 works, but it's not production-ready. Here's what a real ML API needs:

```
Basic API (Tutorial)          Production API
─────────────────────         ──────────────────────────────
No logging                →   Structured logging (JSON)
No error details          →   Detailed, consistent error responses
Hardcoded paths           →   Config via environment variables
No health checks          →   /health + /ready endpoints
No request IDs            →   Correlation IDs for tracing
No input logging          →   Audit trail of all predictions
No response time metrics  →   Latency logging per endpoint
Single-file app           →   Modular project structure
```

---

## Structured Logging

```python title="core/logging.py"
import logging
import json
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Outputs logs as JSON — works with Datadog, CloudWatch, ELK."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_dict.update(record.extra)
        return json.dumps(log_dict)

def setup_logging(level: str = "INFO"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = [handler]

logger = logging.getLogger("insurance_api")
```

---

## App Configuration via Environment Variables

```python title="core/config.py"
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Insurance Premium API"
    app_version: str = "1.0.0"
    
    # Model settings
    model_path: str = "artifacts/insurance_model.pkl"
    model_version: str = "1.0.0"
    
    # Security
    api_key: str = ""
    
    # Logging
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

```bash title=".env"
MODEL_PATH=artifacts/insurance_model.pkl
MODEL_VERSION=2.1.0
API_KEY=super-secret-key-here
LOG_LEVEL=INFO
WORKERS=4
```

---

## Middleware — Request/Response Interceptors

Middleware runs on every request, before and after your endpoint:

```python title="middleware.py"
import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.middleware")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Assign a unique ID to every request
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        
        logger.info({
            "event": "request_received",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        })
        
        # Process the request
        response = await call_next(request)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info({
            "event": "request_completed",
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        })
        
        # Attach request ID to response header for tracing
        response.headers["X-Request-ID"] = request_id
        return response
```

```python title="main.py"
app.add_middleware(RequestLoggingMiddleware)
```

---

## Custom Exception Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean validation errors to clients."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"][1:]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning({
        "event": "validation_error",
        "path": request.url.path,
        "errors": errors
    })
    
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": errors}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors — don't leak stack traces."""
    logger.error({
        "event": "unhandled_exception",
        "error": str(exc),
        "path": request.url.path
    }, exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Contact support"}
    )
```

---

## Improved Health & Readiness Endpoints

```python
from datetime import datetime, timezone

startup_time = datetime.now(timezone.utc)

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — is the service running?"""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/ready", tags=["ops"])
def readiness():
    """Readiness probe — is the service ready to accept traffic?"""
    checks = {
        "model_loaded": "model" in model_store,
        "uptime_seconds": (datetime.now(timezone.utc) - startup_time).seconds
    }
    
    all_ready = all(checks[k] for k in ["model_loaded"])
    
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={"status": "ready" if all_ready else "not_ready", "checks": checks}
    )
```

---

## Improved Prediction Endpoint with Logging

```python
@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict(data: InsuranceInput, request: Request):
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    logger.info({
        "event": "prediction_request",
        "request_id": request_id,
        "input": data.model_dump()
    })
    
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        input_df = pd.DataFrame([data.model_dump()])
        model = model_store["model"]
        prediction = model.predict(input_df)[0]
        confidence = float(max(model.predict_proba(input_df)[0]))
        
        result = PredictionOutput(
            prediction=prediction,
            confidence=round(confidence, 4),
            model_version=settings.model_version
        )
        
        logger.info({
            "event": "prediction_success",
            "request_id": request_id,
            "prediction": result.prediction,
            "confidence": result.confidence
        })
        
        return result
    
    except Exception as e:
        logger.error({
            "event": "prediction_failed",
            "request_id": request_id,
            "error": str(e)
        }, exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed")
```

---

## Full Project Structure After Improvements

```
insurance-api/
├── main.py                      # FastAPI app creation + routers
├── routers/
│   ├── predict.py               # /predict endpoints
│   └── ops.py                   # /health, /ready, /model-info
├── core/
│   ├── config.py                # Settings (pydantic-settings)
│   ├── logging.py               # JSON logging setup
│   └── security.py              # API key validation
├── middleware/
│   └── logging_middleware.py    # Request/response logging
├── schemas/
│   ├── input.py                 # InsuranceInput
│   └── output.py                # PredictionOutput
├── services/
│   └── model_service.py         # Model loading + inference
├── artifacts/
│   └── insurance_model.pkl      # Trained model
├── .env                         # Environment variables
├── .env.example                 # Template for new devs
├── requirements.txt
└── Dockerfile
```

---

## Topics Not Covered in the Video

### Dependency Injection — Reusable Components

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_scheme = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_scheme)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Apply to any endpoint
@app.post("/predict")
def predict(data: InsuranceInput, _: str = Depends(verify_api_key)):
    ...

# Apply to entire router
router = APIRouter(dependencies=[Depends(verify_api_key)])
```

### Prometheus Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
# Exposes /metrics endpoint for Prometheus scraping
```

---

## Q&A

**Q: Where should I put logging — in the endpoint or middleware?**
> Both, for different reasons. Middleware logs request/response metadata (method, path, latency, status) for every request. Endpoint logging captures domain-specific data (what was predicted, which patient ID was accessed). Use middleware for cross-cutting concerns and endpoint-level logging for business logic audit trails.

**Q: What's the difference between `/health` and `/ready`?**
> `/health` (liveness) answers: "Is the process alive?" — Kubernetes restarts the pod if this fails. `/ready` (readiness) answers: "Can this pod handle traffic?" — Kubernetes stops sending traffic if this fails. Your model might still be loading (not ready) even if the process is alive.

**Q: Should I use `print()` or `logging` for debug output?**
> Always use `logging`. It supports log levels (DEBUG, INFO, WARNING, ERROR), structured formatting (JSON), log aggregation (CloudWatch, Datadog), and can be enabled/disabled via config without changing code. `print()` is fine in notebooks, not in APIs.

**Q: What is a correlation ID?**
> A unique ID (UUID) attached to a request that flows through all services that handle it. When a client gets a 500 error, they include the `X-Request-ID` when reporting the bug. You can search your logs by that ID to see the exact trace of what happened.

**Q: How do I test middleware?**
> Use FastAPI's `TestClient`:
> ```python
> from fastapi.testclient import TestClient
> client = TestClient(app)
> response = client.post("/predict", json={...})
> assert "X-Request-ID" in response.headers
> ```
