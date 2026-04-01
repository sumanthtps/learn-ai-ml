---
id: fastapi-setup
title: "02 · FastAPI Philosophy & Setup"
sidebar_label: "02 · FastAPI Philosophy & Setup"
sidebar_position: 2
tags: [fastapi, uvicorn, asgi, wsgi, async, swagger, setup]
---

# FastAPI Philosophy & Setup

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=lXx-_1r0Uss) · **Series:** FastAPI for ML – CampusX

---

## Why FastAPI?

| Feature | Flask (WSGI) | FastAPI (ASGI) |
|---------|-------------|----------------|
| **Performance** | Synchronous, blocking | Async, non-blocking |
| **Input Validation** | Manual / marshmallow | Automatic via Pydantic |
| **API Docs** | Manual Swagger setup | Auto-generated at `/docs` |
| **Type Safety** | None | Full Python type hints |
| **Concurrency** | Single request at a time | Thousands of concurrent requests |
| **Industry Adoption** | Legacy ML projects | Modern ML platforms |

> FastAPI is built on **Starlette** (web framework) + **Pydantic** (data validation) + **Uvicorn** (ASGI server).

---

## The Request Flow: From Client to Model

Understanding this flow is crucial for debugging production issues.

```
Client
  │
  │  HTTP Request (JSON over TCP)
  ▼
Web Server (Uvicorn)
  │
  │  ASGI interface (converts HTTP ↔ Python dicts)
  ▼
FastAPI App (your Python code)
  │
  │  validated Python objects
  ▼
ML Model / Business Logic
  │
  │  Python result
  ▼
FastAPI (serialize to JSON)
  │
  ▼
Web Server → HTTP Response → Client
```

### WSGI vs ASGI

| | WSGI | ASGI |
|-|------|------|
| **Standard** | Older (Flask, Django) | Modern (FastAPI, Django Channels) |
| **Concurrency** | Synchronous — one request at a time | Asynchronous — thousands concurrently |
| **Protocol** | HTTP only | HTTP + WebSockets + HTTP/2 |
| **Server** | Gunicorn | Uvicorn |

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows

# Install FastAPI + Uvicorn
pip install fastapi uvicorn[standard]

# For ML projects (full stack)
pip install fastapi uvicorn[standard] pydantic scikit-learn joblib
```

---

## Your First FastAPI App

```python title="main.py"
from fastapi import FastAPI

# Create the app instance
app = FastAPI(
    title="My ML API",
    description="Insurance Premium Prediction API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

### Run the server

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload

# Production (multiple workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Auto-generated documentation

Once running, FastAPI auto-generates two docs interfaces:

| URL | Interface | Purpose |
|-----|-----------|---------|
| `http://localhost:8000/docs` | **Swagger UI** | Interactive — test endpoints in browser |
| `http://localhost:8000/redoc` | **ReDoc** | Readable — share with stakeholders |
| `http://localhost:8000/openapi.json` | **OpenAPI spec** | Machine-readable — import into Postman |

---

## Understanding the `app` Object

```python
app = FastAPI(
    title="Insurance Premium API",           # Appears in Swagger UI
    description="Predicts premium category", # Shown in docs
    version="2.1.0",                         # API version shown
    docs_url="/docs",                        # Swagger UI path (default)
    redoc_url="/redoc",                      # ReDoc path (default)
    openapi_url="/openapi.json"              # OpenAPI spec path
)
```

---

## Decorators — How Endpoints Are Defined

Every FastAPI endpoint is a Python function decorated with an HTTP method:

```python
@app.get("/items")          # Read — fetch data
@app.post("/items")         # Create — add new data
@app.put("/items/{id}")     # Update — replace entire resource
@app.patch("/items/{id}")   # Partial update — change a field
@app.delete("/items/{id}")  # Delete — remove resource
```

---

## Sync vs Async Endpoints

```python
# Synchronous — blocks the thread while running
# Use when calling CPU-bound tasks (ML inference)
@app.post("/predict")
def predict(data: PredictInput):
    result = model.predict(data.features)
    return {"prediction": result}

# Asynchronous — releases thread while waiting
# Use when calling databases, external APIs, file I/O
@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    patient = await db.fetch(patient_id)  # non-blocking DB call
    return patient
```

> **Rule of thumb for ML APIs:** Use `def` (sync) for CPU-bound inference, `async def` for I/O-bound operations like DB queries.

---

## Project Structure (Best Practice)

```
my-ml-api/
├── main.py                 # App entry point
├── routers/
│   ├── predict.py          # /predict endpoints
│   └── health.py           # /health endpoints
├── models/
│   └── schemas.py          # Pydantic input/output schemas
├── services/
│   └── ml_model.py         # Model loading + inference
├── core/
│   └── config.py           # App settings (env vars)
├── artifacts/
│   └── model.pkl           # Trained model file
├── requirements.txt
├── Dockerfile
└── .env
```

---

## Topics Not Covered in the Video

### Application Startup / Shutdown Events

```python
from contextlib import asynccontextmanager
import joblib

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once
    ml_model["classifier"] = joblib.load("artifacts/model.pkl")
    print("Model loaded")
    yield
    # Shutdown: cleanup
    ml_model.clear()
    print("Model unloaded")

app = FastAPI(lifespan=lifespan)
```

> **Important:** Never load your model inside an endpoint function — it will be reloaded on every request, causing massive latency.

### APIRouter — Splitting a Large App

```python title="routers/predict.py"
from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/insurance")
def predict_insurance(data: InsuranceInput):
    ...

@router.post("/health-risk")
def predict_health_risk(data: HealthInput):
    ...
```

```python title="main.py"
from routers import predict, health

app = FastAPI()
app.include_router(predict.router)
app.include_router(health.router)
```

### CORS — Allow Frontend to Call Your API

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myfrontend.com"],  # or ["*"] for dev
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Environment Variables & Settings

```python title="core/config.py"
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "artifacts/model.pkl"
    api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Q&A

**Q: What is Uvicorn and why do I need it separately?**
> FastAPI is just a web framework — it defines your routes and validation logic. Uvicorn is the ASGI-compatible web server that actually listens on a port and handles HTTP connections. You need both. The command `uvicorn main:app` tells Uvicorn to run the `app` object from `main.py`.

**Q: What does `--reload` do and should I use it in production?**
> `--reload` watches for file changes and automatically restarts the server — essential during development. **Never use `--reload` in production.** It adds overhead and is a security risk. Use `--workers N` instead for production.

**Q: How many workers should I run in production?**
> A common formula is `2 × CPU_cores + 1`. On a 4-core machine: `2 × 4 + 1 = 9 workers`. For ML models, fewer workers with larger RAM per worker is often better to avoid loading the model multiple times.

**Q: Is FastAPI faster than Flask?**
> Yes, significantly. FastAPI + Uvicorn typically handles 3–5× more requests per second than Flask + Gunicorn in benchmarks, primarily because of the async ASGI architecture.

**Q: Can I use FastAPI with Gunicorn?**
> Yes. For production, use Gunicorn as the process manager with Uvicorn workers: `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000`

**Q: What's the difference between `title`, `description`, and `version` in `FastAPI()`?**
> These are purely cosmetic — they appear in your auto-generated Swagger UI. Use `version` to communicate breaking API changes to clients.

---

## Cheat Sheet

```bash
# Install
pip install "fastapi[standard]"

# Run dev server
uvicorn main:app --reload --port 8000

# Run production server (4 workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Run with Gunicorn (recommended for production)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# View docs
open http://localhost:8000/docs
open http://localhost:8000/redoc
```
