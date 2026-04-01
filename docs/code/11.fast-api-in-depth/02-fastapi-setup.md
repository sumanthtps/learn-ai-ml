---
id: fastapi-setup
title: "02 · FastAPI Philosophy & Setup"
sidebar_label: "02 · FastAPI Setup"
sidebar_position: 2
tags: [fastapi, uvicorn, asgi, wsgi, async, setup, beginner]
---

# FastAPI Philosophy & Setup

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=lXx-_1r0Uss) · **Series:** FastAPI for ML – CampusX

---

## Concept Flow

```mermaid
flowchart LR
  Client[Client Request] --> Uvicorn[Uvicorn ASGI Server]
  Uvicorn --> FastAPI[FastAPI App]
  FastAPI --> Pydantic[Pydantic Validation]
  Pydantic --> Endpoint[Route Function]
  Endpoint --> Response[JSON Response]
```

## Why FastAPI and Not Flask?

Before installing anything, understand *why* FastAPI exists. When ML engineers started building APIs, they mostly used Flask — the dominant Python web framework since 2010. Flask works. But it has real problems for modern ML workloads.

### Problem 1: No Data Validation

In Flask, when a client sends `{"age": "thirty-five"}` instead of `{"age": 35}`, Flask happily passes that string to your code. Your ML model then fails with a confusing `TypeError: unsupported operand type(s) for +: 'str' and 'int'` deep inside sklearn. The client receives a generic 500 error with no useful information about what they did wrong.

FastAPI automatically validates every field. Invalid data produces a detailed 422 error **before your function even runs** — listing exactly which fields failed and why.

### Problem 2: Synchronous Architecture (WSGI)

Flask uses WSGI — an older synchronous protocol. It handles one request at a time per worker process. When a request arrives and the worker is busy (waiting for a database query), the new request waits in a queue. This is like a bank with one cashier who serves customers one at a time:

```
Flask (WSGI) — 3 simultaneous requests, 1 worker:

Request A → [======== processing (100ms) ========] → Response A
Request B →                    [======== waiting in queue...  (100ms+ wait!) ========]
Request C →                              [waiting...]

Total time for 3 requests: 300ms
```

FastAPI uses ASGI — a modern async protocol. One worker handles thousands of concurrent requests by switching between them while waiting for I/O:

```
FastAPI (ASGI) — 3 simultaneous requests, 1 worker:

Request A → [process] → [await DB...] → [process] → Response A
Request B →        [process] → [await model...] → Response B  
Request C →             [process] ──────────────────────────► Response C

All 3 complete in roughly 100ms total (the longest single operation)
```

### Problem 3: No Automatic Documentation

In Flask, you write code, then write separate documentation, and they immediately start diverging. FastAPI generates interactive Swagger UI and ReDoc documentation **automatically from your code** — always accurate, never stale.

### The Comparison Table

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Release year | 2010 | 2018 |
| Architecture | WSGI (synchronous) | ASGI (asynchronous) |
| Input validation | Manual (or marshmallow) | Automatic (Pydantic) |
| API documentation | Manual | Auto-generated |
| Type hints | Optional, unused | Core to the framework |
| Performance | Moderate | High (near Node.js) |
| ML industry adoption | Legacy projects | Current standard |

FastAPI was built by Sebastián Ramírez specifically for modern Python with type hints, async, and ML use cases in mind.

---

## Understanding ASGI vs WSGI — The Core Architecture

### The Bank Analogy

**WSGI (Flask) = Old-style bank with one cashier:**
Each customer must be fully served before the next one starts. If a customer says "hold on, I need to check my account" — the cashier waits. Everyone behind them waits. The cashier is doing nothing useful while waiting.

**ASGI (FastAPI) = Modern bank with one multi-tasking cashier:**
When customer A says "hold on, checking my account" — the cashier immediately turns to customer B. When customer A's account is ready, the cashier returns to them. One person serves multiple customers simultaneously by being efficient with wait time.

### What Happens Inside the Server

```
Client 1 ──POST /predict──►  [Uvicorn: receive HTTP request]
                                         │
                              [FastAPI: parse JSON body]
                                         │
                              [Pydantic: validate fields]
                                         │
                              [Your function: call model.predict()]
                                         │ (this takes 5ms)
                                         │ Event loop serves Client 2 here!
                              [FastAPI: serialize result to JSON]
                                         │
Client 1 ◄──200 OK { prediction: "high" }─┘
```

The key: `await` is the keyword that tells Python "while waiting for this, go handle other work."

### What is Uvicorn?

FastAPI is the framework (the recipe). Uvicorn is the web server (the kitchen).

```
Without Uvicorn: You have a recipe but no stove.
With Uvicorn:    The stove runs continuously, receiving HTTP connections
                 from the internet, parsing them, feeding them to FastAPI,
                 and sending FastAPI's responses back.

Command: uvicorn main:app
         │       │   │
         │       │   └── the FastAPI object named 'app' in main.py
         │       └────── the Python file (main.py)
         └────────────── the ASGI server to use
```

Uvicorn is built on `uvloop` — an ultra-fast event loop written in C — which is why FastAPI achieves near-Node.js performance in Python.

---

## Installation — Step by Step

### Step 1: Create a Virtual Environment

A virtual environment isolates your project's packages from system Python, preventing version conflicts between projects.

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate it (you must do this in every new terminal session)
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows (Command Prompt)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

# Confirmation: your prompt now shows (.venv)
(.venv) $ 
```

**Why bother?** If project A needs `scikit-learn==1.4` and project B needs `scikit-learn==1.5`, they conflict in the same Python. Virtual environments give each project its own isolated package space.

### Step 2: Install FastAPI

```bash
pip install "fastapi[standard]"
```

What `fastapi[standard]` installs:
- `fastapi` — the web framework
- `uvicorn[standard]` — ASGI web server with extra performance features
- `pydantic` — data validation (FastAPI's backbone)
- `python-multipart` — for form and file upload support
- `email-validator` — for `EmailStr` Pydantic type

### Step 3: Verify

```bash
python -c "import fastapi; print(f'FastAPI {fastapi.__version__} installed')"
# FastAPI 0.115.0 installed
```

---

## Your First FastAPI Application — Every Line Explained

```python title="main.py"
from fastapi import FastAPI      # Line 1

app = FastAPI()                  # Line 2

@app.get("/")                    # Line 3
def root():                      # Line 4
    return {"message": "Hello"} # Line 5
```

**Line 1 — `from fastapi import FastAPI`**  
Import the `FastAPI` class. This is the central object that will hold your entire application — all routes, middleware, and configuration.

**Line 2 — `app = FastAPI()`**  
Create one instance of your application. By convention it's called `app`. When you run `uvicorn main:app`, the `main` is the filename and `app` is this object. All your routes attach to this object.

**Line 3 — `@app.get("/")`**  
This is a **decorator** — Python syntax that wraps the function below it. `@app.get("/")` means: "register the function below as a handler for HTTP GET requests to the path `/`." The decorator does two things:
1. Tells FastAPI which HTTP method + URL path this function handles
2. Registers it in FastAPI's internal routing table

**Line 4 — `def root():`**  
A regular Python function. FastAPI calls this function whenever a `GET /` request arrives. You can name it anything — FastAPI uses the decorator, not the function name, for routing.

**Line 5 — `return {"message": "Hello"}`**  
Return a Python dictionary. FastAPI **automatically converts this to JSON** and sets the `Content-Type: application/json` header. You never call `json.dumps()` manually. The client receives: `{"message": "Hello"}` with HTTP status 200.

### Run It

```bash
uvicorn main:app --reload
```

The `--reload` flag watches for file changes and auto-restarts the server. Perfect for development — save a file, see changes instantly. **Never use `--reload` in production** — it wastes CPU watching files that never change in production.

You'll see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Open http://localhost:8000 — you see `{"message": "Hello"}`. You built an API.

---

## The Auto-Generated Docs — FastAPI's Killer Feature

FastAPI generates two documentation interfaces automatically:

**http://localhost:8000/docs — Swagger UI**

An interactive web UI where you can:
- See all your endpoints and their expected inputs/outputs
- Try them out from the browser — fill in fields and click Execute
- See the actual HTTP request and response

This is your primary debugging tool during development.

**http://localhost:8000/redoc — ReDoc**

A clean, readable documentation format. Good for sharing with stakeholders or creating client documentation.

**http://localhost:8000/openapi.json — OpenAPI Spec**

Machine-readable JSON description of your entire API. Postman, code generators, and SDK builders consume this.

The best part: **you write zero documentation**. FastAPI generates everything from your Python code — your function names, type hints, docstrings, and Pydantic models all become documentation automatically.

---

## A More Realistic App — Full Explanation

```python title="main.py"
from fastapi import FastAPI

# ─── App configuration ────────────────────────────────────────────
# These metadata fields appear in your Swagger UI header
app = FastAPI(
    title="Patient Management API",
    description="API for managing patient records in the clinic system",
    version="1.0.0",
    docs_url="/docs",       # where Swagger UI lives (default)
    redoc_url="/redoc",     # where ReDoc lives (default)
)

# ─── In-memory data (for learning only) ──────────────────────────
# We'll replace this with a real database in the advanced topics
patients_db = {
    "P001": {"id": "P001", "name": "Ravi Kumar", "age": 35},
    "P002": {"id": "P002", "name": "Priya Patel", "age": 28},
}

# ─── Root endpoint ────────────────────────────────────────────────
@app.get("/")
def root():
    """
    The docstring becomes the endpoint description in Swagger UI.
    Returns basic API information — a common convention.
    """
    return {
        "name": "Patient Management API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

# ─── Health check endpoint ────────────────────────────────────────
# This endpoint is called by:
# - Load balancers (to check if this server is alive)
# - Monitoring systems (to alert when the service goes down)
# - Kubernetes (to decide whether to restart the pod)
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── Data endpoint ────────────────────────────────────────────────
@app.get("/patients")
def get_patients():
    # FastAPI converts the dict to JSON automatically
    # Returns all patients as a JSON object
    return patients_db
```

---

## Sync vs Async — When to Use Each

FastAPI supports both. Choosing correctly matters for performance.

### `def` (synchronous) — For CPU-Bound Work

```python
@app.post("/predict")
def predict(data: InputData):
    # ML inference is CPU-heavy — it runs on the CPU, not waiting for I/O
    # Using regular 'def' is correct.
    # FastAPI runs this in a thread pool automatically (doesn't block event loop)
    result = model.predict([[data.age, data.bmi]])
    return {"prediction": result[0]}
```

### `async def` — For I/O-Bound Work

```python
@app.get("/patients/{id}")
async def get_patient(id: str):
    # Database queries are I/O-bound — CPU is idle while waiting for DB
    # 'async def' + 'await' lets the event loop handle other requests during the wait
    patient = await database.fetch_one(id)
    return patient
```

### The Decision Rule

```
Is your code waiting for something external?
(database, HTTP calls to other services, file reads)
→ Use async def + await

Is your code doing heavy computation?
(ML inference, image processing, math)
→ Use regular def
```

**Common mistake:** Using `async def` for ML inference. ML inference is CPU-bound. Running it in `async def` blocks the event loop, making your API handle requests sequentially — slower than using `def` which FastAPI runs in a thread pool.

---

## Project Structure for Real Applications

A single `main.py` works for learning but not for a 50-endpoint production API. Here's the structure you'll grow into:

```
my-ml-api/
│
├── main.py                  ← App creation + router registration
│
├── routers/                 ← One file per feature area
│   ├── patients.py          ← GET/POST/PUT/DELETE /patients
│   ├── predictions.py       ← POST /predict
│   └── ops.py               ← GET /health, /ready, /metrics
│
├── schemas/                 ← Pydantic models (validation + serialization)
│   ├── patient.py           ← PatientCreate, PatientResponse
│   └── prediction.py        ← PredictionInput, PredictionOutput
│
├── services/                ← Business logic (no HTTP concerns here)
│   └── ml_service.py        ← Model loading, inference, preprocessing
│
├── core/
│   ├── config.py            ← Settings from environment variables
│   └── database.py          ← Database engine + session factory
│
├── models/                  ← ORM models (database table definitions)
│   └── patient.py
│
├── artifacts/
│   └── model.pkl            ← Your trained ML model
│
├── .env                     ← Secrets (NEVER commit to git!)
├── .env.example             ← Template for other developers
├── requirements.txt
└── Dockerfile
```

### Using APIRouter to Split Endpoints

```python title="routers/patients.py"
from fastapi import APIRouter

# A router is a mini-FastAPI — it has the same decorator API
# prefix="/patients" means all routes here start with /patients
# tags=["patients"] groups them in Swagger UI
router = APIRouter(prefix="/patients", tags=["patients"])

@router.get("")          # This becomes GET /patients
def list_patients():
    ...

@router.get("/{id}")     # This becomes GET /patients/{id}
def get_patient(id: str):
    ...
```

```python title="main.py"
from fastapi import FastAPI
from routers import patients, predictions, ops

app = FastAPI(title="ML API")

# Register all routers with the main app
app.include_router(patients.router)
app.include_router(predictions.router)
app.include_router(ops.router)
```

Clean, organized, easy to navigate.

---

## Model Loading — The Most Important Pattern

**Never load your ML model inside an endpoint function.** This is the single biggest performance mistake in ML APIs.

```python
# ❌ WRONG — model loads on EVERY request (adds 200-500ms each time)
@app.post("/predict")
def predict(data: InputData):
    model = joblib.load("model.pkl")  # ← this is the problem
    return model.predict(...)

# ✅ CORRECT — load once at startup, reuse forever
from contextlib import asynccontextmanager
import joblib

model_store = {}  # module-level dict to hold the model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything before yield: runs at startup (once)
    print("Loading model...")
    model_store["model"] = joblib.load("artifacts/model.pkl")
    print("Model loaded!")
    
    yield  # ← the app runs here, serving all requests
    
    # Everything after yield: runs at shutdown (cleanup)
    model_store.clear()
    print("Model unloaded.")

app = FastAPI(lifespan=lifespan)  # connect lifespan to app

@app.post("/predict")
def predict(data: InputData):
    model = model_store["model"]  # instant dict lookup — no disk I/O
    return {"prediction": str(model.predict([[data.age, data.bmi]])[0])}
```

The `lifespan` context manager is the correct modern FastAPI pattern (FastAPI 0.93+). It replaces the older `@app.on_event("startup")` decorator.

---

## Configuration — Using Environment Variables

```python title="core/config.py"
from pydantic_settings import BaseSettings  # pip install pydantic-settings
from functools import lru_cache

class Settings(BaseSettings):
    """
    All application config in one place.
    Values are read from environment variables or .env file.
    
    Why not hardcode? Hardcoded values:
    - Get committed to git (secrets leak)
    - Require code changes to configure (no separate dev/prod config)
    - Break 12-factor app principles
    """
    app_name: str = "ML API"
    debug: bool = False
    model_path: str = "artifacts/model.pkl"
    model_version: str = "1.0.0"
    api_key: str = ""           # empty = auth disabled
    database_url: str = "postgresql://localhost/mldb"
    
    class Config:
        env_file = ".env"       # read from .env file in development

@lru_cache()                    # call this function only once, cache the result
def get_settings() -> Settings:
    return Settings()
```

```bash title=".env"
# Local development — never commit this file!
DEBUG=true
MODEL_PATH=artifacts/model_v2.pkl
API_KEY=dev-secret-key
DATABASE_URL=postgresql://postgres:password@localhost/mldb
```

On your production server, set these as real environment variables — no `.env` file needed.

---

## Running FastAPI: Development vs Production

```bash
# Development — auto-reload, single worker, verbose logging
uvicorn main:app --reload --port 8000

# Production — multiple workers, all interfaces, no auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Production with Gunicorn managing workers (most robust)
gunicorn main:app \
  -w 4 \                              # 4 worker processes
  -k uvicorn.workers.UvicornWorker \  # each worker is async Uvicorn
  --timeout 120 \                     # request timeout in seconds
  --bind 0.0.0.0:8000
```

**Worker count formula:** `2 × CPU_cores + 1`  
4-core machine → 9 workers. But for ML APIs with large models, fewer workers saves memory (each worker loads its own model copy).

---

## Q&A

**Q: What exactly is an "endpoint"?**

An endpoint is one specific combination of HTTP method + URL path that maps to one function. `GET /patients` is an endpoint. `POST /patients` is a *different* endpoint (same URL, different method). `GET /patients/P001` is yet another. Each does something distinct and maps to its own function in your code.

**Q: Why use `--host 0.0.0.0` in production?**

By default, Uvicorn binds to `127.0.0.1` (localhost), only accepting connections from the same machine. In a Docker container or cloud server, you need to accept connections from anywhere — `0.0.0.0` means "listen on all network interfaces." Without this, your API is inaccessible from outside the server.

**Q: What's the difference between `FastAPI()` and `APIRouter()`?**

`FastAPI()` is the main application — you create one, it's the root. `APIRouter()` is a sub-application for organizing routes — you create many, then attach them to the main app with `app.include_router()`. Think of FastAPI as the company and APIRouter as departments.

**Q: Can I have multiple FastAPI apps in one project?**

Yes, and you can mount them on each other using `app.mount()`. But for most projects, one `FastAPI()` instance with multiple `APIRouter()` sub-routers is the right pattern.

**Q: Is FastAPI production-ready?**

Yes. FastAPI is used in production by Netflix, Uber, Microsoft, and hundreds of ML companies. It's one of the highest-starred Python repositories on GitHub. The library is stable, well-tested, and actively maintained.

---

## Cheat Sheet

```bash
# Install
pip install "fastapi[standard]"

# Minimal app (5 lines)
# main.py: from fastapi import FastAPI; app = FastAPI()
# @app.get("/"); def root(): return {"hello": "world"}

# Run development server
uvicorn main:app --reload

# Run production server (4 workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# View documentation
open http://localhost:8000/docs      # Swagger UI
open http://localhost:8000/redoc     # ReDoc
```
