---
id: performance-optimization
title: "20 · Performance Optimization — Profiling, Caching, Concurrency"
sidebar_label: "20 · Performance"
sidebar_position: 20
tags: [performance, caching, redis, profiling, async, optimization, advanced]
---

# Performance Optimization — Profiling, Caching, Concurrency

> **Advanced Topic** — Making your ML API handle 10× more traffic with the same infrastructure.

---

## Performance Thinking: Measure Before You Optimize

A common mistake: spending a week optimizing the wrong thing. Always **profile first**, optimize second.

```
The Optimization Loop:

  1. MEASURE baseline (throughput, latency)
         ↓
  2. PROFILE (find the actual bottleneck)
         ↓
  3. OPTIMIZE the bottleneck
         ↓
  4. MEASURE again (verify improvement)
         ↓
  Back to 1 if needed
```

**Throughput**: How many requests per second can your API handle?
**Latency**: How long does one request take? (P50, P95, P99 — the 50th, 95th, 99th percentile)

---

## Baseline Measurement

```bash
# Measure baseline with 'hey' (simple HTTP benchmark)
pip install hey  # or: brew install hey

hey -n 1000 -c 50 -m POST \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":27.9,"children":2,"smoker":"no","region":"southeast"}' \
  http://localhost:8000/predict

# Output example:
# Requests/sec: 342.45        ← throughput
# Average: 0.1462 secs        ← avg latency
# 50th percentile: 0.1456     ← P50 (median)
# 95th percentile: 0.2134     ← P95
# 99th percentile: 0.3891     ← P99
```

Target numbers for a well-optimized single-instance ML API:
- Throughput: > 500 req/sec
- P50 latency: < 20ms
- P99 latency: < 200ms

---

## Optimization 1: Model Loading (The Biggest Win)

This is the most impactful optimization and the most common mistake. Never load the model inside an endpoint function.

```python
# ❌ CATASTROPHICALLY WRONG — loads model on every request
@app.post("/predict")
def predict(data: InsuranceInput):
    model = joblib.load("artifacts/model.pkl")  # ← 200-500ms per request!
    return model.predict(...)

# ✅ CORRECT — load once at startup, reuse forever
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["model"] = joblib.load("artifacts/model.pkl")  # once at startup
    yield

@app.post("/predict")
def predict(data: InsuranceInput):
    model = model_store["model"]  # instant dict lookup
    return model.predict(...)
```

**Impact**: Transforms 300ms+ latency into 5-10ms latency. This is not premature optimization — it's correctness.

---

## Optimization 2: Redis Caching for Identical Inputs

If users frequently send the same features (same age, bmi, region, etc.), run inference once and cache the result.

```python
import redis.asyncio as aioredis
import hashlib
import json
from typing import Optional

class PredictionCache:
    """Redis-backed cache for prediction results."""
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
    
    def _make_key(self, features: dict) -> str:
        """
        Create a deterministic cache key from the input features.
        
        We sort keys before hashing so {"age": 35, "bmi": 27} and
        {"bmi": 27, "age": 35} produce the same key.
        
        MD5 is fine here (not for security, just for key generation).
        """
        canonical = json.dumps(features, sort_keys=True)
        return f"prediction:{hashlib.md5(canonical.encode()).hexdigest()}"
    
    async def get(self, features: dict) -> Optional[dict]:
        """Return cached prediction or None if not cached."""
        key = self._make_key(features)
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None
    
    async def set(self, features: dict, result: dict, ttl: int = 3600):
        """Cache a prediction result for ttl seconds."""
        key = self._make_key(features)
        await self.redis.setex(key, ttl, json.dumps(result))

# Initialize once at startup
prediction_cache: Optional[PredictionCache] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_cache
    prediction_cache = PredictionCache(settings.redis_url)
    model_store["model"] = joblib.load(settings.model_path)
    yield

# Use in endpoint
@app.post("/predict")
async def predict(data: InsuranceInput):
    features = data.model_dump()
    
    # 1. Check cache first
    cached = await prediction_cache.get(features)
    if cached:
        cached["cache_hit"] = True
        return cached
    
    # 2. Run inference (cache miss)
    result = run_inference(features)
    
    # 3. Cache the result for 1 hour
    await prediction_cache.set(features, result, ttl=3600)
    result["cache_hit"] = False
    return result
```

**When caching helps most**: When the same input appears frequently. For discrete features (smoker: yes/no, region: 4 values), the number of unique combinations is limited, so cache hit rate can be very high.

**When caching doesn't help**: For continuous features (bmi=27.9123...), every input is unique — cache is always a miss. In this case, skip caching.

---

## Optimization 3: Async Inference with ThreadPoolExecutor

scikit-learn's `model.predict()` is synchronous and CPU-bound. If you use `async def` in FastAPI but then call synchronous CPU-bound code, you block the async event loop — preventing other requests from being served.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Create a thread pool for ML inference
# max_workers = number of CPU cores (or less if you want to leave headroom)
inference_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml-inference")

def _run_inference_sync(features: dict) -> dict:
    """
    Pure synchronous function that runs in the thread pool.
    Must be thread-safe (model.predict() is thread-safe in scikit-learn).
    """
    model = model_store["model"]
    df = pd.DataFrame([features])
    prediction = str(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    classes = model_store["classes"]
    probabilities = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    return {
        "prediction": prediction,
        "confidence": probabilities[prediction],
        "probabilities": probabilities,
    }

@app.post("/predict")
async def predict(data: InsuranceInput):
    """
    async def + run_in_executor: 
    The inference runs in a thread pool without blocking the event loop.
    While thread is computing inference, the event loop handles other requests.
    """
    loop = asyncio.get_running_loop()
    
    result = await loop.run_in_executor(
        inference_pool,         # which thread pool to use
        _run_inference_sync,    # which function to run
        data.model_dump()       # argument to the function
    )
    
    result["model_version"] = settings.model_version
    return result
```

**Impact**: With 4 CPU cores and 4 thread pool workers, you can handle 4 inference requests truly simultaneously instead of sequentially.

---

## Optimization 4: Uvicorn Workers and Gunicorn

Uvicorn's `--workers N` flag creates N independent processes. Each loads the model separately and handles requests independently — true parallelism.

```bash
# Development: 1 worker, auto-reload
uvicorn main:app --reload

# Production: multiple workers
# Formula: 2 × CPU_cores + 1
# For 4-core machine: 9 workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 9

# More robust: Gunicorn manages processes
gunicorn main:app \
  -w 4 \                              # 4 worker processes
  -k uvicorn.workers.UvicornWorker \  # each worker is async Uvicorn
  --timeout 120 \                     # request timeout
  --graceful-timeout 30 \             # wait 30s for in-flight requests on shutdown
  --keep-alive 5 \                    # keep-alive timeout
  --bind 0.0.0.0:8000
```

**Memory warning**: Each worker loads its own copy of the model. A 500MB Random Forest × 9 workers = 4.5GB RAM. Adjust worker count to fit available memory.

---

## Optimization 5: Response Compression

```python
from fastapi.middleware.gzip import GZipMiddleware

# Compress any response body larger than 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

For batch prediction responses (thousands of records), this reduces response size by 60-80%, cutting network transfer time proportionally.

---

## Optimization 6: Database Connection Pooling

```python
# Too many connections → PostgreSQL runs out of connection slots
engine = create_async_engine(url, pool_size=100)  # ❌

# Right-sized pool
# Rule: pool_size = workers × 2
# With 4 workers: pool_size=8
engine = create_async_engine(
    url,
    pool_size=8,
    max_overflow=4,      # allow up to 4 extra connections under peak load
    pool_timeout=30,     # wait up to 30s for a connection from pool
    pool_recycle=1800,   # reconnect every 30 min (handles dropped connections)
    pool_pre_ping=True,  # test connections before using
)
```

---

## Profiling — Finding the Real Bottleneck

```bash
# py-spy: attach to running process, no code changes needed
pip install py-spy

# Find the PID of your uvicorn process
pgrep -f uvicorn

# Record a 30-second flame graph
py-spy record -o flamegraph.svg --pid <PID> --duration 30
open flamegraph.svg
# Wider bars = more CPU time. Look for unexpected wide bars in YOUR code.

# Top-mode: live view like htop but for Python
py-spy top --pid <PID>
```

---

## Performance Checklist

```
Critical (do these first):
☐ Model loaded at startup (not per-request)
☐ Using uvicorn --workers N (N = 2 × cores + 1, within memory limit)
☐ DB connection pool sized correctly (pool_size = workers × 2)

High impact:
☐ Redis caching for repeated inputs (if input space is limited)
☐ run_in_executor for CPU-bound inference
☐ GZip middleware for large responses
☐ DB indexes on filtered columns

Medium impact:
☐ Structured JSON logging (faster than plain text at scale)
☐ Avoid model.predict() inside loops — batch your inputs
☐ Use pandas efficiently (avoid .iterrows(), use vectorized operations)

Monitoring:
☐ P50/P95/P99 latency tracked (not just averages)
☐ Throughput (requests/second) tracked
☐ Error rate tracked and alerted on
☐ Memory usage per worker tracked (model loading, leak detection)
```

---

## Q&A

**Q: My P50 is 10ms but P99 is 2 seconds. What's happening?**

The 1% slowest requests are taking 200× longer than median. Common causes: (1) DB connection pool exhausted — some requests wait for a free connection. (2) Garbage collection pauses in Python. (3) A slow path in your code triggered by specific inputs. Profile during load test to identify.

**Q: Should I use `def` or `async def` for ML inference endpoints?**

Use `def` (synchronous). FastAPI automatically runs synchronous endpoints in a thread pool. Use `async def` only for I/O-bound work (database queries, external HTTP calls). Using `async def` for CPU-bound inference is wrong — it blocks the event loop.

**Q: Redis caching for predictions — won't stale predictions be a problem when I update the model?**

Yes! When you deploy a new model, flush the cache:
```python
async def flush_prediction_cache():
    await redis_client.delete(*await redis_client.keys("prediction:*"))
```
Call this as part of your model deployment process.

**Q: How do I tell if I'm CPU-bound vs I/O-bound?**

During load testing, watch CPU usage. If CPU is at 100%: CPU-bound (add workers or optimize inference). If CPU is at 20% with high latency: I/O-bound (DB is slow, add indexes or check connection pool).
