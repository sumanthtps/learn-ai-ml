---
id: what-is-api
title: "01 · What is an API?"
sidebar_label: "01 · What is an API?"
sidebar_position: 1
tags: [api, fundamentals, rest, http, ml-deployment]
---

# What is an API?

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=WJKsPchji0Q) · **Series:** FastAPI for ML – CampusX

---

## Why APIs matter for ML Engineers

When you train a machine learning model, the model lives inside a Jupyter notebook or a Python script. It's useful to *you*, but the world can't use it yet. To make your model available to:

- a **web application** (React / Angular frontend)
- a **mobile app** (Android / iOS)
- **another backend service**
- **third-party partners**

…you need to **wrap your model in an API**. In the real world, 9 out of 10 companies build their ML inference layer using an API framework, and **FastAPI** is the industry standard for Python ML projects.

---

## What is an API?

> **API** stands for **Application Programming Interface** — a mechanism that enables two software components to communicate with each other using a defined set of rules, protocols, and data formats.

Think of an API as a **waiter** in a restaurant:
- You (client) place an order (request)
- The waiter (API) carries the request to the kitchen (backend/model)
- The kitchen prepares the response
- The waiter brings the food (response) back to you

```
Client  ──── request ────▶  API  ──── call ────▶  Backend / ML Model
Client  ◀─── response ────  API  ◀─── result ────  Backend / ML Model
```

---

## A Concrete ML Example

```
Client sends:  POST /predict
               { "feature1": 23.5, "feature2": 1.2 }

API returns:   200 OK
               { "prediction": "High Risk", "confidence": 0.87 }
```

Your ML model never needs to know *who* is calling it. It just receives features and returns a prediction. This separation of concerns is the power of APIs.

---

## Key API Concepts

### 1. Endpoint
A specific URL path that performs one action. Each ML operation (predict, retrain, health-check) gets its own endpoint.

```
GET  /health          → Check if the API is alive
GET  /model-info      → Return model metadata
POST /predict         → Run inference
POST /batch-predict   → Run inference on a batch
```

### 2. HTTP Protocol
APIs on the web communicate over HTTP. Every HTTP interaction has:

| Part | Description | Example |
|------|-------------|---------|
| **Method** | The action type | `GET`, `POST`, `PUT`, `DELETE` |
| **URL** | The resource address | `https://api.myml.com/predict` |
| **Headers** | Metadata about the request | `Content-Type: application/json` |
| **Body** | The payload (for POST/PUT) | `{"age": 30, "income": 50000}` |
| **Status Code** | Whether it succeeded | `200 OK`, `422 Unprocessable Entity` |

### 3. JSON — The Universal Data Format
APIs almost always send/receive **JSON** (JavaScript Object Notation):

```json
{
  "patient_id": "P123",
  "age": 45,
  "smoker": true,
  "region": "northeast",
  "predicted_premium": "high"
}
```

### 4. REST — The Architectural Style
Most modern APIs are **RESTful**:
- **Stateless** — each request is independent; no session stored server-side
- **Resource-based** — endpoints represent nouns (`/patients`, `/predictions`)
- **HTTP method semantics** — GET=read, POST=create, PUT=update, DELETE=remove

---

## API in the ML Deployment Lifecycle

```
┌─────────────────────────────────────────────────────┐
│                 ML Deployment Journey                │
├─────────────┬──────────────┬────────────────────────┤
│  Notebook   │  API Layer   │  Serving Infrastructure│
│             │              │                        │
│  train.py   │  main.py     │  Docker + AWS/GCP/Azure│
│  model.pkl  │  (FastAPI)   │                        │
└─────────────┴──────────────┴────────────────────────┘
```

---

## Topics Not Covered in the Video (But Important for Daily Work)

### API Versioning
Always version your API so clients don't break when you update:

```python
# Bad: no versioning
@app.get("/predict")

# Good: versioned
@app.get("/v1/predict")
@app.get("/v2/predict")  # new model, new schema
```

### API Authentication
In production, your API must be secured:

| Method | Use Case |
|--------|----------|
| **API Key** (header) | Simple service-to-service |
| **Bearer Token (JWT)** | User-facing, stateless |
| **OAuth2** | Third-party integrations |
| **mTLS** | High-security internal services |

```python
# Simple API Key via header
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/predict")
async def predict(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403)
    ...
```

### Rate Limiting
Protect your ML model from being overwhelmed:

```python
# Using slowapi
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/predict")
@limiter.limit("10/minute")
async def predict(request: Request):
    ...
```

### API Documentation Standards
- **OpenAPI / Swagger** — the standard format; FastAPI generates this automatically
- **Postman Collections** — share runnable examples with your team

---

## Quick Reference: HTTP Status Codes for ML APIs

| Code | Meaning | When to Use |
|------|---------|-------------|
| `200 OK` | Success | Prediction returned |
| `201 Created` | Resource created | New record inserted |
| `400 Bad Request` | Client sent bad data | Missing feature in input |
| `401 Unauthorized` | No/invalid auth | Missing API key |
| `403 Forbidden` | Auth ok, permission denied | Access to restricted model |
| `404 Not Found` | Resource doesn't exist | Patient ID not found |
| `422 Unprocessable Entity` | Validation failed | Feature value out of range |
| `429 Too Many Requests` | Rate limit hit | Inference overload |
| `500 Internal Server Error` | Server crashed | Model threw an exception |

---

## Q&A

**Q: Why can't I just expose my Python function directly instead of building an API?**
> A raw Python function can't be called over the internet. An API wraps your function in a web server that listens for HTTP requests, validates inputs, handles errors, and returns structured responses. It also enables multiple clients (web, mobile, other services) to call your model without knowing anything about Python.

**Q: Do I always need FastAPI? What about Flask?**
> Flask works, but FastAPI offers automatic input validation, async support, and automatic OpenAPI docs generation — all of which matter enormously in production ML systems. Flask uses synchronous WSGI; FastAPI uses async ASGI (via `uvicorn`), which handles concurrent requests far more efficiently.

**Q: What's the difference between REST and GraphQL?**
> REST has fixed endpoints per resource; GraphQL has a single endpoint where the client specifies exactly what data it needs. For ML APIs, REST is almost always the right choice — your prediction endpoint has a fixed input/output schema.

**Q: What is a webhook vs an API?**
> An API is *you* calling someone else. A webhook is *them* calling you. Example: you build a `/predict` API (you get called). A webhook would be Stripe calling your `/payment-success` endpoint when a payment completes.

**Q: How do I test my API without a frontend?**
> Use **Postman**, **curl**, **httpie**, or FastAPI's built-in `/docs` (Swagger UI). For automated testing, use `pytest` with the `TestClient` from FastAPI.

---

## Further Reading

- [REST API Best Practices — Microsoft Azure Docs](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)
- [FastAPI Official Docs](https://fastapi.tiangolo.com)
- [HTTP Status Codes Reference](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
