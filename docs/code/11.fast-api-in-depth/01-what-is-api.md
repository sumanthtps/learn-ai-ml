---
id: what-is-api
title: "01 · What is an API?"
sidebar_label: "01 · What is an API?"
sidebar_position: 1
tags: [api, fundamentals, rest, http, ml-deployment, beginner]
---

# What is an API?

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=WJKsPchji0Q) · **Series:** FastAPI for ML – CampusX

---

## Visual Reference

![Client-server model](https://commons.wikimedia.org/wiki/Special:Redirect/file/Client-server_model.svg)

Source: [Wikimedia Commons - Client-server model](https://commons.wikimedia.org/wiki/File:Client-server_model.svg)

## Start Here: The Problem Every ML Engineer Faces

You have just trained a machine learning model. It predicts loan defaults with 91% accuracy. It runs beautifully in your Jupyter notebook. You are proud of it.

Then your manager asks: *"Can the mobile app team use this model?"*

You pause. Your model is a Python object — a `.pkl` file that only runs inside Python. The mobile app is built in Flutter. The web frontend is React. The company's data pipeline is in Java. None of them speak Python. None of them can import your `.pkl` file.

**This gap — between a working model and a usable product — is what APIs solve.**

An API is the universal translator between your ML model and every other system in the world, regardless of programming language or platform.

---

## What is an API? (Three Ways to Understand It)

**API** = **A**pplication **P**rogramming **I**nterface

### Mental Model 1: The Restaurant Waiter

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   YOU (Client)       WAITER (API)        KITCHEN (Server / Model)   │
│                                                                      │
│   "I want pasta"  ──────────────►   "One pasta, please"             │
│                                                                      │
│                                         [Kitchen cooks pasta]        │
│                                                                      │
│   🍝 gets pasta   ◄──────────────   "Here's the pasta"             │
│                                                                      │
│   KEY INSIGHT: You never enter the kitchen.                         │
│   You don't know how the pasta is made.                              │
│   The waiter (API) shields you from all that complexity.             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

In software:
- **You** = the mobile app, web frontend, or data pipeline
- **Waiter** = the API (defines valid requests, routes them, returns responses)
- **Kitchen** = your ML model, database, or business logic

### Mental Model 2: The Electrical Socket

A socket has a standard interface: two holes, specific voltage. You can plug in a lamp, a phone charger, a laptop — anything that matches the interface. The socket doesn't care what you plug in. The appliance doesn't care what's inside the wall.

An API is the same: a standard interface that any client can use without knowing the internal implementation.

### Mental Model 3: The Formal Definition

An API is a **contract**:
> "If you send me data in this format, I will return data in that format."

The ML model's API contract:
> "Send me `{age: int, smoker: bool, region: str}`. I will return `{prediction: 'high'|'medium'|'low', confidence: float}`."

---

## Why APIs Are Critical for ML

Without an API, your model is isolated:

```
Without API:                    With API:
─────────────                   ──────────────────────────────────────
                                
Jupyter Notebook                Mobile App ──► POST /predict ──► ML Model
     ↑                          Web App    ──► POST /predict ──► ML Model  
  (only you)                    Java Pipeline ► POST /predict ──► ML Model
                                Any Language ──► POST /predict ──► ML Model
```

With an API, your model becomes a **service** — callable from anywhere, by anything.

In the real world:
- **OpenAI** exposes GPT-4 as an API → millions of apps use it
- **Google** exposes Vision AI as an API → anyone can add image recognition
- **Stripe** exposes payment processing as an API → any website can accept payments

Your ML model can follow the same pattern: train once, serve everywhere.

---

## How an HTTP API Actually Works — Step by Step

All web APIs use HTTP (the same protocol your browser uses). Here's the complete flow when a mobile app calls your fraud detection model:

```
Step 1: User taps "Pay ₹5000" on their phone

Step 2: The phone app prepares an HTTP Request:
        ┌─────────────────────────────────────────┐
        │ POST /predict HTTP/1.1                  │  ← method + URL
        │ Host: api.mybank.com                    │  ← headers
        │ Content-Type: application/json          │
        │ Authorization: Bearer eyJhbGci...       │
        │                                         │
        │ {                                       │  ← body (the data)
        │   "amount": 5000,                       │
        │   "merchant": "Amazon",                 │
        │   "card_present": false                 │
        │ }                                       │
        └─────────────────────────────────────────┘

Step 3: Request travels over the internet to your server

Step 4: Uvicorn (web server) receives it

Step 5: FastAPI reads + validates the JSON body with Pydantic

Step 6: Your function runs → calls ML model → gets prediction

Step 7: FastAPI sends HTTP Response:
        ┌─────────────────────────────────────────┐
        │ HTTP/1.1 200 OK                         │  ← status code
        │ Content-Type: application/json          │  ← headers
        │                                         │
        │ {                                       │  ← body (the result)
        │   "decision": "approve",                │
        │   "fraud_score": 0.12,                  │
        │   "confidence": 0.94                    │
        │ }                                       │
        └─────────────────────────────────────────┘

Step 8: Phone app reads the response → shows "✅ Payment Approved"

Total time: < 100 milliseconds
```

---

## HTTP: The Language APIs Speak

HTTP (HyperText Transfer Protocol) is the grammar both sides must follow. Every HTTP interaction has these parts:

### The Method — What Action You Want

```
GET    → "Give me data"          (read, never modifies anything)
POST   → "Here's data, process it" (create, submit)
PUT    → "Replace this resource" (full update)
PATCH  → "Update just these fields" (partial update)
DELETE → "Remove this resource"  (delete)
```

For ML APIs, you'll use mostly:
- `GET /health` — check if the service is running
- `POST /predict` — send features, get predictions
- `GET /model/info` — get model metadata

### The URL — What Resource You're Accessing

```
https://api.mybank.com/v1/predict
│       │              │  │
│       │              │  └── endpoint (what function to call)
│       │              └───── version (v1, v2, etc.)
│       └──────────────────── domain (where the server lives)
└──────────────────────────── protocol (HTTPS = secure HTTP)
```

### The Status Code — What Happened

The server always replies with a 3-digit code. Learn these by heart:

| Code | Name | Meaning | When You See It |
|------|------|---------|-----------------|
| `200` | OK | Everything worked | Successful GET or POST |
| `201` | Created | New resource was made | After POST that creates |
| `400` | Bad Request | You sent garbage | Missing required field |
| `401` | Unauthorized | Who are you? | Missing/wrong API key |
| `403` | Forbidden | I know who you are, still no | Wrong permissions |
| `404` | Not Found | Doesn't exist | Wrong URL |
| `422` | Unprocessable Entity | Data failed validation | Wrong data type or range |
| `429` | Too Many Requests | Slow down | Rate limit hit |
| `500` | Internal Server Error | Server crashed | Bug in your code |
| `503` | Service Unavailable | Server overloaded/down | Model not loaded yet |

---

## JSON: The Universal Data Format

APIs almost always exchange data as **JSON** (JavaScript Object Notation). Despite the "JavaScript" name, every language reads and writes JSON.

```json
{
  "patient_id": "P001",
  "name": "Ravi Kumar",
  "age": 35,
  "weight": 78.5,
  "smoker": false,
  "medications": ["metformin", "lisinopril"],
  "address": {
    "city": "Mumbai",
    "pin": "400001"
  },
  "notes": null
}
```

JSON data types:
| JSON Type | Example | Python equivalent |
|-----------|---------|-------------------|
| String | `"Mumbai"` | `str` |
| Number | `78.5`, `35` | `float`, `int` |
| Boolean | `true`, `false` | `True`, `False` |
| Null | `null` | `None` |
| Object | `{ "key": "val" }` | `dict` |
| Array | `["a", "b"]` | `list` |

---

## REST: The Design Philosophy

Most modern APIs follow **REST** (Representational State Transfer) principles. REST is not a technology — it's a set of design guidelines that make APIs predictable and easy to use.

### Principle 1: Resources, Not Actions

```
❌ Bad (action-based, RPC style):
  POST /getAllPatients
  POST /createNewPatient
  POST /deletePatient?id=P001

✅ Good (resource-based, REST style):
  GET    /patients           → list patients
  POST   /patients           → create a patient
  GET    /patients/P001      → get one patient
  PUT    /patients/P001      → update patient P001
  DELETE /patients/P001      → delete patient P001
```

The URL identifies the **thing** (noun). The HTTP method specifies the **action** (verb).

### Principle 2: Stateless

Every request is completely independent. The server remembers nothing between requests. Each request must carry all information needed (credentials, data, context).

This is why you send `Authorization: Bearer <token>` on every single request — the server has no memory of your previous login.

### Principle 3: Consistent Patterns

If you learn how `/patients` works, you can predict how `/doctors` or `/prescriptions` will work. Consistent naming makes APIs learnable.

---

## Your ML API in the Deployment Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Complete ML Project Journey                       │
│                                                                     │
│  1. DATA PREP    2. MODEL TRAINING    3. API LAYER    4. CLOUD      │
│  ────────────    ───────────────────  ──────────────  ───────────── │
│  Pandas/SQL      sklearn/PyTorch      FastAPI          Docker        │
│  Feature eng     model.fit()          Pydantic         AWS EC2       │
│  EDA             model.pkl            endpoints        CI/CD         │
│                                                                     │
│  ← This series covers steps 3 and 4 →                              │
│                                                                     │
│  Without steps 3 & 4: model lives in notebook, used by no one      │
│  With steps 3 & 4:    model powers mobile apps, web apps, pipelines │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Real-World Examples of ML APIs

Every AI product you use daily is a FastAPI (or similar) endpoint under the hood:

| Product | What their ML API does |
|---------|----------------------|
| Google Translate | Receives text → returns translated text |
| Spotify Discover Weekly | Receives listening history → returns song recommendations |
| Gmail Smart Compose | Receives email draft → returns suggested completions |
| Netflix recommendations | Receives watch history → returns show suggestions |
| OpenAI ChatGPT | Receives messages → streams back generated tokens |
| GitHub Copilot | Receives code context → returns code suggestions |

When you call `openai.chat.completions.create(...)`, you're making a `POST` request to `https://api.openai.com/v1/chat/completions`. That's an API. FastAPI lets you build the same thing for your own models.

---

## What We Build in This Series

**Videos 1–7 (Fundamentals):** Patient Management API
A full CRUD API for managing patient records. You learn every HTTP method, path parameters, query parameters, and Pydantic validation through a relatable domain.

**Videos 8–9 (ML Serving):** Insurance Premium Prediction API  
A Random Forest model wrapped in FastAPI. Users send demographic features, the API returns a prediction. Connected to a Streamlit frontend.

**Videos 10–12 (Deployment):** Dockerized + AWS Deployed API  
The same API packaged in Docker, pushed to AWS ECR, running on EC2 with HTTPS — accessible by anyone worldwide.

**Advanced Topics:** Full production patterns — async PostgreSQL, JWT auth, Celery jobs, WebSocket streaming, testing, caching, and three industry-grade projects.

---

## Q&A

**Q: What's the difference between an API and a website?**

A website renders HTML, CSS, and JavaScript for humans to look at and click. An API returns raw JSON data for other code to process. Many services offer both: a website for humans, and an API for developers building on top of it. When you search Google, their website shows you results in HTML. When a developer queries Google's Search API, they get results as JSON.

**Q: What's the difference between a REST API and a GraphQL API?**

REST has multiple endpoints (one per resource/action): `/patients`, `/doctors`, `/prescriptions`. GraphQL has a single endpoint and the client specifies exactly what data it needs in the query. For ML APIs, REST is almost always the right choice — your prediction endpoint has a fixed, well-defined input/output schema. Use GraphQL when clients need highly flexible queries over complex, interconnected data.

**Q: Do I need to know web development to use FastAPI?**

No. FastAPI abstracts away all web server complexity — HTTP parsing, connection management, serialization. You write Python functions. FastAPI handles everything else. If you can write `def predict(features):`, you can build a FastAPI endpoint.

**Q: Can any programming language call a FastAPI endpoint?**

Yes. Any language that can make HTTP requests (all of them) can call your API. Python's `requests`, JavaScript's `fetch`, Java's `HttpClient`, Go's `net/http`, Postman, curl — they all work identically because HTTP is the universal language.

**Q: What is a "web server" and why do I need Uvicorn separately from FastAPI?**

FastAPI is a framework — it knows about your routes, validation rules, and how to handle requests. But it doesn't know how to listen on a network port or accept TCP connections. Uvicorn is the actual web server that does that. It listens on port 8000, accepts connections, parses HTTP, and passes requests to FastAPI. They're separate because the framework and the server are different concerns — you could swap Uvicorn for Hypercorn without changing any FastAPI code.

---

## Summary

| Concept | One-Line Definition |
|---------|-------------------|
| **API** | A contract that lets software talk to software |
| **HTTP** | The protocol (rules) that APIs use |
| **JSON** | The data format APIs use |
| **REST** | Design philosophy: nouns in URL, verbs as HTTP methods |
| **Endpoint** | One specific URL + method combination that does one thing |
| **Request** | Data sent from client to server |
| **Response** | Data sent from server back to client |
| **Status Code** | 3-digit number telling you what happened |

In the next video, we install FastAPI and write our first working endpoint in 5 lines of Python.
