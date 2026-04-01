---
id: post-request-body
title: "06 · POST Request & Request Body"
sidebar_label: "06 · POST & Request Body"
sidebar_position: 6
tags: [post, request-body, pydantic, create, fastapi, beginner]
---

# POST Request & Request Body

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=sw8V7mLl3OI) · **Series:** FastAPI for ML – CampusX

---

## Why POST Needs a Request Body

Query parameters work well for simple values — a filter, a sort direction, a page number. But what if you need to send a patient's complete record to create it?

```
# Query params — works for simple values
GET /patients?city=Mumbai&age=35

# But for creating a patient, you need all these fields:
name, age, city, weight, height, smoker, blood_type, emergency_contact...

# Putting all this in a URL is ugly, has length limits, and
# is insecure (URLs appear in browser history and server logs)
```

The solution: a **request body** — a separate payload attached to the HTTP request, distinct from the URL.

```
POST /patients
Content-Type: application/json

{
  "name": "Anita Sharma",
  "age": 32,
  "city": "Delhi",
  "weight": 60.5,
  "height": 165,
  "smoker": false,
  "blood_type": "B+"
}
```

POST is the method for "here's data, create something with it."

---

## How FastAPI Knows Where Each Parameter Comes From

FastAPI uses a simple rule to automatically determine where each function parameter comes from:

```
Rule 1: Is the parameter name in the URL path {placeholder}?
        → PATH PARAMETER (extracted from URL)

Rule 2: Is the parameter type a Pydantic BaseModel?
        → REQUEST BODY (read from JSON body)

Rule 3: None of the above?
        → QUERY PARAMETER (read from ?key=value in URL)
```

```python
@app.post("/hospitals/{hospital_id}/patients")
def create_patient(
    hospital_id: str,              # Rule 1: in path → PATH PARAM
    notify_staff: bool = False,    # Rule 3: not in path, not Pydantic → QUERY PARAM
    patient: PatientCreate,        # Rule 2: is a Pydantic model → REQUEST BODY
):
    # hospital_id: from URL  /hospitals/H01/patients
    # notify_staff: from URL ?notify_staff=true
    # patient: from JSON body {"name": ..., "age": ...}
    pass
```

---

## The Complete Request Flow — Traced Step by Step

Here is exactly what happens when a client sends a POST request to create a patient:

```
1. Client sends HTTP request:
   POST /patients HTTP/1.1
   Host: localhost:8000
   Content-Type: application/json    ← tells server: body is JSON
   Content-Length: 87
   
   {"name":"Anita Sharma","age":32,"city":"Delhi","weight":60.5,"height":165}

2. Uvicorn receives the TCP connection and parses the HTTP protocol

3. FastAPI reads the request body (the raw JSON bytes)

4. FastAPI sees 'patient: PatientCreate' parameter → needs to build PatientCreate

5. FastAPI calls: PatientCreate.model_validate_json(body_bytes)

6. Pydantic validates each field:
   ✅ name: "Anita Sharma" → str, min_length=2 OK
   ✅ age: 32 → int, 0≤32≤150 OK
   ✅ city: "Delhi" → str OK
   ✅ weight: 60.5 → float, >0 OK
   ✅ height: 165.0 → float, >0 OK
   ✅ smoker: False → bool OK (not provided → default)
   
   If ANY validation fails → FastAPI returns 422, your function never runs

7. Pydantic creates the PatientCreate object, FastAPI passes it to your function

8. Your function runs: creates the record, saves to DB

9. Your function returns a Python dict

10. FastAPI converts it to JSON, sets Content-Type: application/json, status 201

11. Client receives: HTTP 201 Created { "id": "P003", "bmi": 22.22, ... }
```

---

## Building the Create Endpoint — Step by Step

### Step 1: Define the Input Schema

```python
from pydantic import BaseModel, Field, computed_field

class PatientCreate(BaseModel):
    """
    Defines what data the client must send to create a patient.
    
    Notice: no 'id' field. The server generates the ID.
    The client should NOT provide IDs for new resources.
    """
    name: str = Field(
        min_length=2,
        max_length=100,
        description="Full name of the patient"
    )
    age: int = Field(
        ge=0,
        le=150,
        description="Patient's age in years"
    )
    city: str = Field(description="City of residence")
    weight: float = Field(gt=0, description="Weight in kilograms")
    height: float = Field(gt=0, description="Height in centimeters")
    smoker: bool = Field(default=False, description="Whether the patient smokes")
    
    @computed_field
    @property
    def bmi(self) -> float:
        """Server-calculated BMI — client never sends this."""
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)
```

### Step 2: Define the Response Schema

```python
class PatientResponse(BaseModel):
    """
    Defines what the server returns after creating a patient.
    Different from PatientCreate: includes id and bmi.
    
    model_config = {"from_attributes": True} allows creating
    this schema from ORM model attributes (needed for databases).
    """
    id: str
    name: str
    age: int
    city: str
    weight: float
    height: float
    smoker: bool
    bmi: float
    
    model_config = {"from_attributes": True}
```

### Step 3: Write the Endpoint

```python
import json
from fastapi import FastAPI, HTTPException

app = FastAPI()
DB_FILE = "patients.json"

def load_db():
    with open(DB_FILE) as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=2)

@app.post(
    "/patients",
    response_model=PatientResponse,  # controls what fields are returned
    status_code=201,                  # 201 Created (not 200 OK)
    summary="Create a new patient",
    description="Creates a patient record. The server assigns the ID and calculates BMI.",
)
def create_patient(patient: PatientCreate):
    """
    The 'patient' parameter tells FastAPI:
    1. Read the request body as JSON
    2. Validate it against PatientCreate
    3. If valid: pass the PatientCreate object here
    4. If invalid: return 422 with details (this function never runs)
    
    response_model=PatientResponse tells FastAPI:
    1. Validate that we return all required fields
    2. Filter out any extra fields not in PatientResponse
    3. Show the correct response schema in Swagger UI
    """
    db = load_db()
    
    # Check for duplicate patient names (simple uniqueness check)
    existing_names = [p["name"].lower() for p in db.values()]
    if patient.name.lower() in existing_names:
        # 409 Conflict: this resource already exists
        raise HTTPException(
            status_code=409,
            detail=f"A patient named '{patient.name}' already exists"
        )
    
    # Generate server-side ID
    # We pad with zeros: P001, P002, ..., P999
    patient_id = f"P{len(db) + 1:03d}"
    
    # Convert Pydantic model → plain dict, then add server-generated fields
    new_record = patient.model_dump()   # includes computed bmi field
    new_record["id"] = patient_id
    
    db[patient_id] = new_record
    save_db(db)
    
    # Return the complete record. response_model=PatientResponse will:
    # 1. Validate it has all required fields
    # 2. Filter to only PatientResponse fields
    # 3. Serialize to JSON
    return new_record
```

---

## What `response_model` Does For You

This is a safety feature that protects you from accidentally exposing internal data:

```python
# Your internal representation might have sensitive fields:
internal_record = {
    "id": "P001",
    "name": "Ravi Kumar",
    "age": 35,
    # ↓ These should NEVER appear in API responses
    "internal_diagnosis_code": "ICD-10-CM-E11.9",
    "billing_status": "unpaid",
    "insurance_fraud_flag": True,
}

# response_model=PatientResponse will:
# ✅ Include: id, name, age, city, weight, height, smoker, bmi
# ❌ Filter out: internal_diagnosis_code, billing_status, insurance_fraud_flag
```

Without `response_model`, you might accidentally return internal fields to external clients. With it, you get automatic data filtering as a security layer.

---

## Understanding the 422 Error Response

When Pydantic validation fails, FastAPI returns a detailed 422:

```bash
curl -X POST http://localhost:8000/patients \
  -H "Content-Type: application/json" \
  -d '{"name": "A", "age": -5, "city": "Delhi", "weight": 0, "height": 165}'
```

Response:
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "name"],
      "msg": "String should have at least 2 characters",
      "input": "A",
      "ctx": {"min_length": 2}
    },
    {
      "type": "greater_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be greater than or equal to 0",
      "input": -5,
      "ctx": {"ge": 0}
    },
    {
      "type": "greater_than",
      "loc": ["body", "weight"],
      "msg": "Input should be greater than 0",
      "input": 0,
      "ctx": {"gt": 0}
    }
  ]
}
```

The `loc` field tells you exactly where the error is:
- `["body", "name"]` → inside the request body, in the name field
- `["body", "address", "pin_code"]` → nested: body → address object → pin_code field
- `["query", "limit"]` → a query parameter (not body)
- `["path", "patient_id"]` → a path parameter

---

## Background Tasks — Respond Fast, Work Later

Sometimes you want to respond immediately but do extra work after (send a welcome email, log to analytics, update a cache):

```python
from fastapi import BackgroundTasks

def send_welcome_email(patient_name: str, patient_id: str):
    """
    Runs AFTER the HTTP response is sent to the client.
    The client doesn't wait for this — they get their 201 instantly.
    
    Use BackgroundTasks for work that:
    - Doesn't affect the response
    - Takes < 30 seconds
    - Can tolerate failure (if the server crashes, the task is lost)
    
    For longer or more reliable background work, use Celery (Advanced Topics).
    """
    import time
    time.sleep(2)  # simulate slow email sending
    print(f"✉️ Welcome email sent to {patient_name} (ID: {patient_id})")

@app.post("/patients", response_model=PatientResponse, status_code=201)
def create_patient(
    patient: PatientCreate,
    background_tasks: BackgroundTasks,  # FastAPI injects this automatically
):
    db = load_db()
    pid = f"P{len(db)+1:03d}"
    record = {**patient.model_dump(), "id": pid}
    db[pid] = record
    save_db(db)
    
    # Queue the email task
    # add_task(function, arg1, arg2, ...)
    background_tasks.add_task(send_welcome_email, patient.name, pid)
    
    # Response sent immediately — client doesn't wait for the email
    return record
    # Email sends in the background after this return
```

---

## Q&A

**Q: How does FastAPI know to read from the body vs a query param?**

The type annotation. If the parameter is a Pydantic `BaseModel` subclass, FastAPI reads the request body and validates it against that model. If it's a simple Python type (`str`, `int`, `bool`) and not in the path, it's a query parameter. If it's in the URL path `{placeholder}`, it's a path parameter.

**Q: What's the Content-Type header and do I always need it?**

`Content-Type: application/json` tells the server that your body is JSON (not XML, not form data). Without it, many servers can't parse your body. FastAPI returns 422 if the body isn't valid JSON. HTTP clients like Python's `requests` library set this header automatically when you pass a `json=` argument. The `curl` command requires `-H "Content-Type: application/json"` manually.

**Q: Why `status_code=201` instead of `200`?**

HTTP 200 OK is generic — "request succeeded". HTTP 201 Created is specific — "request succeeded AND a new resource was created as a direct result." Using 201 communicates more precisely to clients and conforms to REST conventions. Some client frameworks treat 201 differently (showing a "created" notification rather than a generic success).

**Q: Can I make a field required in the response but optional in the request?**

Yes — that's exactly what separate input/output schemas are for:
```python
class PatientCreate(BaseModel):
    name: str           # required in request
    notes: Optional[str] = None   # optional in request

class PatientResponse(BaseModel):
    id: str             # required in response (server always provides it)
    name: str
    notes: str = ""     # always present in response (never null)
    created_at: datetime  # server-generated, not in request at all
```

**Q: What happens if I return extra fields not in `response_model`?**

FastAPI silently filters them out. Your function can return a dict with 20 fields; if `response_model` only has 5 fields, the client receives only those 5. This is intentional — it prevents accidental data leakage.
