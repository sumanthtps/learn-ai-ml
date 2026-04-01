---
id: post-request-body
title: "06 · POST Request & Request Body"
sidebar_label: "06 · POST & Request Body"
sidebar_position: 6
tags: [post, request-body, pydantic, create, fastapi]
---

# POST Request & Request Body

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=sw8V7mLl3OI) · **Series:** FastAPI for ML – CampusX

---

## What is a Request Body?

For `POST` and `PUT` requests, the data travels in the **request body** — not the URL. This allows sending complex, structured JSON data.

```
POST /patients
Content-Type: application/json

{
  "name": "Anita Sharma",
  "age": 32,
  "city": "Delhi",
  "smoker": false,
  "weight": 60.5,
  "height": 165
}
```

Query parameters are fine for simple values. Request body is necessary when:
- Sending complex/nested data
- Sending multiple fields
- Sending data with types beyond string (arrays, booleans, floats)
- Data is sensitive (don't put passwords in URLs!)

---

## Accepting a Request Body in FastAPI

FastAPI detects request body by looking at function parameters that are **Pydantic BaseModel** instances:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal
import json

app = FastAPI()

class PatientCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=150)
    city: str
    weight: float = Field(gt=0, description="Weight in kg")
    height: float = Field(gt=0, description="Height in cm")
    smoker: bool = False

    @computed_field
    @property
    def bmi(self) -> float:
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)

@app.post("/patients", status_code=201)
def create_patient(patient: PatientCreate):
    patients = load_patients()
    
    # Check for duplicate
    if any(p["name"] == patient.name for p in patients.values()):
        raise HTTPException(status_code=409, detail="Patient already exists")
    
    # Generate ID
    patient_id = f"P{len(patients) + 1:03d}"
    
    # Store including computed BMI
    patients[patient_id] = patient.model_dump()
    patients[patient_id]["id"] = patient_id
    
    save_patients(patients)
    
    return {"message": "Patient created", "id": patient_id, "bmi": patient.bmi}
```

---

## How FastAPI Handles the Request Body

```
1. Client sends POST with JSON body
2. Uvicorn receives the HTTP request
3. FastAPI reads the body bytes
4. FastAPI passes body to Pydantic model constructor
5. Pydantic validates all fields (type, constraints)
   ├── FAIL → FastAPI returns 422 with error details
   └── PASS → FastAPI calls your function with validated model
6. Your function runs and returns a response
7. FastAPI serializes return value to JSON
8. Client receives 201 Created with response JSON
```

---

## The 422 Validation Error Response

When validation fails, FastAPI returns a detailed 422 automatically:

```json
{
  "detail": [
    {
      "type": "int_parsing",
      "loc": ["body", "age"],
      "msg": "Input should be a valid integer",
      "input": "thirty",
      "url": "https://errors.pydantic.dev/..."
    },
    {
      "type": "greater_than",
      "loc": ["body", "weight"],
      "msg": "Input should be greater than 0",
      "input": -5.0,
      "ctx": {"gt": 0}
    }
  ]
}
```

You get this for free — no manual validation code needed.

---

## Separating Request and Response Models

A common production pattern: use different models for input and output.

```python
# INPUT — what the client sends
class PatientCreate(BaseModel):
    name: str
    age: int
    city: str
    weight: float
    height: float
    smoker: bool = False
    # No ID — server generates it

# RESPONSE — what the server returns
class PatientResponse(BaseModel):
    id: str
    name: str
    age: int
    city: str
    bmi: float
    smoker: bool
    # No password, no internal fields

@app.post("/patients", status_code=201, response_model=PatientResponse)
def create_patient(patient: PatientCreate) -> PatientResponse:
    ...
```

The `response_model=PatientResponse` on the decorator:
- Filters out fields not in `PatientResponse` (even if your dict has extras)
- Generates accurate response schema in Swagger UI
- Validates your own output (raises 500 if your code returns wrong types)

---

## Combining Path Params, Query Params, and Body

FastAPI can mix all three in a single endpoint:

```python
@app.post("/hospitals/{hospital_id}/patients")
def create_patient_in_hospital(
    hospital_id: str,                # path param
    notify: bool = False,            # query param
    patient: PatientCreate = Body()  # request body
):
    # hospital_id → from URL path
    # notify → from ?notify=true query string
    # patient → from request body JSON
    ...
```

FastAPI's rule:
- In URL path → **path parameter**
- No path match, simple type → **query parameter**
- Pydantic model → **request body**

---

## The Full CRUD Implementation (so far)

```python title="main.py"
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, computed_field

app = FastAPI(title="Patient Management API", version="1.0.0")

DB_PATH = "patients.json"

def load_db() -> dict:
    with open(DB_PATH) as f:
        return json.load(f)

def save_db(data: dict):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

class PatientCreate(BaseModel):
    name: str = Field(min_length=2)
    age: int = Field(ge=0, le=150)
    city: str
    weight: float = Field(gt=0)
    height: float = Field(gt=0)
    smoker: bool = False

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / ((self.height / 100) ** 2), 2)

# READ — all patients
@app.get("/patients")
def get_all_patients():
    return load_db()

# READ — single patient
@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404, f"Patient {patient_id} not found")
    return db[patient_id]

# CREATE — new patient
@app.post("/patients", status_code=201)
def create_patient(patient: PatientCreate):
    db = load_db()
    patient_id = f"P{len(db)+1:03d}"
    db[patient_id] = {**patient.model_dump(), "id": patient_id}
    save_db(db)
    return {"id": patient_id, "message": "Created"}
```

---

## Topics Not Covered in the Video

### `Body()` — Embedding Multiple Models

```python
from fastapi import Body

@app.post("/admit")
def admit_patient(
    patient: PatientCreate = Body(embed=True),    # {"patient": {...}}
    doctor: DoctorAssign = Body(embed=True)       # {"doctor": {...}}
):
    # JSON: {"patient": {...}, "doctor": {...}}
    ...
```

### Form Data vs JSON Body

```python
from fastapi import Form

# For HTML form submissions (not JSON)
@app.post("/login")
def login(username: str = Form(), password: str = Form()):
    ...
```

```python
# For file uploads
from fastapi import File, UploadFile

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    contents = await file.read()
    with open(f"artifacts/{file.filename}", "wb") as f:
        f.write(contents)
    return {"filename": file.filename}
```

### Returning Custom Headers in Response

```python
from fastapi import Response

@app.post("/patients", status_code=201)
def create_patient(patient: PatientCreate, response: Response):
    ...
    patient_id = "P001"
    response.headers["Location"] = f"/patients/{patient_id}"
    return {"id": patient_id}
```

### Background Tasks — Don't Block on Non-Critical Work

```python
from fastapi import BackgroundTasks

def send_welcome_email(patient_name: str):
    # This runs AFTER the response is sent
    email_service.send(f"Welcome, {patient_name}!")

@app.post("/patients", status_code=201)
def create_patient(
    patient: PatientCreate,
    background_tasks: BackgroundTasks
):
    # ... create patient ...
    background_tasks.add_task(send_welcome_email, patient.name)
    return {"id": patient_id, "message": "Created"}
    # Response is returned immediately; email sends in background
```

---

## Q&A

**Q: How does FastAPI know that a parameter is a request body vs a query param?**
> If the parameter type is a Pydantic `BaseModel`, FastAPI treats it as a request body. Simple Python types (`str`, `int`, `bool`) are treated as query params (unless they appear in the path).

**Q: Can a GET request have a body?**
> Technically yes, but it's non-standard and many clients/proxies strip it. For GET requests, use query parameters. Use POST/PUT if you need complex inputs.

**Q: What if I want to accept arbitrary JSON without a fixed schema?**
> Use `dict` or `Any` as the type:
> ```python
> from typing import Any
> @app.post("/ingest")
> def ingest(data: dict[str, Any]):
>     ...
> ```
> This skips Pydantic validation — only do this if you truly need arbitrary input.

**Q: How do I return a 201 status code with a body?**
> Use `status_code=201` in the decorator. FastAPI still serializes and returns whatever you return from the function. `status_code` in the decorator sets the *default* success code, but you can override it with `Response`.

**Q: What is `model_dump(exclude_unset=True)` used for?**
> When doing a PATCH (partial update), you only want to update the fields the client explicitly sent. `exclude_unset=True` gives you only those fields, ignoring fields that defaulted to None.

---

## Testing with curl

```bash
# Create a patient
curl -X POST http://localhost:8000/patients \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Anita Sharma",
    "age": 32,
    "city": "Delhi",
    "weight": 60.5,
    "height": 165,
    "smoker": false
  }'

# See all patients
curl http://localhost:8000/patients

# Get specific patient
curl http://localhost:8000/patients/P001
```
