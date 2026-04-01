---
id: http-methods
title: "03 · HTTP Methods in FastAPI"
sidebar_label: "03 · HTTP Methods"
sidebar_position: 3
tags: [http, get, post, put, delete, crud, fastapi]
---

# HTTP Methods in FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=O8KrViWNhOM) · **Series:** FastAPI for ML – CampusX

---

## The Project: Patient Management System

Throughout this playlist, we build a **Patient Management API** — a backend that stores patient records and exposes CRUD operations. This is used as a proxy for any real-world ML or data application.

```
Domain: Healthcare
Data store: patients.json (simulating a database)
Operations: Create / Read / Update / Delete patients
```

Sample patient record:

```json
{
  "id": "P001",
  "name": "Ramesh Kumar",
  "age": 45,
  "weight": 80.5,
  "height": 170,
  "blood_type": "A+",
  "smoker": false,
  "city": "Mumbai",
  "insurance_premium": "medium"
}
```

---

## HTTP Methods — The Verbs of REST

Every HTTP request uses a **method (verb)** that tells the server what to do with the resource.

| Method | CRUD Operation | Safe? | Idempotent? | Has Body? |
|--------|---------------|-------|------------|-----------|
| `GET` | **Read** | ✅ Yes | ✅ Yes | ❌ No |
| `POST` | **Create** | ❌ No | ❌ No | ✅ Yes |
| `PUT` | **Update (replace)** | ❌ No | ✅ Yes | ✅ Yes |
| `PATCH` | **Update (partial)** | ❌ No | ❌ No | ✅ Yes |
| `DELETE` | **Delete** | ❌ No | ✅ Yes | Optional |

- **Safe**: doesn't modify data on the server
- **Idempotent**: calling it 10 times = calling it once (same result)

---

## Implementing HTTP Methods in FastAPI

### GET — Retrieve Data

```python title="main.py"
from fastapi import FastAPI
import json

app = FastAPI()

def load_patients():
    with open("patients.json", "r") as f:
        return json.load(f)

# GET all patients
@app.get("/patients")
def get_all_patients():
    return load_patients()

# GET a single patient by ID
@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    patients = load_patients()
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patients[patient_id]
```

### POST — Create New Data

```python
from fastapi import HTTPException
from pydantic import BaseModel

class PatientCreate(BaseModel):
    name: str
    age: int
    city: str

@app.post("/patients", status_code=201)
def create_patient(patient: PatientCreate):
    patients = load_patients()
    patient_id = f"P{len(patients) + 1:03d}"
    patients[patient_id] = patient.model_dump()
    
    with open("patients.json", "w") as f:
        json.dump(patients, f)
    
    return {"message": "Patient created", "id": patient_id}
```

### PUT — Update (Full Replace)

```python
@app.put("/patients/{patient_id}")
def update_patient(patient_id: str, patient: PatientCreate):
    patients = load_patients()
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patients[patient_id] = patient.model_dump()
    
    with open("patients.json", "w") as f:
        json.dump(patients, f)
    
    return {"message": "Patient updated"}
```

### DELETE — Remove Data

```python
@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    patients = load_patients()
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    del patients[patient_id]
    
    with open("patients.json", "w") as f:
        json.dump(patients, f)
    
    return {"message": f"Patient {patient_id} deleted"}
```

---

## Response Models — Controlling Output

```python
from pydantic import BaseModel
from typing import Optional

class PatientResponse(BaseModel):
    id: str
    name: str
    age: int
    city: str
    # NOTE: password/sensitive fields NOT included here

@app.get("/patients/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str):
    # FastAPI will automatically filter fields
    # not in PatientResponse
    ...
```

---

## HTTPException — Returning Meaningful Errors

```python
from fastapi import HTTPException

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    patients = load_patients()
    
    if patient_id not in patients:
        raise HTTPException(
            status_code=404,
            detail=f"Patient with ID '{patient_id}' not found"
        )
    
    return patients[patient_id]
```

The client receives:
```json
{
  "detail": "Patient with ID 'P999' not found"
}
```

---

## Response Status Codes in Decorators

```python
# Default is 200
@app.get("/patients")

# Explicitly set 201 for created resources
@app.post("/patients", status_code=201)

# 204 No Content for successful deletes
@app.delete("/patients/{id}", status_code=204)
```

---

## Topics Not Covered in the Video

### PATCH — Partial Updates (Common in Real APIs)

`PUT` replaces the entire resource. `PATCH` updates only the fields you provide:

```python
from typing import Optional

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None

@app.patch("/patients/{patient_id}")
def patch_patient(patient_id: str, updates: PatientUpdate):
    patients = load_patients()
    if patient_id not in patients:
        raise HTTPException(status_code=404)
    
    # Only update fields that were provided
    existing = patients[patient_id]
    updated_data = updates.model_dump(exclude_unset=True)  # key!
    existing.update(updated_data)
    patients[patient_id] = existing
    
    with open("patients.json", "w") as f:
        json.dump(patients, f)
    
    return existing
```

### HEAD and OPTIONS — Less Common But Real

```python
# HEAD — like GET but returns headers only (no body)
# FastAPI handles HEAD automatically for GET routes

# OPTIONS — returns what methods a route supports
# Used by browsers for CORS preflight checks
# FastAPI handles OPTIONS automatically
```

### Returning Different Response Types

```python
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse

@app.get("/report")
def get_report():
    return FileResponse("report.pdf", media_type="application/pdf")

@app.get("/ping")
def ping():
    return PlainTextResponse("pong")

@app.get("/patients")
def get_patients():
    return JSONResponse(
        content={"patients": []},
        status_code=200,
        headers={"X-Total-Count": "0"}
    )
```

### Endpoint Tags — Organizing Swagger UI

```python
@app.get("/patients", tags=["patients"])
def get_patients(): ...

@app.post("/predict", tags=["ml"])
def predict(): ...
```

---

## Q&A

**Q: When should I use POST vs PUT?**
> Use `POST` when creating a *new* resource (you don't know its ID yet). Use `PUT` when updating a *known* resource by its ID, replacing it entirely. Use `PATCH` for partial updates.

**Q: Why is GET idempotent but POST is not?**
> Calling `GET /patients` 5 times returns the same result — the data isn't changed. Calling `POST /patients` 5 times creates 5 patients — the state of the server changes each time.

**Q: Can I put data in the body of a GET request?**
> Technically yes, but you *shouldn't*. GET is for retrieval; data should go in query parameters. Many HTTP clients and proxies strip the body of GET requests.

**Q: What's the right status code for a validation error?**
> `422 Unprocessable Entity`. FastAPI returns this automatically when Pydantic validation fails.

**Q: What's the difference between `raise HTTPException` and `return`?**
> `raise HTTPException` immediately stops the function and returns an error response to the client. `return` sends a successful response. Use `raise` for error cases.

**Q: Should ML predict endpoints use GET or POST?**
> Use `POST` — even though prediction doesn't "create" anything, you're sending complex JSON data in the body which is idiomatic with POST. GET requests don't have a standard body. Some teams use `GET /predict?feature1=5&feature2=3` for simple models, but POST is more maintainable.

---

## HTTP Method Decision Tree

```
Want to retrieve data?
  ├── Yes, no side effects → GET
  └── No, sending data →
        Creating a new resource? → POST
        Updating an existing resource entirely? → PUT
        Updating specific fields only? → PATCH
        Removing a resource? → DELETE
```
