---
id: http-methods
title: "03 · HTTP Methods in FastAPI"
sidebar_label: "03 · HTTP Methods"
sidebar_position: 3
tags: [http, get, post, put, delete, crud, fastapi, beginner]
---

# HTTP Methods in FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=O8KrViWNhOM) · **Series:** FastAPI for ML – CampusX

---

## The Project We Build Together

From this video onward, everything is practical. We build a **Patient Management API** — a backend for a clinic that stores patient records.

Why this domain? A clinic is intuitive — everyone understands what "create a patient record", "look up a patient", "update their city", and "delete a record" mean. These four operations map perfectly to the four HTTP methods you need to master.

The same patterns apply to any ML system:
- Create a training experiment → POST
- Retrieve model metrics → GET
- Update a model's stage → PUT/PATCH
- Delete an old model version → DELETE

**Our data store (for now):** A JSON file called `patients.json`. Simple, zero setup, lets us focus on learning HTTP without database complexity.

```json title="patients.json"
{
  "P001": {
    "id": "P001",
    "name": "Ravi Kumar",
    "age": 35,
    "city": "Mumbai",
    "weight": 78.5,
    "height": 175.0,
    "smoker": false
  },
  "P002": {
    "id": "P002",
    "name": "Priya Patel",
    "age": 28,
    "city": "Bengaluru",
    "weight": 62.0,
    "height": 162.0,
    "smoker": false
  }
}
```

We use a dictionary keyed by patient ID (not a list). Why? `patients["P001"]` is O(1) — instant, regardless of how many patients exist. Searching a list for P001 is O(n) — gets slower as the list grows.

---

## HTTP Methods — The Four Verbs of REST

HTTP methods (also called HTTP verbs) tell the server what *action* you want to perform on a resource. In English, we have verbs: *get, create, update, delete*. HTTP has the same:

```
"Give me all patients"                → GET    /patients
"Create this new patient"             → POST   /patients
"Replace patient P001 entirely"       → PUT    /patients/P001
"Update only patient P001's city"     → PATCH  /patients/P001
"Delete patient P001"                 → DELETE /patients/P001
```

The URL identifies the **thing** (noun). The HTTP method specifies the **action** (verb). This separation is the essence of REST design.

### Properties of HTTP Methods

| Method | CRUD | Safe? | Idempotent? | Has Body? |
|--------|------|-------|------------|-----------|
| `GET` | Read | ✅ Yes | ✅ Yes | ❌ No |
| `POST` | Create | ❌ No | ❌ No | ✅ Yes |
| `PUT` | Replace | ❌ No | ✅ Yes | ✅ Yes |
| `PATCH` | Partial Update | ❌ No | ❌ No | ✅ Yes |
| `DELETE` | Delete | ❌ No | ✅ Yes | Usually no |

**Safe** = never modifies data. GET is safe — calling it 100 times changes nothing.

**Idempotent** = calling it multiple times has the same result as calling it once. `DELETE /patients/P001` ten times → P001 is deleted (same end state). `POST /patients` ten times → ten new patients (different state each time, not idempotent).

---

## Setting Up the Code

```python title="main.py"
import json
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Patient Management API", version="1.0.0")

DB_FILE = "patients.json"

def load_db() -> dict:
    """
    Read the JSON file and return as Python dict.
    
    Why re-read on every request?
    If we read once at startup and store in memory, two concurrent
    POST requests could both read the same state, add different patients,
    and whichever writes last overwrites the other's addition.
    
    Re-reading ensures we always have the latest state.
    (In production, a real database handles this with transactions.)
    """
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data: dict) -> None:
    """
    Write dictionary back to JSON file.
    indent=2 makes the file human-readable (formatted with indentation).
    """
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)
```

---

## GET — Reading Data Without Side Effects

`GET` is the simplest and most common HTTP method. It retrieves data and **never modifies anything**. Calling `GET /patients` 1000 times leaves the database unchanged.

### GET All Records

```python
@app.get("/patients")
def list_all_patients():
    """
    Return every patient in the database.
    
    No input parameters needed — just return everything.
    FastAPI converts the Python dict to JSON automatically.
    
    HTTP Response:
    200 OK
    Content-Type: application/json
    
    {
      "P001": {"id": "P001", "name": "Ravi Kumar", ...},
      "P002": {"id": "P002", "name": "Priya Patel", ...}
    }
    """
    patients = load_db()
    return patients
```

### GET One Record

```python
@app.get("/patients/{patient_id}")
def get_one_patient(patient_id: str):
    """
    Return a single patient identified by patient_id.
    
    {patient_id} in the path is a PATH PARAMETER.
    FastAPI extracts it from the URL and passes it as the function argument.
    
    Request:  GET /patients/P001
    FastAPI:  calls get_one_patient(patient_id="P001")
    """
    patients = load_db()
    
    # Check if the patient exists before trying to return them
    if patient_id not in patients:
        # HTTPException immediately stops execution and returns an error response.
        # status_code=404: "Not Found" — the standard code for missing resources
        # detail: the human-readable error message the client will see
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found in the database"
        )
    
    return patients[patient_id]
```

**What `raise HTTPException` does:**

When Python sees `raise HTTPException(status_code=404, detail="...")`, it immediately:
1. Stops executing your function
2. Tells FastAPI to send an HTTP response with status code 404
3. Sets the response body to `{"detail": "Patient 'P999' not found in the database"}`

Without this check, accessing `patients["P999"]` would raise a Python `KeyError`, which FastAPI would catch and return as a 500 Internal Server Error — unhelpful to the client.

---

## POST — Creating New Resources

`POST` sends data in the request body to create a new resource. Unlike GET, POST has a body — a JSON payload containing the new data.

To validate the incoming JSON, we use Pydantic:

```python
from pydantic import BaseModel, Field, computed_field

class PatientCreate(BaseModel):
    """
    Defines what data the client must provide to create a patient.
    
    FastAPI uses this class to:
    1. Parse the incoming JSON body into a Python object
    2. Validate each field (type, constraints)
    3. Return 422 with details if anything fails
    4. Generate the request body schema in Swagger UI
    
    The client never provides 'id' or 'bmi' — the server calculates these.
    """
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=150)         # ge=≥0, le=≤150
    city: str
    weight: float = Field(gt=0)            # gt=strictly greater than 0
    height: float = Field(gt=0)
    smoker: bool = False                   # optional with default

    @computed_field
    @property
    def bmi(self) -> float:
        """Automatically calculated — not provided by the client."""
        height_m = self.height / 100       # convert cm to meters
        return round(self.weight / (height_m ** 2), 2)


@app.post("/patients", status_code=201)
def create_patient(patient: PatientCreate):
    """
    Create a new patient record.
    
    The 'patient' parameter type is PatientCreate (a Pydantic model).
    FastAPI sees this and knows: "read the JSON body, validate it against
    PatientCreate, and pass the resulting object to this function."
    
    If validation fails → 422 error (function never called)
    If validation passes → function receives a validated PatientCreate object
    
    status_code=201: "Created" — more specific than 200 "OK"
    Use 201 when a new resource is created.
    """
    db = load_db()
    
    # Generate a unique patient ID
    # f"P{len(db) + 1:03d}" with 5 patients: "P006"
    # :03d pads with zeros to 3 digits
    patient_id = f"P{len(db) + 1:03d}"
    
    # Convert Pydantic model to plain dict and add the server-generated ID
    new_record = patient.model_dump()    # {"name": ..., "age": ..., "bmi": ...}
    new_record["id"] = patient_id        # {"name": ..., "id": "P006", ...}
    
    db[patient_id] = new_record
    save_db(db)
    
    return {
        "message": "Patient created successfully",
        "id": patient_id,
        "bmi": patient.bmi
    }
```

---

## PUT — Full Replacement Update

`PUT` replaces an entire resource. You send all fields, even unchanged ones. The server discards the old record and stores your new one.

Think of it like filling out a form from scratch — even if you only changed your city, you still fill in your name, age, and everything else.

```python
@app.put("/patients/{patient_id}")
def replace_patient(patient_id: str, patient: PatientCreate):
    """
    Completely replace a patient's record.
    
    All fields are required (same schema as PatientCreate).
    The old record is entirely discarded and replaced.
    
    Use PUT when the client always has the complete current state
    and wants to replace everything.
    """
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found. Cannot update."
        )
    
    # Full replacement — build completely new record
    updated = patient.model_dump()
    updated["id"] = patient_id          # preserve the original ID
    
    db[patient_id] = updated            # old record gone, new record in
    save_db(db)
    
    return {"message": "Patient updated", "patient": updated}
```

---

## PATCH — Partial Update (Most Common in Practice)

`PATCH` updates only the fields you send. Everything else stays as-is. This is more practical than PUT — if you only want to change the city, you only send the city.

The critical tool: `model_dump(exclude_unset=True)` — returns only the fields the client actually sent.

```python
from typing import Optional

class PatientPatch(BaseModel):
    """
    For PATCH, every field is Optional.
    The client sends only what they want to change.
    """
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    smoker: Optional[bool] = None


@app.patch("/patients/{patient_id}")
def update_patient_fields(patient_id: str, updates: PatientPatch):
    """
    Update only the provided fields. All others remain unchanged.
    """
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    
    existing = db[patient_id]
    
    # exclude_unset=True returns ONLY fields the client explicitly sent
    # Client sends {"city": "Chennai"}:
    #   Without exclude_unset: {"name": None, "age": None, "city": "Chennai", ...}
    #   With exclude_unset:    {"city": "Chennai"}   ← only what they sent
    changed_fields = updates.model_dump(exclude_unset=True)
    
    # Merge: update changed fields, keep everything else
    existing.update(changed_fields)
    db[patient_id] = existing
    save_db(db)
    
    return {"message": "Patient updated", "patient": existing}
```

---

## DELETE — Removing Resources

```python
from fastapi import Response

@app.delete("/patients/{patient_id}", status_code=204)
def delete_patient(patient_id: str):
    """
    Delete a patient record permanently.
    
    Returns 204 No Content on success — no response body.
    Why 204 and not 200? 
    204 means "success, but there's nothing to return."
    After deletion, the resource no longer exists — there's nothing to send back.
    
    200 would imply there's a body to read.
    204 correctly signals "done, nothing more to say."
    """
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' doesn't exist. Nothing to delete."
        )
    
    del db[patient_id]    # remove the patient from the dict
    save_db(db)
    
    # Must return Response(status_code=204) explicitly for 204 endpoints
    # Returning None or nothing may cause FastAPI to serialize 'null'
    return Response(status_code=204)
```

---

## The Complete CRUD API in One File

```python title="main.py"
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, computed_field

app = FastAPI(title="Patient Management API", version="1.0.0")
DB_FILE = "patients.json"

def load_db() -> dict:
    with open(DB_FILE) as f: return json.load(f)

def save_db(data: dict):
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=2)

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

class PatientPatch(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    smoker: Optional[bool] = None

@app.get("/patients", tags=["patients"])
def list_patients():
    return load_db()

@app.get("/patients/{pid}", tags=["patients"])
def get_patient(pid: str):
    db = load_db()
    if pid not in db: raise HTTPException(404, f"Patient '{pid}' not found")
    return db[pid]

@app.post("/patients", status_code=201, tags=["patients"])
def create_patient(patient: PatientCreate):
    db = load_db()
    pid = f"P{len(db)+1:03d}"
    db[pid] = {**patient.model_dump(), "id": pid}
    save_db(db)
    return db[pid]

@app.put("/patients/{pid}", tags=["patients"])
def replace_patient(pid: str, patient: PatientCreate):
    db = load_db()
    if pid not in db: raise HTTPException(404, "Patient not found")
    db[pid] = {**patient.model_dump(), "id": pid}
    save_db(db)
    return db[pid]

@app.patch("/patients/{pid}", tags=["patients"])
def patch_patient(pid: str, updates: PatientPatch):
    db = load_db()
    if pid not in db: raise HTTPException(404, "Patient not found")
    db[pid].update(updates.model_dump(exclude_unset=True))
    save_db(db)
    return db[pid]

@app.delete("/patients/{pid}", status_code=204, tags=["patients"])
def delete_patient(pid: str):
    db = load_db()
    if pid not in db: raise HTTPException(404, "Patient not found")
    del db[pid]
    save_db(db)
    return Response(status_code=204)
```

---

## Testing Your Endpoints

### Option 1: Swagger UI (Easiest)
1. Run `uvicorn main:app --reload`
2. Open http://localhost:8000/docs
3. Click any endpoint → "Try it out" → fill fields → "Execute"

### Option 2: curl (Terminal)

```bash
# List all patients
curl http://localhost:8000/patients

# Get specific patient
curl http://localhost:8000/patients/P001

# Create patient (note: -d sends JSON body)
curl -X POST http://localhost:8000/patients \
  -H "Content-Type: application/json" \
  -d '{"name":"Anita Sharma","age":32,"city":"Delhi","weight":60.5,"height":165}'

# Update only city (PATCH)
curl -X PATCH http://localhost:8000/patients/P001 \
  -H "Content-Type: application/json" \
  -d '{"city": "Chennai"}'

# Delete patient
curl -X DELETE http://localhost:8000/patients/P001
# Returns 204 No Content (empty body)
```

---

## Q&A

**Q: Why does GET have no body? Why can't I send JSON in a GET request?**

The HTTP specification says GET requests should be "safe" — they read data without side effects. The convention is that all GET data goes in the URL (path or query params). Some clients and proxies strip the body from GET requests entirely. If you need to send complex data to retrieve something, use POST.

**Q: When do I use PUT vs PATCH?**

PUT: the client has the full current state of the resource and wants to replace everything. Like saving all settings at once.  
PATCH: the client wants to change specific fields only. Like editing just your profile picture.  
In practice, PATCH is more common because it's bandwidth-efficient and doesn't require the client to know all current field values.

**Q: What if I forget the `tags=["patients"]` parameter?**

Nothing breaks — your endpoints still work. Tags are cosmetic: they group endpoints in Swagger UI into collapsible sections. Without tags, all endpoints appear in a flat alphabetical list, which gets confusing with 20+ endpoints.

**Q: Should I always check if a resource exists before updating/deleting?**

Yes. Without the check: if `patient_id` isn't in the dict, Python raises `KeyError`, FastAPI returns a confusing 500 error. With the check: FastAPI returns a clear 404 "Patient not found." Always handle the "resource doesn't exist" case explicitly.

**Q: Is returning a dict from my endpoint the same as returning JSON?**

Yes. FastAPI automatically calls `json.dumps()` on any dict/list you return and sets `Content-Type: application/json`. You never need to manually serialize to JSON.
