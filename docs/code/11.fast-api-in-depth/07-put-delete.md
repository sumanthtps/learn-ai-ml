---
id: put-delete
title: "07 · PUT & DELETE — Completing CRUD"
sidebar_label: "07 · PUT & DELETE"
sidebar_position: 7
tags: [put, delete, patch, update, crud, fastapi, beginner]
---

# PUT & DELETE — Completing CRUD

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=XVu22pTwWE8) · **Series:** FastAPI for ML – CampusX

---

## The CRUD Checklist

By Video 6, we had Create and Read working. This video completes the picture:

```
✅ GET    /patients            → Read all
✅ GET    /patients/{id}       → Read one
✅ POST   /patients            → Create
🔲 PUT    /patients/{id}       → Replace (Update)
🔲 PATCH  /patients/{id}       → Partial Update
🔲 DELETE /patients/{id}       → Delete
```

With all six endpoints, clients can perform every possible operation on patient records.

---

## PUT — Full Replacement

### What "Full Replacement" Means

PUT replaces the entire resource. You must send every field — even those you're not changing. The server discards the old record completely and stores whatever you send.

```
Before PUT:                       Client sends PUT with:
──────────────────────            ──────────────────────
{                                 {
  "name": "Ravi Kumar",             "name": "Ravi Kumar",    ← unchanged
  "age": 35,                        "age": 36,               ← birthday!
  "city": "Mumbai",                 "city": "Pune",          ← moved
  "weight": 78.5,                   "weight": 78.5,          ← same
  "height": 175.0,                  "height": 175.0,         ← same
  "smoker": false                   "smoker": false          ← same
}                                 }

After PUT:
{ "name": "Ravi Kumar", "age": 36, "city": "Pune", "weight": 78.5, ... }
The old record is completely gone. The new record is stored.
```

### When to Use PUT

PUT is appropriate when:
- The client has the complete current state of the resource
- The client wants to replace everything at once
- Example: "Edit Profile" form where the user sees all fields and clicks Save

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, computed_field

app = FastAPI()

class PatientUpdate(BaseModel):
    """
    For PUT, all fields are REQUIRED.
    No Optional, no defaults.
    The client must provide everything.
    """
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=150)
    city: str
    weight: float = Field(gt=0)
    height: float = Field(gt=0)
    smoker: bool   # no default — client must explicitly send true or false
    
    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / ((self.height / 100) ** 2), 2)


@app.put("/patients/{patient_id}")
def replace_patient(patient_id: str, patient: PatientUpdate):
    """
    Completely replace an existing patient's record.
    
    The patient_id in the URL tells us WHICH patient to update.
    The request body tells us WHAT the new values should be.
    """
    db = load_db()
    
    if patient_id not in db:
        # Can't replace something that doesn't exist
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found. Use POST /patients to create a new patient."
        )
    
    # Build completely new record from the client's data
    new_record = patient.model_dump()   # all fields the client sent
    new_record["id"] = patient_id       # preserve the server-assigned ID
    
    # Overwrite — old record is gone, new record takes its place
    db[patient_id] = new_record
    save_db(db)
    
    return {"message": "Patient record replaced", "patient": new_record}
```

---

## PATCH — Partial Update

### What "Partial Update" Means

PATCH is more practical than PUT in most real applications. You only send the fields you want to change. Everything else stays exactly as it was.

```
Before PATCH:                     Client sends PATCH with:
──────────────────────            ──────────────────────
{                                 {
  "name": "Ravi Kumar",             "city": "Chennai"    ← only this changed
  "age": 35,                    }
  "city": "Mumbai",           
  "weight": 78.5,             
  "height": 175.0,
  "smoker": false
}                                 

After PATCH:
{
  "name": "Ravi Kumar",     ← untouched
  "age": 35,                ← untouched
  "city": "Chennai",        ← updated!
  "weight": 78.5,           ← untouched
  "height": 175.0,          ← untouched
  "smoker": false           ← untouched
}
```

### The Critical Implementation Detail: `exclude_unset=True`

This is the most important concept in implementing PATCH correctly:

```python
from typing import Optional

class PatientPatch(BaseModel):
    """
    For PATCH, EVERY field is Optional.
    The client sends only what they want to change.
    Unsent fields keep their existing values in the database.
    """
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    smoker: Optional[bool] = None
```

Now, the problem and solution:

```python
# Client sends: {"city": "Chennai"}
updates = PatientPatch(city="Chennai")

# Problem: model_dump() without exclude_unset=True
updates.model_dump()
# → {"name": None, "age": None, "city": "Chennai", "weight": None, "height": None, "smoker": None}
# Merging this would set name=None, age=None, etc. — DELETING existing data! ❌

# Solution: model_dump(exclude_unset=True)
updates.model_dump(exclude_unset=True)
# → {"city": "Chennai"}
# Only the field the client actually sent. ✅
```

`exclude_unset=True` is the difference between "update only what was sent" and "overwrite everything with None."

```python
@app.patch("/patients/{patient_id}")
def update_patient_fields(patient_id: str, updates: PatientPatch):
    """
    Update only the specified fields.
    All other fields remain exactly as they are.
    """
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    
    existing = db[patient_id]   # get current record
    
    # Get ONLY what the client explicitly sent
    changed = updates.model_dump(exclude_unset=True)
    
    # Merge: keep existing, overwrite only changed fields
    existing.update(changed)
    
    db[patient_id] = existing
    save_db(db)
    
    return {"message": "Patient updated", "updated_fields": list(changed.keys()), "patient": existing}
```

---

## DELETE — Removing Resources

### Hard Delete vs Soft Delete

**Hard delete** permanently removes the record from the database. Simple but irreversible.

**Soft delete** marks the record as deleted (sets `is_deleted = true`) without actually removing it. The record stays in the database, allowing:
- Audit trails ("P001 was deleted by admin on 2024-01-15")
- Undo operations ("undelete" for admin users)
- Regulatory compliance (some industries require data retention)

For learning, we'll implement hard delete. For production, consider soft delete.

```python
from fastapi import Response

@app.delete("/patients/{patient_id}", status_code=204)
def delete_patient(patient_id: str):
    """
    Permanently delete a patient record.
    
    Returns HTTP 204 No Content on success.
    
    Why 204 and not 200?
    - 200 OK: "here's your result" (implies a response body)
    - 204 No Content: "done, nothing to return" (empty body)
    
    After deletion, there's no patient to return. 204 is semantically correct.
    """
    db = load_db()
    
    if patient_id not in db:
        # The patient doesn't exist — tell the client explicitly
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' does not exist."
        )
    
    # Capture the name for the response message
    deleted_name = db[patient_id]["name"]
    
    # Remove from the dictionary
    del db[patient_id]
    save_db(db)
    
    # IMPORTANT: return Response(status_code=204), not return None
    # If you return None, FastAPI may try to serialize it → conflict with 204
    return Response(status_code=204)
```

### Soft Delete Pattern

```python
from datetime import datetime, timezone

class Patient(BaseModel):
    ...
    is_deleted: bool = False
    deleted_at: Optional[str] = None

@app.delete("/patients/{patient_id}")
def soft_delete_patient(patient_id: str):
    """
    Mark patient as deleted without removing the record.
    Preserves audit history and allows recovery.
    """
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    
    if db[patient_id].get("is_deleted"):
        raise HTTPException(
            status_code=410,   # 410 Gone: resource existed but was deleted
            detail=f"Patient '{patient_id}' has already been deleted"
        )
    
    db[patient_id]["is_deleted"] = True
    db[patient_id]["deleted_at"] = datetime.now(timezone.utc).isoformat()
    save_db(db)
    
    return {"message": f"Patient '{patient_id}' has been deactivated"}

# Modify GET to exclude deleted records by default
@app.get("/patients")
def list_patients(include_deleted: bool = False):
    db = load_db()
    patients = list(db.values())
    
    if not include_deleted:
        # Filter out soft-deleted records for normal queries
        patients = [p for p in patients if not p.get("is_deleted", False)]
    
    return patients
```

---

## The Complete Patient Management API

Here is the entire API in one clean file with all six endpoints:

```python title="main.py"
import json
from typing import Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, computed_field

app = FastAPI(
    title="Patient Management API",
    description="Full CRUD API for managing patient records",
    version="1.0.0"
)

DB_FILE = "patients.json"

# ─── Utilities ──────────────────────────────────────────────────

def load_db() -> dict:
    with open(DB_FILE) as f:
        return json.load(f)

def save_db(data: dict):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─── Schemas ────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
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

# ─── Endpoints ──────────────────────────────────────────────────

@app.get("/patients", tags=["patients"])
def list_patients(include_deleted: bool = False):
    """List all patients. Use ?include_deleted=true for soft-deleted records."""
    db = load_db()
    patients = list(db.values())
    if not include_deleted:
        patients = [p for p in patients if not p.get("is_deleted", False)]
    return patients

@app.get("/patients/{pid}", tags=["patients"])
def get_patient(pid: str):
    """Get a specific patient by ID."""
    db = load_db()
    if pid not in db:
        raise HTTPException(404, f"Patient '{pid}' not found")
    if db[pid].get("is_deleted"):
        raise HTTPException(410, f"Patient '{pid}' has been deleted")
    return db[pid]

@app.post("/patients", status_code=201, tags=["patients"])
def create_patient(patient: PatientCreate):
    """Create a new patient record."""
    db = load_db()
    pid = f"P{len(db)+1:03d}"
    record = {**patient.model_dump(), "id": pid, "is_deleted": False}
    db[pid] = record
    save_db(db)
    return record

@app.put("/patients/{pid}", tags=["patients"])
def replace_patient(pid: str, patient: PatientCreate):
    """Completely replace a patient's record."""
    db = load_db()
    if pid not in db:
        raise HTTPException(404, "Patient not found")
    db[pid] = {**patient.model_dump(), "id": pid, "is_deleted": False}
    save_db(db)
    return db[pid]

@app.patch("/patients/{pid}", tags=["patients"])
def patch_patient(pid: str, updates: PatientPatch):
    """Update only the provided fields."""
    db = load_db()
    if pid not in db:
        raise HTTPException(404, "Patient not found")
    db[pid].update(updates.model_dump(exclude_unset=True))
    save_db(db)
    return db[pid]

@app.delete("/patients/{pid}", status_code=204, tags=["patients"])
def delete_patient(pid: str):
    """Soft-delete a patient (marks as deleted, preserves record)."""
    db = load_db()
    if pid not in db:
        raise HTTPException(404, "Patient not found")
    if db[pid].get("is_deleted"):
        raise HTTPException(410, "Patient already deleted")
    db[pid]["is_deleted"] = True
    db[pid]["deleted_at"] = datetime.now(timezone.utc).isoformat()
    save_db(db)
    return Response(status_code=204)
```

---

## Q&A

**Q: When should I choose PUT over PATCH?**

Choose PUT when: the client always sends the complete current state (like a settings page that shows all fields). Choose PATCH when: the client sends only what changed (like a "change city" form). In practice, PATCH is more common because it's more bandwidth-efficient and doesn't require the client to know all current values before updating.

**Q: Why does `return Response(status_code=204)` work better than `return None`?**

When you specify `status_code=204` on the decorator and `return None`, FastAPI may serialize `null` into the response body, which contradicts the "No Content" semantics. `Response(status_code=204)` explicitly creates an empty HTTP response with exactly the right code and no body.

**Q: What HTTP status code means "resource existed but was deleted"?**

`410 Gone`. It's different from `404 Not Found` (never existed) — `410` says "this used to exist but was permanently removed." Use it when you implement soft delete and a client tries to access a deleted record.

**Q: What if two requests try to delete the same patient simultaneously?**

With a JSON file "database," the second delete would get a 404 (first delete already removed it). With a real database and transactions, this is handled atomically. This is one of many reasons why JSON files are learning tools only — use PostgreSQL in production.

**Q: Should DELETE endpoints return the deleted resource?**

Convention says no — return 204 No Content. The resource no longer exists; returning it would be misleading. Some teams return 200 with `{"message": "deleted"}` — acceptable but non-standard. Stick with 204 for proper REST semantics.
