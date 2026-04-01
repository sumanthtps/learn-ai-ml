---
id: put-delete
title: "07 · PUT & DELETE in FastAPI"
sidebar_label: "07 · PUT & DELETE"
sidebar_position: 7
tags: [put, delete, patch, update, crud, fastapi]
---

# PUT & DELETE in FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=XVu22pTwWE8) · **Series:** FastAPI for ML – CampusX

---

## Completing CRUD: Update and Delete

By this point, we have `C`reate and `R`ead in our Patient Management API. Now we add `U`pdate and `D`elete to complete full CRUD.

```
✅ GET  /patients           → Read all patients
✅ GET  /patients/{id}      → Read one patient
✅ POST /patients           → Create patient
🔲 PUT  /patients/{id}      → Update patient
🔲 DELETE /patients/{id}    → Delete patient
```

---

## PUT — Full Update

`PUT` **replaces** the entire resource. The client sends all fields, even unchanged ones.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json

class PatientUpdate(BaseModel):
    """All fields required for full update."""
    name: str = Field(min_length=2)
    age: int = Field(ge=0, le=150)
    city: str
    weight: float = Field(gt=0)
    height: float = Field(gt=0)
    smoker: bool

@app.put("/patients/{patient_id}")
def update_patient(patient_id: str, patient: PatientUpdate):
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found"
        )
    
    # Full replace — overwrite all fields
    db[patient_id] = {
        **patient.model_dump(),
        "id": patient_id  # preserve the ID
    }
    save_db(db)
    
    return {"message": "Patient updated", "patient": db[patient_id]}
```

---

## PATCH — Partial Update

`PATCH` updates **only the fields provided**. Use `Optional` + `exclude_unset=True`.

```python
from typing import Optional

class PatientPatch(BaseModel):
    """All fields optional for partial update."""
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    smoker: Optional[bool] = None

@app.patch("/patients/{patient_id}")
def patch_patient(patient_id: str, updates: PatientPatch):
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    existing = db[patient_id]
    
    # Only update fields that were explicitly sent
    changed = updates.model_dump(exclude_unset=True)
    existing.update(changed)
    
    db[patient_id] = existing
    save_db(db)
    
    return {"message": "Patient updated", "patient": existing}
```

Request: `PATCH /patients/P001` with `{"city": "Pune"}` → only city changes.

---

## DELETE — Remove a Resource

```python
from fastapi import Response

@app.delete("/patients/{patient_id}", status_code=204)
def delete_patient(patient_id: str):
    db = load_db()
    
    if patient_id not in db:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found"
        )
    
    deleted_name = db[patient_id]["name"]
    del db[patient_id]
    save_db(db)
    
    return Response(status_code=204)  # 204 No Content — no body
```

> **Note:** HTTP 204 means success with no response body. Don't return data from a 204 endpoint — it will be ignored.

---

## Complete CRUD API — Full Code

```python title="main.py"
import json
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, computed_field

app = FastAPI(title="Patient Management API", version="1.0.0")

DB_PATH = "patients.json"

# ─── Utilities ──────────────────────────────────────────────────

def load_db() -> dict:
    with open(DB_PATH) as f:
        return json.load(f)

def save_db(data: dict):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

# ─── Schemas ────────────────────────────────────────────────────

class PatientBase(BaseModel):
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

class PatientCreate(PatientBase):
    pass

class PatientUpdate(PatientBase):
    pass  # All fields required (PUT semantics)

class PatientPatch(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    smoker: Optional[bool] = None

# ─── Endpoints ──────────────────────────────────────────────────

@app.get("/patients", tags=["patients"])
def get_all():
    return load_db()

@app.get("/patients/{patient_id}", tags=["patients"])
def get_one(patient_id: str):
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    return db[patient_id]

@app.post("/patients", status_code=201, tags=["patients"])
def create(patient: PatientCreate):
    db = load_db()
    pid = f"P{len(db)+1:03d}"
    db[pid] = {**patient.model_dump(), "id": pid}
    save_db(db)
    return {"id": pid, "message": "Created"}

@app.put("/patients/{patient_id}", tags=["patients"])
def update(patient_id: str, patient: PatientUpdate):
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    db[patient_id] = {**patient.model_dump(), "id": patient_id}
    save_db(db)
    return db[patient_id]

@app.patch("/patients/{patient_id}", tags=["patients"])
def patch(patient_id: str, updates: PatientPatch):
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    db[patient_id].update(updates.model_dump(exclude_unset=True))
    save_db(db)
    return db[patient_id]

@app.delete("/patients/{patient_id}", status_code=204, tags=["patients"])
def delete(patient_id: str):
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404, "Patient not found")
    del db[patient_id]
    save_db(db)
    return Response(status_code=204)
```

---

## Topics Not Covered in the Video

### Soft Delete vs Hard Delete

In production, you rarely permanently delete records. Use a **soft delete** pattern:

```python
class Patient(BaseModel):
    id: str
    name: str
    ...
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None

@app.delete("/patients/{patient_id}")
def soft_delete(patient_id: str):
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404)
    
    from datetime import datetime, timezone
    db[patient_id]["is_deleted"] = True
    db[patient_id]["deleted_at"] = datetime.now(timezone.utc).isoformat()
    save_db(db)
    return {"message": "Patient deactivated"}

# Modify GET to exclude soft-deleted records
@app.get("/patients")
def get_all(include_deleted: bool = False):
    db = load_db()
    if not include_deleted:
        return {k: v for k, v in db.items() if not v.get("is_deleted")}
    return db
```

### Optimistic Locking — Preventing Race Conditions

```python
class PatientUpdate(BaseModel):
    ...
    version: int  # client sends the version they last read

@app.put("/patients/{patient_id}")
def update_patient(patient_id: str, patient: PatientUpdate):
    db = load_db()
    current_version = db[patient_id].get("version", 0)
    
    if patient.version != current_version:
        raise HTTPException(
            status_code=409,
            detail="Conflict: patient was modified by another request"
        )
    
    db[patient_id] = {**patient.model_dump(), "version": current_version + 1}
    save_db(db)
    return db[patient_id]
```

### Bulk Operations

```python
from typing import List

@app.delete("/patients")
def bulk_delete(patient_ids: List[str]):
    db = load_db()
    missing = [pid for pid in patient_ids if pid not in db]
    
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Patients not found: {missing}"
        )
    
    for pid in patient_ids:
        del db[pid]
    save_db(db)
    
    return {"deleted": len(patient_ids)}
```

---

## Q&A

**Q: When should I use PUT vs PATCH?**
> `PUT` is semantically a full replacement — send all fields every time. `PATCH` is a partial update — send only changed fields. For most real applications, `PATCH` is more practical and bandwidth-efficient. Many teams use only `PATCH` and skip `PUT`.

**Q: Should DELETE return a body?**
> Best practice is `204 No Content` (no body). Some teams return `200 OK` with a confirmation message — that's acceptable too. Avoid returning the deleted resource in the body (it no longer exists).

**Q: What HTTP status code for "resource already exists" on POST?**
> `409 Conflict`. Use this when a POST would create a duplicate (e.g., duplicate patient name or email).

**Q: What if I want to update nested fields in a PATCH?**
> With `exclude_unset=True` on nested Pydantic models, only the top-level fields are tracked. For deep partial updates, you may need to handle the dict merge manually or use a library like `deepmerge`.

**Q: How do I handle concurrent writes to the JSON file?**
> A plain JSON file doesn't handle concurrent writes safely — multiple requests writing simultaneously can corrupt data. In production, use a real database (PostgreSQL, MongoDB) or at minimum a thread lock around file I/O. This JSON approach is for learning only.

**Q: How do I roll back a failed update?**
> Load the current state before modifying, apply changes, and only `save_db()` after all operations succeed. If something raises an exception mid-way, the in-memory changes are discarded without saving. For real transactions, use a database with transaction support.
