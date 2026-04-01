---
id: path-query-params
title: "04 · Path & Query Parameters"
sidebar_label: "04 · Path & Query Params"
sidebar_position: 4
tags: [path-params, query-params, url, routing, fastapi]
---

# Path & Query Parameters in FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=VVVKEfhXCQ4) · **Series:** FastAPI for ML – CampusX

---

## Two Ways to Pass Data in a URL

When data is too simple to deserve a request body, you embed it in the URL itself. FastAPI supports two URL-based ways to pass data:

| Type | Location | Example | Use Case |
|------|----------|---------|----------|
| **Path Parameter** | Part of the URL path | `/patients/P001` | Identify a specific resource |
| **Query Parameter** | After `?` in the URL | `/patients?sort_by=age&order=desc` | Filtering, sorting, pagination |

---

## Path Parameters

Path parameters are **dynamic segments** of the URL path used to identify a specific resource.

```
/patients/{patient_id}
          └──────────── path parameter
```

### Basic Usage

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    patients = load_patients()
    
    if patient_id not in patients:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found"
        )
    
    return patients[patient_id]
```

Call: `GET /patients/P001` → returns patient P001

### Type Validation in Path Parameters

FastAPI automatically validates and converts the type:

```python
# Integer path param — FastAPI rejects non-integer paths
@app.get("/items/{item_id}")
def get_item(item_id: int):    # automatically converted from string
    return {"id": item_id, "type": type(item_id).__name__}

# /items/42    → {"id": 42, "type": "int"}   ✅
# /items/abc   → 422 Unprocessable Entity    ❌
```

### Path Parameter with Constraints (Field validators)

```python
from fastapi import Path

@app.get("/patients/{patient_id}")
def get_patient(
    patient_id: str = Path(
        min_length=4,
        max_length=10,
        pattern=r"^P\d{3}$",   # must match P followed by 3 digits
        description="Patient ID in format P001"
    )
):
    ...
```

### Multiple Path Parameters

```python
@app.get("/hospitals/{hospital_id}/patients/{patient_id}")
def get_patient_in_hospital(hospital_id: str, patient_id: str):
    return {"hospital": hospital_id, "patient": patient_id}
```

---

## Query Parameters

Query parameters come **after the `?`** in the URL and are optional by default.

```
/patients?sort_by=age&order=asc&limit=10
           └──────────────────────────── query parameters
```

### Basic Usage

```python
# All parameters not in the path → automatically query params
@app.get("/patients")
def get_patients(sort_by: str = "name", order: str = "asc"):
    patients = load_patients()
    
    reverse = (order == "desc")
    sorted_patients = dict(
        sorted(
            patients.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=reverse
        )
    )
    return sorted_patients
```

Call: `GET /patients?sort_by=age&order=desc`

### Optional Query Parameters

```python
from typing import Optional

@app.get("/patients")
def get_patients(
    sort_by: Optional[str] = None,
    order: str = "asc",
    limit: Optional[int] = None,
    offset: int = 0
):
    patients = load_patients()
    items = list(patients.values())
    
    # Apply sorting if sort_by provided
    if sort_by:
        items.sort(key=lambda x: x.get(sort_by, 0), reverse=(order == "desc"))
    
    # Apply pagination
    if limit:
        items = items[offset:offset + limit]
    
    return items
```

### Query Parameters with Validation

```python
from fastapi import Query

@app.get("/patients")
def get_patients(
    sort_by: str = Query(
        default="name",
        enum=["name", "age", "weight", "height", "bmi"],
        description="Field to sort by"
    ),
    order: str = Query(
        default="asc",
        enum=["asc", "desc"]
    ),
    limit: int = Query(default=100, ge=1, le=1000)  # 1 ≤ limit ≤ 1000
):
    ...
```

---

## Path vs Query — When to Use Each

### Use Path Parameters for:
- Resource **identity** — `/patients/P001`, `/models/v2`
- Things that are **required** — a route without the ID wouldn't make sense
- Hierarchical resources — `/hospitals/H01/wards/W2/patients/P001`

### Use Query Parameters for:
- **Filtering** — `?city=Mumbai&smoker=false`
- **Sorting** — `?sort_by=age&order=desc`
- **Pagination** — `?page=2&limit=20`
- **Optional behavior** — `?include_deleted=true`
- **Search** — `?q=ramesh`

---

## Real-world Example: Patient Search Endpoint

```python
from typing import Optional, Literal
from fastapi import Query, Path

@app.get("/patients")
def search_patients(
    city: Optional[str] = Query(None, description="Filter by city"),
    smoker: Optional[bool] = Query(None, description="Filter by smoking status"),
    min_age: Optional[int] = Query(None, ge=0, description="Minimum age"),
    max_age: Optional[int] = Query(None, le=150, description="Maximum age"),
    sort_by: Literal["name", "age", "bmi"] = Query("name"),
    order: Literal["asc", "desc"] = Query("asc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    patients = list(load_patients().values())
    
    # Filter
    if city:
        patients = [p for p in patients if p.get("city") == city]
    if smoker is not None:
        patients = [p for p in patients if p.get("smoker") == smoker]
    if min_age:
        patients = [p for p in patients if p.get("age", 0) >= min_age]
    if max_age:
        patients = [p for p in patients if p.get("age", 0) <= max_age]
    
    # Sort
    patients.sort(key=lambda x: x.get(sort_by, ""), reverse=(order == "desc"))
    
    # Paginate
    start = (page - 1) * page_size
    return {
        "total": len(patients),
        "page": page,
        "page_size": page_size,
        "data": patients[start:start + page_size]
    }
```

---

## Topics Not Covered in the Video

### URL Encoding for Special Characters

Query parameters with special characters need URL encoding:

```
# Space → %20 or +
/patients?name=Ramesh%20Kumar

# In Python with requests library:
import requests
r = requests.get("/patients", params={"name": "Ramesh Kumar", "city": "New Delhi"})
# requests handles encoding automatically
```

### Multiple Values for the Same Query Parameter

```python
from typing import List

@app.get("/patients")
def get_patients_by_cities(
    cities: List[str] = Query(default=[])
):
    # /patients?cities=Mumbai&cities=Delhi&cities=Pune
    ...
```

### Enum Path Parameters

```python
from enum import Enum

class SortField(str, Enum):
    name = "name"
    age = "age"
    bmi = "bmi"

@app.get("/patients")
def get_patients(sort_by: SortField = SortField.name):
    ...
    # Swagger UI shows a dropdown of valid values
```

### Path Parameter Order Matters

```python
# WRONG — FastAPI will match /patients/me as patient_id="me"
@app.get("/patients/{patient_id}")
def get_patient(patient_id: str): ...

@app.get("/patients/me")   # ← never reached!
def get_current_patient(): ...

# CORRECT — put specific routes BEFORE parameterized ones
@app.get("/patients/me")   # ← must come first
def get_current_patient(): ...

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str): ...
```

---

## Q&A

**Q: What happens if a required path parameter is missing?**
> FastAPI returns `404 Not Found` — the URL simply doesn't match any route, since the route pattern requires that segment.

**Q: What happens if a required query parameter is missing?**
> If it has no default value: FastAPI returns `422 Unprocessable Entity — field required`. If it has a default: the default is used silently.

**Q: Can I use the same parameter name in both path and query?**
> No — that would cause confusion. If `patient_id` is in the path, don't also define `patient_id` as a query parameter.

**Q: How do I pass a list of values as query params?**
> Use `List[str]` type with `Query(default=[])`. Each value is a separate `?key=value` pair in the URL.

**Q: How should I design URLs for ML prediction APIs?**
> ```
> POST /v1/predict/insurance       # Insurance model
> POST /v1/predict/churn           # Churn model
> GET  /v1/models                  # List available models
> GET  /v1/models/{model_id}/info  # Model metadata
> ```

**Q: Is there a limit to URL length?**
> Browsers typically enforce ~2,000 chars. Servers allow up to ~8,000. If you're hitting limits, your data belongs in the request body, not query params.

---

## Quick Reference

```python
# Path parameter
@app.get("/resource/{id}")
def fn(id: str): ...

# Path parameter with validation
@app.get("/resource/{id}")
def fn(id: str = Path(min_length=3, pattern=r"^\w+$")): ...

# Optional query parameter
@app.get("/resource")
def fn(filter: Optional[str] = None): ...

# Required query parameter (no default)
@app.get("/resource")
def fn(q: str): ...   # 422 if missing

# Query parameter with constraints
@app.get("/resource")
def fn(limit: int = Query(10, ge=1, le=100)): ...

# Enum query parameter
@app.get("/resource")
def fn(sort: Literal["asc", "desc"] = "asc"): ...
```
