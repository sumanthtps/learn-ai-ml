---
id: path-query-params
title: "04 · Path & Query Parameters"
sidebar_label: "04 · Path & Query Params"
sidebar_position: 4
tags: [path-params, query-params, url, routing, fastapi, beginner]
---

# Path & Query Parameters in FastAPI

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=VVVKEfhXCQ4) · **Series:** FastAPI for ML – CampusX

---

## Two Ways to Put Data in a URL

When data is simple enough to fit in the URL (a patient ID, a filter value, a sort direction), you don't need a request body. FastAPI supports two URL-based mechanisms:

**Path parameters** — embedded in the URL path itself:
```
/patients/P001          → patient_id = "P001"
/models/v2/info         → version = "v2"
/hospitals/H01/patients/P003  → hospital_id = "H01", patient_id = "P003"
```

**Query parameters** — appended after `?` in the URL:
```
/patients?city=Mumbai&smoker=false    → city = "Mumbai", smoker = False
/patients?sort_by=age&order=desc      → sort_by = "age", order = "desc"
/patients?page=2&page_size=20         → page = 2, page_size = 20
```

Think of it this way:
- **Path params** identify *which* resource you want (a specific patient)
- **Query params** describe *how* you want the resource(s) returned (filtered, sorted, paginated)

---

## Path Parameters — Identifying Specific Resources

A path parameter is a variable segment in the URL, written with curly braces `{name}` in your route definition.

### Basic Usage

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    """
    {patient_id} in the decorator is the placeholder.
    'patient_id: str' in the function signature is the Python variable.
    FastAPI extracts the URL segment and passes it to the function.
    
    GET /patients/P001  → patient_id = "P001"
    GET /patients/abc   → patient_id = "abc"
    GET /patients/123   → patient_id = "123" (still a string!)
    """
    db = load_db()
    if patient_id not in db:
        raise HTTPException(404, f"Patient '{patient_id}' not found")
    return db[patient_id]
```

### Type Validation in Path Parameters

One of FastAPI's best features: it converts and validates path parameters based on their type annotation.

```python
@app.get("/items/{item_id}")
def get_item(item_id: int):    # ← type annotation: int, not str
    """
    FastAPI extracts the URL segment as a string ("42"),
    then converts it to an integer (42) based on your type annotation.
    
    If conversion fails, FastAPI returns 422 automatically.
    
    GET /items/42    → item_id = 42  (int)  ✅
    GET /items/abc   → 422 Unprocessable Entity ❌
    """
    return {"id": item_id, "python_type": type(item_id).__name__}
```

Compare to the old way (Flask/manual):
```python
# Old way: manual validation
item_id_str = request.args.get("id")
try:
    item_id = int(item_id_str)
except (TypeError, ValueError):
    return jsonify({"error": "id must be an integer"}), 422

# FastAPI way: type annotation does it all
def get_item(item_id: int):  # done!
```

### Path Parameters with Constraints

Use `Path()` for additional validation beyond type:

```python
from fastapi import Path

@app.get("/patients/{patient_id}")
def get_patient(
    patient_id: str = Path(
        min_length=4,
        max_length=10,
        pattern=r"^P\d{3}$",      # regex: must be "P" followed by 3 digits
        description="Patient ID in format P001 through P999",
        examples=["P001", "P042"]  # shown in Swagger UI
    )
):
    """
    Accepted:  P001, P042, P999
    Rejected:  P01 (too short), P0001 (doesn't match pattern)
               abc (no leading P), 001 (no leading P)
    """
    ...
```

### Multiple Path Parameters

```python
@app.get("/hospitals/{hospital_id}/patients/{patient_id}")
def get_patient_in_hospital(hospital_id: str, patient_id: str):
    """
    Hierarchical resources use nested path parameters.
    GET /hospitals/H01/patients/P042
    → hospital_id = "H01"
    → patient_id  = "P042"
    """
    return {"hospital": hospital_id, "patient": patient_id}
```

---

## The Route Order Trap — A Critical Gotcha

When you have both a literal route and a parameterized route, **order matters**.

```python
# ❌ WRONG ORDER — /patients/statistics is NEVER reached
@app.get("/patients/{patient_id}")    # FastAPI sees /patients/statistics
def get_patient(patient_id: str):     # and treats "statistics" as the patient_id!
    ...

@app.get("/patients/statistics")      # This route shadows — never matched
def get_statistics():
    ...
```

```python
# ✅ CORRECT ORDER — specific literals BEFORE parameterized routes
@app.get("/patients/statistics")      # checked first — matches /patients/statistics
def get_statistics():
    ...

@app.get("/patients/{patient_id}")    # checked second — matches everything else
def get_patient(patient_id: str):
    ...
```

FastAPI matches routes in the order they're defined. Put specific literal paths before parameterized ones.

---

## Query Parameters — Filtering, Sorting, Paginating

Query parameters appear after `?` in the URL. In FastAPI, any function parameter that is **not in the URL path** and **not a Pydantic model** becomes a query parameter automatically.

### Basic Query Parameters

```python
@app.get("/patients")
def list_patients(
    sort_by: str = "name",     # default value → query param is optional
    order: str = "asc",        # if not in URL, uses "asc"
):
    """
    GET /patients                 → sort_by="name", order="asc"
    GET /patients?sort_by=age     → sort_by="age",  order="asc"
    GET /patients?sort_by=age&order=desc → sort_by="age", order="desc"
    """
    patients = list(load_db().values())
    
    reverse = (order == "desc")
    patients.sort(key=lambda p: p.get(sort_by, ""), reverse=reverse)
    
    return patients
```

### Optional Query Parameters

```python
from typing import Optional

@app.get("/patients")
def list_patients(
    city: Optional[str] = None,     # not required — defaults to None
    smoker: Optional[bool] = None,  # not required — defaults to None
):
    """
    GET /patients                        → all patients
    GET /patients?city=Mumbai            → only Mumbai patients
    GET /patients?smoker=true            → only smokers
    GET /patients?city=Delhi&smoker=false → non-smokers in Delhi
    """
    patients = list(load_db().values())
    
    # Only filter if the parameter was provided
    if city is not None:
        patients = [p for p in patients if p.get("city") == city]
    if smoker is not None:
        patients = [p for p in patients if p.get("smoker") == smoker]
    
    return patients
```

### Required Query Parameters (No Default)

```python
@app.get("/patients/search")
def search_patients(q: str):    # no default → REQUIRED
    """
    GET /patients/search?q=ravi  → searches for "ravi"
    GET /patients/search         → 422 Error: "q" field required
    """
    patients = list(load_db().values())
    return [p for p in patients if q.lower() in p["name"].lower()]
```

### Query Parameters with Validation

```python
from fastapi import Query
from typing import Literal

@app.get("/patients")
def list_patients(
    sort_by: Literal["name", "age", "city", "bmi"] = Query(
        default="name",
        description="Field to sort results by"
    ),
    order: Literal["asc", "desc"] = Query(
        default="asc",
        description="Sort direction"
    ),
    page: int = Query(
        default=1,
        ge=1,             # must be ≥ 1 (no page 0)
        description="Page number (1-indexed)"
    ),
    page_size: int = Query(
        default=20,
        ge=1,             # at least 1 result
        le=100,           # at most 100 results
        description="Number of results per page"
    ),
):
    """
    Literal["name", "age", "city", "bmi"] does two things:
    1. In Swagger UI: renders as a dropdown (not free text)
    2. At runtime: rejects any value not in the list with 422
    
    GET /patients?sort_by=invalid   → 422 (not in Literal list)
    GET /patients?page_size=500     → 422 (exceeds le=100)
    """
    patients = list(load_db().values())
    
    # Sort
    patients.sort(
        key=lambda p: p.get(sort_by, ""),
        reverse=(order == "desc")
    )
    
    # Paginate
    total = len(patients)
    start = (page - 1) * page_size
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "data": patients[start:start + page_size]
    }
```

---

## A Full Search Endpoint

Combining all parameters into a production-ready search endpoint:

```python
@app.get("/patients")
def search_patients(
    # Filtering
    city: Optional[str] = Query(None, description="Filter by city"),
    smoker: Optional[bool] = Query(None, description="Filter by smoking status"),
    min_age: Optional[int] = Query(None, ge=0, description="Minimum age"),
    max_age: Optional[int] = Query(None, le=150, description="Maximum age"),
    
    # Sorting
    sort_by: Literal["name", "age", "city"] = Query("name"),
    order: Literal["asc", "desc"] = Query("asc"),
    
    # Pagination
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    patients = list(load_db().values())
    
    # Apply filters (only those provided)
    if city:
        patients = [p for p in patients if city.lower() in p.get("city", "").lower()]
    if smoker is not None:
        patients = [p for p in patients if p.get("smoker") == smoker]
    if min_age is not None:
        patients = [p for p in patients if p.get("age", 0) >= min_age]
    if max_age is not None:
        patients = [p for p in patients if p.get("age", 0) <= max_age]
    
    # Sort
    patients.sort(key=lambda p: p.get(sort_by, ""), reverse=(order == "desc"))
    
    # Paginate
    total = len(patients)
    start = (page - 1) * page_size
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "data": patients[start:start + page_size]
    }
```

---

## Path vs Query — The Decision Guide

```
Which do I use?
│
├── Am I identifying a SPECIFIC resource?
│   └── Yes → PATH PARAMETER
│          /patients/P001
│          /models/v2/info
│
└── Am I describing HOW to return a COLLECTION?
    ├── Filtering  → QUERY PARAM  ?city=Mumbai&smoker=false
    ├── Sorting    → QUERY PARAM  ?sort_by=age&order=desc
    ├── Pagination → QUERY PARAM  ?page=2&page_size=20
    └── Searching  → QUERY PARAM  ?q=ravi+kumar
```

Never put sensitive data (passwords, tokens) in URLs — they appear in server logs and browser history. Use request body or headers for sensitive data.

---

## Q&A

**Q: What happens if a required path parameter is missing?**

The URL simply doesn't match the route pattern. FastAPI returns 404 Not Found — because the route `/patients` (without an ID) doesn't exist if your only route is `/patients/{patient_id}`. (To handle both, define separate routes.)

**Q: What if a query parameter has the wrong type?**

FastAPI returns 422 Unprocessable Entity with a detailed error. For example, `GET /patients?page=abc` when `page: int` is defined returns: `{"detail": [{"loc": ["query", "page"], "msg": "value is not a valid integer"}]}`.

**Q: How do I accept multiple values for the same query param?**

```python
from typing import List
@app.get("/patients/by-cities")
def get_by_cities(cities: List[str] = Query(default=[])):
    # GET /patients/by-cities?cities=Mumbai&cities=Delhi&cities=Pune
    # → cities = ["Mumbai", "Delhi", "Pune"]
```

**Q: Why does my `/patients/statistics` endpoint never get called?**

You likely defined `/patients/{patient_id}` before `/patients/statistics`. FastAPI matches routes in definition order, and `{patient_id}` greedily matches "statistics". Move the specific route above the parameterized one.

**Q: Can query parameters have special characters like spaces?**

Yes, but they must be URL-encoded. A space becomes `%20` or `+`. So `?name=Ravi Kumar` is sent as `?name=Ravi%20Kumar`. Most HTTP client libraries (Python's `requests`, JavaScript's `fetch` with URLSearchParams) handle encoding automatically.
