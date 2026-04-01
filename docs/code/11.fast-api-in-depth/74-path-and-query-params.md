---
title: "Path Parameters and Query Parameters in FastAPI"
sidebar_position: 74
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 4. Path & Query Params in FastAPI | Video 4 | CampusX
- YouTube video ID: `VVVKEfhXCQ4`
- Transcript pages in the uploaded PDF: 65-85

## Why this lesson matters

This lesson teaches one of the most practical URL design concepts in API work: when should data be part of the URL path, and when should it be sent as a query parameter?

Many beginners know the syntax but not the reasoning. The reasoning matters more than the syntax because good URL design affects readability, maintainability, docs clarity, and client understanding.

## What the transcript covers

The transcript explains:

- what path parameters are
- what query parameters are
- the difference between identifying a specific resource and modifying how results are fetched
- examples using the patient project
- how FastAPI automatically parses parameter values and enforces types

## Path parameter: the core idea

A path parameter is a dynamic segment inside the URL path used to identify a specific resource.

### Example

```text
/patients/3
```

Here, `3` is not just some input value. It identifies which patient resource you want.

### FastAPI example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/patients/{patient_id}')
def get_patient(patient_id: int) -> dict:
    return {'requested_patient_id': patient_id}
```

If the request is:

```text
GET /patients/3
```

FastAPI passes `3` into `patient_id`.

## Why path parameters are powerful

They make URLs expressive.

Compare:

- `/patients/3`
- `/get-patient?id=3`

Both can work, but the first usually reads more naturally when the client is asking for one specific resource.

## Query parameter: the core idea

A query parameter modifies or refines how data is fetched. It usually does not identify the primary resource itself.

### Example

```text
/patients?city=Delhi&sort_by=bmi
```

This still refers to the `patients` collection, but the client is requesting filtered or customized output.

### FastAPI example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/patients')
def list_patients(city: str | None = None, sort_by: str | None = None) -> dict:
    return {'city': city, 'sort_by': sort_by}
```

## The decision rule that engineers use

A very useful rule of thumb is:

### Use a path parameter when the value identifies *which* resource.

Examples:

- `/patients/12`
- `/orders/98`
- `/users/sumanth`

### Use a query parameter when the value controls *how* the resource should be returned.

Examples:

- `/patients?city=Pune`
- `/orders?status=paid`
- `/users?active=true&page=2`

This one rule solves most beginner confusion.

## Path vs query with concrete examples

### Example 1: specific patient

```text
GET /patients/7
```

Meaning:
- give me patient number 7

### Example 2: filter patient list

```text
GET /patients?city=Mumbai
```

Meaning:
- give me patients, but only those from Mumbai

### Example 3: combine both

```text
GET /patients/7?include_history=true
```

Meaning:
- identify patient 7 using the path
- modify the response using a query parameter

This pattern is very common in real systems.

## Why the transcript's patient example is useful

The transcript first shows an endpoint that returns all patients, then asks: what if we want just one chosen patient?

That question naturally leads to path parameters.

Then once individual access is clear, filtering and sorting become natural motivations for query parameters.

This is a good teaching order because it mirrors how APIs usually grow in real projects:

1. get everything
2. get one specific record
3. filter the list
4. sort or paginate the list

## Type validation in FastAPI

One of the nicest things about FastAPI is that path and query parameters can be type annotated directly.

### Example

```python
@app.get('/square/{number}')
def square(number: int) -> dict:
    return {'square': number * number}
```

If a client sends:

```text
GET /square/4
```

it works.

If the client sends something that cannot be treated as an integer, FastAPI returns a validation error.

That means less manual parsing code for you.

## Optional query parameters

Query parameters are often optional.

```python
@app.get('/patients')
def list_patients(city: str | None = None) -> dict:
    return {'city_filter': city}
```

If no city is provided, the endpoint can return all patients.

This makes query parameters well-suited for filtering, pagination, searching, and sorting.

## Multiple query parameters

```python
@app.get('/patients')
def list_patients(
    city: str | None = None,
    min_age: int | None = None,
    sort_by: str | None = None,
    page: int = 1
) -> dict:
    return {
        'city': city,
        'min_age': min_age,
        'sort_by': sort_by,
        'page': page
    }
```

This is very common in list endpoints.

## A subtle but important design principle

Path parameters usually reflect the resource hierarchy.

Examples:

- `/users/10/orders`
- `/users/10/orders/99`

This makes relationships explicit.

Query parameters usually reflect retrieval controls.

Examples:

- `/users/10/orders?status=shipped`
- `/users/10/orders?page=3`

Once you understand this distinction, URLs become much easier to design cleanly.

## Common mistakes beginners make

### 1. Putting filter-like values into the path
Bad:

```text
/patients/Delhi
```

This is confusing because `Delhi` is not a specific patient identifier.

Better:

```text
/patients?city=Delhi
```

### 2. Putting the primary identifier into the query when the endpoint is clearly about one resource
Possible but less expressive:

```text
/patients?id=7
```

Usually clearer:

```text
/patients/7
```

### 3. Forgetting that query parameters are strings in the URL until parsed
FastAPI helps parse them, but the underlying transport is textual.

### 4. Using too many overloaded query parameters without documentation
List endpoints can become hard to use if the parameter contract is unclear.

### 5. Mixing business rules with poor naming
Choose names that match the domain: `patient_id`, `city`, `sort_by`, `page`, `limit`.

## Worked example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/patients/{patient_id}')
def get_patient(patient_id: int, include_history: bool = False) -> dict:
    return {
        'patient_id': patient_id,
        'include_history': include_history
    }
```

Request:

```text
GET /patients/5?include_history=true
```

Interpretation:

- `5` identifies which patient
- `include_history=true` controls how much detail to include

That is the clean separation you want.

## Daily engineering additions beyond the transcript

### 1. Use pagination query parameters for list endpoints
Examples: `page`, `limit`, `offset`

### 2. Be careful with query parameter naming consistency
Do not mix `page_size`, `limit`, and `size` randomly across endpoints.

### 3. Validate allowed sort fields
Never blindly trust arbitrary `sort_by` values if they map to database fields.

### 4. Document optional parameters clearly
Auto docs help, but clear naming still matters.

### 5. Keep URLs readable
A technically valid URL can still be hard to understand if too much meaning is crammed into it.

## Important Q&A

### 1. When should I use a path parameter?
When the value identifies a specific resource or resource hierarchy.

### 2. When should I use a query parameter?
When the value modifies filtering, sorting, pagination, searching, or output style.

### 3. Can both be used together?
Yes. A path parameter can identify the resource, and query parameters can refine the response.

### 4. Why is `sort_by` usually a query parameter?
Because it changes how the data is returned, not which base resource is being addressed.

### 5. Does FastAPI validate path and query parameters?
Yes. If you use type hints, FastAPI parses and validates them automatically.

## Quick revision

- Path parameter = identifies resource.
- Query parameter = refines retrieval.
- `/patients/7` and `/patients?city=Pune` solve different problems.
- FastAPI can validate both using type hints.
- Good URL design is about semantics, not just syntax.
