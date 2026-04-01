---
title: "HTTP Methods in FastAPI"
sidebar_position: 73
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 3. HTTP Methods in FastAPI | Video 3 | CampusX
- YouTube video ID: `O8KrViWNhOM`
- Transcript pages in the uploaded PDF: 51-64

## Why this lesson matters

This lesson moves from introduction into actual API design. The transcript starts a patient-management style project and uses that project to explain why different HTTP methods exist.

If every endpoint were just "one URL that does everything," APIs would become ambiguous and hard to reason about. HTTP methods solve that by giving semantic meaning to requests.

## What the transcript covers

The transcript introduces:

- a doctor-patient data management project
- the idea of resources in API design
- GET as a read operation
- using a JSON file as a simple data source for learning
- the mapping between user intention and HTTP methods

## First principle: APIs are not just paths, they are method plus path

These two are different operations even though the path is the same:

- `GET /patients`
- `POST /patients`

The path identifies the resource area. The HTTP method tells the server what kind of action is intended.

This is a key design idea.

## The project context from the transcript

The transcript uses a healthcare-like example where doctors want to maintain patient information digitally instead of relying on scattered paper records.

That gives a very natural CRUD problem:

- create patient data
- read patient data
- update patient data
- delete patient data

Those four actions map neatly to common HTTP methods.

## Resource-first thinking

A good beginner mistake to avoid is designing endpoints around verbs first, such as:

- `/getPatient`
- `/createPatient`
- `/deletePatient`

A more standard REST-style approach is to think in terms of resources:

- `/patients`
- `/patients/{patient_id}`

Then use HTTP methods to express the action.

### Example

- `GET /patients` - get all patients
- `GET /patients/3` - get patient 3
- `POST /patients` - create a patient
- `PUT /patients/3` - update patient 3
- `DELETE /patients/3` - delete patient 3

This makes the API easier to understand and scale.

## GET method: what it means

The transcript starts with read-style behavior, which is appropriate because GET is the method used to retrieve data.

### Characteristics of GET

- used to fetch data
- should not modify server state
- often used for listing or reading resources
- usually accepts input through path parameters or query parameters

### Example

```python
from fastapi import FastAPI

app = FastAPI()

patients = [
    {'id': 1, 'name': 'Asha', 'city': 'Delhi'},
    {'id': 2, 'name': 'Ravi', 'city': 'Pune'}
]

@app.get('/patients')
def get_patients() -> list[dict]:
    return patients
```

This endpoint returns data. It does not add, edit, or remove anything.

## Why GET should not change data

This is not just a style preference. It matters for predictable system behavior.

Browsers, caches, monitoring tools, and clients often assume GET is safe to repeat.

If a GET request secretly deletes data or increments counters in a critical workflow, it creates confusing and dangerous behavior.

### Safe operation
`GET /patients/2`

### Unsafe misuse
`GET /patients/2/delete`

The second pattern may work technically, but it violates expected semantics.

## The five common HTTP methods you should know

### GET
Fetch data.

### POST
Create a new resource or trigger processing.

### PUT
Replace or update a resource, usually by sending a full updated representation.

### PATCH
Partially update a resource.

### DELETE
Remove a resource.

The transcript focuses first on GET, but the full mental model should be learned together.

## CRUD mapping

CRUD stands for:

- Create
- Read
- Update
- Delete

Typical mapping:

- Create -> POST
- Read -> GET
- Update -> PUT or PATCH
- Delete -> DELETE

This is one of the most foundational mappings in API work.

## Example using the patient project

### Read all patients

```python
@app.get('/patients')
def get_patients() -> list[dict]:
    return patients
```

### Read one patient

```python
@app.get('/patients/{patient_id}')
def get_patient(patient_id: int) -> dict:
    for patient in patients:
        if patient['id'] == patient_id:
            return patient
    return {'error': 'patient not found'}
```

### Create

```python
@app.post('/patients')
def create_patient(patient: dict) -> dict:
    patients.append(patient)
    return patient
```

The transcript later covers POST and update operations separately, but seeing the mapping early helps.

## Why learning with a JSON file is useful

The transcript uses a simple JSON file or small in-memory data source instead of a full database in the early stage.

That is pedagogically useful because it isolates API concepts.

You can focus on:

- route definitions
- method meaning
- request and response flow
- parameter handling

without getting distracted by SQL, migrations, or ORM configuration.

### But remember
In production, a file-based approach is rarely enough for concurrent multi-user systems.

## Idempotency: an important concept for engineers

This is not always taught early, but it is valuable.

### Idempotent operation
If doing the same request multiple times results in the same final state, the operation is idempotent.

Examples:

- GET is generally idempotent
- PUT is generally considered idempotent
- DELETE is usually treated as idempotent in effect
- POST is often not idempotent

### Why it matters
Retries happen in distributed systems. If a network retry occurs, the method semantics can affect data integrity.

## Status codes: method semantics are not enough

A good API should pair methods with sensible status codes.

Examples:

- `200 OK` - successful read
- `201 Created` - successful creation
- `204 No Content` - successful delete with no body
- `404 Not Found` - resource missing
- `400 Bad Request` - malformed input

A route is clearer when method, path, and status codes all align.

## Worked scenario

Suppose a doctor app needs to retrieve patient data for follow-up.

### Good design

```text
GET /patients/17
```

Why it is good:

- `patients` clearly identifies the resource collection
- `17` identifies a specific member
- GET tells the server the client only wants to read

### Poor design

```text
POST /getPatientData?id=17
```

Why it is poor:

- POST is semantically odd for a read-only operation
- endpoint naming becomes inconsistent
- tooling and docs become less intuitive

## Common mistakes beginners make

### 1. Treating HTTP methods as optional decoration
They are part of the API contract, not random labels.

### 2. Using POST for everything
This makes the API harder to understand and integrate.

### 3. Putting actions into resource names unnecessarily
Prefer resource-oriented naming where possible.

### 4. Letting GET modify data
That creates surprising behavior and breaks expectations.

### 5. Returning success responses without appropriate status codes
Clients often rely on status codes for control flow.

## Daily engineering additions beyond the transcript

### 1. Think about cacheability
GET requests can sometimes be cached, which is useful for read-heavy workloads.

### 2. Think about auditability
Update and delete operations often require stronger auditing than read operations.

### 3. Think about retries
Idempotent methods behave better under network retries.

### 4. Think about observability
Logs should record method, path, status, and latency.

### 5. Think about authorization
Reading may require different permissions than updating or deleting.

## Important Q&A

### 1. Why is GET the right method for viewing patient data?
Because GET is meant for retrieval without changing server state.

### 2. Why do APIs need multiple HTTP methods?
Because the same resource may support different kinds of operations, and method semantics make those intentions explicit.

### 3. Why is resource-based naming useful?
It creates consistency and makes APIs easier to understand, document, and extend.

### 4. Why is using POST for all operations a bad idea?
Because it hides intent, weakens semantics, and makes the API harder to reason about.

### 5. What is the relationship between CRUD and HTTP methods?
CRUD describes the business actions, and HTTP methods provide a standard way to express those actions at the protocol level.

## Quick revision

- API design is method plus path, not path alone.
- GET is for reading.
- Resource-oriented naming is cleaner than action-heavy naming.
- HTTP methods make APIs predictable.
- File-based examples are good for learning, but production usually needs stronger persistence.
