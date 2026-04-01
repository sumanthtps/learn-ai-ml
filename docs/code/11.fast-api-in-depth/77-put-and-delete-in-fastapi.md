---
title: "PUT and DELETE in FastAPI"
sidebar_position: 77
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 7. PUT & DELETE in FastAPI | Video 6 | CampusX
- YouTube video ID: `XVu22pTwWE8`
- Transcript pages in the uploaded PDF: 147-161

## Why this lesson matters

Reading and creating data are only half of CRUD. Real systems also need the ability to change existing data and remove outdated or invalid records.

This lesson covers update and delete operations, which are crucial because mistakes here can directly affect data integrity.

## What the transcript covers

The transcript explains:

- update operations using PUT
- deletion using DELETE
- the patient-management project continuation
- the need to identify which record is being modified
- basic not-found handling and state-change thinking

## PUT: the update method

PUT is commonly used to update or replace an existing resource.

### Example

```text
PUT /patients/3
```

Meaning:
- update patient number 3 using the provided body data

### FastAPI example

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
patients = {
    1: {'name': 'Asha', 'age': 25, 'city': 'Delhi'}
}

class PatientUpdate(BaseModel):
    name: str
    age: int
    city: str

@app.put('/patients/{patient_id}')
def update_patient(patient_id: int, patient: PatientUpdate) -> dict:
    if patient_id not in patients:
        return {'error': 'patient not found'}
    patients[patient_id] = patient.model_dump()
    return {'message': 'updated', 'data': patients[patient_id]}
```

## What PUT means conceptually

The classic REST interpretation is that PUT replaces the representation of the resource at that URL.

In practice, many beginner projects use PUT more loosely for general update operations. That is okay for learning, but you should understand the deeper idea.

### Strong interpretation
Client sends the full updated state.

### Looser beginner interpretation
Client sends the fields needed to perform an update.

In more mature API design, PATCH is often used for partial updates.

## PUT vs PATCH

This is an important production distinction.

### PUT
Used when the client sends the full intended new representation or a complete replacement-style payload.

### PATCH
Used when the client wants to change only some fields.

### Example

Full update with PUT:

```json
{
  "name": "Asha",
  "age": 26,
  "city": "Delhi"
}
```

Partial update with PATCH:

```json
{
  "age": 26
}
```

The transcript focuses on PUT, but in real systems you should know both patterns.

## Why the resource ID is usually in the path

When updating, the server must know which record to modify.

That is why the endpoint usually looks like:

```text
PUT /patients/3
```

The path identifies the target resource. The body carries the new data.

This separation is clean and consistent.

## DELETE: the remove method

DELETE is used when the client wants to remove a resource.

### Example

```text
DELETE /patients/3
```

Meaning:
- remove patient number 3

### FastAPI example

```python
@app.delete('/patients/{patient_id}')
def delete_patient(patient_id: int) -> dict:
    if patient_id not in patients:
        return {'error': 'patient not found'}
    deleted = patients.pop(patient_id)
    return {'message': 'deleted', 'data': deleted}
```

## Hard delete vs soft delete

The transcript introduces the basic delete concept, but in engineering practice you should distinguish:

### Hard delete
The record is physically removed.

### Soft delete
The record remains in storage but is marked inactive or deleted.

Example fields:

- `is_deleted = true`
- `deleted_at = timestamp`

Soft delete is often used when:

- auditability matters
- records may need recovery
- compliance rules require retention

## What should happen if the resource does not exist?

A mature API should not silently pretend everything worked.

Typical behavior:

- return `404 Not Found`
- include a clear error message

### FastAPI style with proper exceptions

```python
from fastapi import HTTPException

@app.delete('/patients/{patient_id}')
def delete_patient(patient_id: int) -> dict:
    if patient_id not in patients:
        raise HTTPException(status_code=404, detail='patient not found')
    patients.pop(patient_id)
    return {'message': 'deleted successfully'}
```

This is cleaner than returning informal error dictionaries.

## Idempotency in update and delete operations

### PUT
Usually considered idempotent. Sending the same full update again should lead to the same final state.

### DELETE
Often treated as idempotent in effect. Deleting the same resource twice should not produce inconsistent state, though the second call may return 404.

This matters in real systems because retries happen.

## Worked example

Suppose patient 5 changes city from Pune to Bengaluru.

Request:

```text
PUT /patients/5
```

Body:

```json
{
  "name": "Rohit",
  "age": 34,
  "city": "Bengaluru"
}
```

Server action:

- identify patient 5
- validate payload
- update stored data
- return confirmation

Deletion case:

```text
DELETE /patients/5
```

Server action:

- identify patient 5
- check existence and authorization
- remove or mark deleted
- return success or 404

## Why update and delete need stronger controls

Read endpoints are often less dangerous than write endpoints.

Update and delete operations can:

- corrupt data
- remove critical records
- trigger downstream workflows
- create audit and compliance issues

That is why production systems often apply stricter rules here:

- authorization checks
- audit logging
- confirmation workflows
- validation against current state

## Common mistakes beginners make

### 1. Using PUT for partial updates without understanding the semantic tradeoff
It may work, but the API contract becomes less precise.

### 2. Deleting without checking existence
Clients need predictable behavior when the resource is missing.

### 3. Forgetting authorization
Just because the route exists does not mean every user should be allowed to call it.

### 4. Returning vague responses
State-changing operations should communicate result clearly.

### 5. Mixing resource ID and body responsibilities
The path should identify the target. The body should carry updated values.

## Daily engineering additions beyond the transcript

### 1. Consider optimistic locking or version checks
In concurrent systems, two users may try to update the same record.

### 2. Add audit logs
Who changed what and when is often crucial.

### 3. Be explicit about partial vs full updates
Your API contract should make this clear.

### 4. Think about downstream dependencies
Deleting a user or patient record may affect other systems.

### 5. Design delete carefully in compliance-sensitive domains
Healthcare, finance, and legal systems often require retention policies.

## Important Q&A

### 1. Why is PUT associated with update operations?
Because it is commonly used to replace or update the representation of an existing resource.

### 2. Is DELETE always a physical deletion?
No. Many systems use soft delete for auditability and recovery.

### 3. What should happen if the record does not exist?
The API should usually return a `404 Not Found` style response.

### 4. Why do engineers distinguish PUT and PATCH?
Because full replacement and partial update are different semantics.

### 5. Why is authorization especially important for update and delete?
Because these operations directly change system state and can have serious consequences.

## Quick revision

- PUT updates an existing resource.
- DELETE removes a resource.
- Path identifies the target resource; body carries update data.
- Not-found handling should be explicit.
- In real systems, update and delete require audit, authorization, and careful design.
