---
title: "Pydantic Crash Course for FastAPI"
sidebar_position: 75
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 5. Pydantic Crash Course | Data Validation in Python | CampusX
- YouTube video ID: `lRArylZCeOs`
- Transcript pages in the uploaded PDF: 86-128

## Why this lesson matters

This is one of the most important lessons in the playlist. FastAPI becomes truly powerful when request and response data are expressed through schemas instead of being handled as loose dictionaries.

Pydantic gives FastAPI a formal way to say:

- what fields exist
- what type each field should be
- which fields are required
- which values are optional
- which constraints are allowed

Without this, APIs quickly become fragile.

## What the transcript covers

The transcript explains:

- the problem with weakly structured input handling
- why validation matters in Python APIs
- how Pydantic models are declared
- required and optional fields
- type validation and automatic parsing
- why Pydantic is deeply connected to FastAPI's developer experience

## The real problem Pydantic solves

Python is dynamically typed. That is productive, but it also means you can easily receive input in unexpected formats.

### Example without validation

```python
payload = {
    'age': 'twenty six',
    'smoker': 'sometimes',
    'bmi': 'high'
}
```

If your code expects numeric or boolean values, this payload can break logic or produce silent bugs.

A good API should fail early and clearly when the input contract is violated.

That is what Pydantic helps with.

## What is a schema?

A schema is a formal description of data structure.

It answers questions like:

- what fields are expected?
- which fields are required?
- what type should each field have?
- what defaults apply?
- what values are allowed?

Pydantic lets you express this schema in Python classes.

## Your first Pydantic model

```python
from pydantic import BaseModel

class Patient(BaseModel):
    name: str
    age: int
    city: str
    bmi: float
    smoker: bool
```

This model means:

- `name` must be text
- `age` must be an integer
- `city` must be text
- `bmi` must be numeric
- `smoker` must be boolean

## Why this is better than raw dictionaries

With raw dictionaries, every route has to manually check fields.

With a schema, you define the contract once and reuse it.

### Raw dictionary approach

```python
if 'age' not in payload:
    raise ValueError('age missing')
if not isinstance(payload['age'], int):
    raise ValueError('age must be int')
```

### Schema approach

```python
class Patient(BaseModel):
    age: int
```

Much less repetition. Much more clarity.

## Pydantic parsing and validation

Pydantic does not just store values. It validates and, where possible, parses them.

### Example

```python
from pydantic import BaseModel

class Patient(BaseModel):
    age: int
    bmi: float
```

Input:

```python
Patient(age='26', bmi='24.7')
```

Pydantic can often convert these into the right Python types.

But if conversion is not sensible, it raises a validation error.

This balance is useful because APIs often receive JSON where values look textual at first.

## Required vs optional fields

### Required field
A field with no default is required.

```python
class Patient(BaseModel):
    name: str
```

### Optional field
A field can be optional.

```python
class Patient(BaseModel):
    name: str
    notes: str | None = None
```

This means `notes` can be omitted or set to `None`.

## Default values

```python
class Patient(BaseModel):
    active: bool = True
```

If the client does not provide `active`, it defaults to `True`.

Defaults are useful, but use them carefully. A default should represent a safe business assumption, not just a convenient shortcut.

## Field constraints

Pydantic becomes much more useful when you add business-level restrictions.

```python
from pydantic import BaseModel, Field

class Patient(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=120)
    bmi: float = Field(gt=0)
```

### Meaning

- `min_length=2` - name must have at least 2 characters
- `ge=0` - age must be greater than or equal to 0
- `le=120` - upper age bound
- `gt=0` - BMI must be positive

This is much stronger than checking only primitive types.

## Why Pydantic matters so much in FastAPI

FastAPI uses these models to power multiple features at once:

- request body validation
- automatic conversion to Python objects
- generated OpenAPI schema
- interactive docs
- response model control

That means a single schema definition supports both correctness and developer productivity.

## Example inside FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Patient(BaseModel):
    name: str
    age: int
    city: str

@app.post('/patients')
def create_patient(patient: Patient) -> dict:
    return {
        'message': 'patient created',
        'patient': patient.model_dump()
    }
```

What happens?

- FastAPI reads the incoming JSON body
- Pydantic validates it against `Patient`
- if validation succeeds, `patient` becomes a Python object with attributes
- if validation fails, FastAPI returns a validation error automatically

## Nested models

Real APIs often contain nested structure.

```python
from pydantic import BaseModel

class Address(BaseModel):
    city: str
    pin_code: str

class Patient(BaseModel):
    name: str
    age: int
    address: Address
```

This is important because production payloads are rarely flat.

## List fields

```python
class Patient(BaseModel):
    name: str
    allergies: list[str] = []
```

This is useful for tags, labels, feature names, or multiple categories.

## Why schemas improve team communication

When data contracts are encoded formally, team communication improves.

Frontend, QA, backend, and ML engineers can all refer to the same structure.

Instead of saying:

"The body probably has age, smoker, maybe city, not sure if bmi is optional."

You can say:

"Use the `PredictionInput` schema."

That is much clearer.

## Pydantic outside FastAPI

Although this playlist uses Pydantic mainly through FastAPI, you should know it is also useful independently.

Examples:

- validating config files
- validating pipeline settings
- validating experiment metadata
- checking incoming event payloads
- parsing data from external services

It is a general Python data validation tool, not only a FastAPI add-on.

## Common mistakes beginners make

### 1. Using raw dictionaries everywhere
This loses the benefits of schemas and creates repetitive validation code.

### 2. Making everything optional
If everything is optional, your contract becomes weak and error-prone.

### 3. Confusing type hinting with actual validation
A plain Python class with type hints is not enough. Pydantic models actively validate.

### 4. Ignoring business constraints
Type checks alone are often insufficient. Age being an integer does not mean it is valid.

### 5. Returning internal fields accidentally
Response models should be carefully designed so you do not expose sensitive or irrelevant information.

## Worked example: ML prediction input

```python
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    age: int = Field(ge=18, le=100)
    bmi: float = Field(gt=0)
    smoker: bool
    city: str
```

Why this is good:

- bad ages are rejected early
- BMI must be positive
- the API contract is obvious
- docs can show expected fields automatically

## Daily engineering additions beyond the transcript

### 1. Separate input and output schemas
The data you accept is not always the same as the data you return.

### 2. Use explicit names
`PredictionInput` is clearer than `DataModel1`.

### 3. Keep schema ownership clean
Do not let schema definitions scatter randomly across files.

### 4. Think about schema versioning
As APIs evolve, field additions and removals must be managed carefully.

### 5. Validate as close to the boundary as possible
Bad input should be rejected at the API boundary, not deep inside business logic.

## Important Q&A

### 1. Why is dynamic typing not enough for production APIs?
Because APIs receive uncontrolled external input, and you need clear contract enforcement.

### 2. What does Pydantic mainly provide?
Structured schemas, validation, parsing, and better API contracts.

### 3. Why is Pydantic especially valuable in FastAPI?
Because FastAPI uses it for validation, docs, schema generation, and request parsing.

### 4. Can Pydantic be used outside FastAPI?
Yes. It is a general-purpose Python validation tool.

### 5. What is the biggest practical benefit of schemas?
They turn vague incoming data into explicit, validated, maintainable contracts.

## Quick revision

- Pydantic gives structure to API data.
- Schemas define fields, types, defaults, and constraints.
- FastAPI uses Pydantic heavily for request validation.
- Good schemas reduce bugs and improve team communication.
- Type checks alone are not enough; business constraints matter too.
