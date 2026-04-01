---
id: pydantic
title: "05 · Pydantic Crash Course"
sidebar_label: "05 · Pydantic"
sidebar_position: 5
tags: [pydantic, validation, schemas, types, basemodel]
---

# Pydantic Crash Course — Data Validation in Python

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=lRArylZCeOs) · **Series:** FastAPI for ML – CampusX

---

## Why Pydantic?

Python is dynamically typed. That's great for rapid prototyping, but in production:

```python
# This runs without error — but silently corrupts your data
patient = {"age": "forty-five", "weight": "heavy"}
model.predict(patient["age"] + 5)   # 💥 TypeError at runtime
```

**Pydantic** is a data validation library that enforces types **at runtime**:

```python
from pydantic import BaseModel

class Patient(BaseModel):
    age: int
    weight: float

p = Patient(age="forty-five", weight=80)  # 💥 ValidationError immediately
```

Pydantic is used by:
- **FastAPI** — for request/response validation
- **LangChain / LlamaIndex** — for structured LLM outputs
- **Hydra / Dynaconf** — for ML config files
- **SQLModel** — for database schemas

---

## BaseModel — The Core of Pydantic

```python
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class InsurancePremium(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class Patient(BaseModel):
    patient_id: str
    name: str
    age: int
    weight: float
    height: float
    smoker: bool
    city: str
    premium: Optional[InsurancePremium] = None   # optional field
```

---

## Type Coercion vs Strict Mode

By default, Pydantic **coerces** compatible types:

```python
class Item(BaseModel):
    count: int
    price: float

# Pydantic auto-converts compatible types
item = Item(count="5", price="19.99")   # strings → int/float ✅
print(item.count)   # 5 (int)
print(item.price)   # 19.99 (float)

# Strict mode — no coercion
class StrictItem(BaseModel):
    model_config = ConfigDict(strict=True)
    count: int

StrictItem(count="5")   # 💥 ValidationError — must be int
```

---

## Field — Adding Constraints and Metadata

```python
from pydantic import BaseModel, Field

class Patient(BaseModel):
    patient_id: str = Field(
        min_length=4,
        max_length=10,
        pattern=r"^P\d{3}$",
        description="Patient ID in format P001",
        examples=["P001", "P042"]
    )
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=150, description="Age in years")
    weight: float = Field(gt=0, le=700, description="Weight in kg")
    height: float = Field(gt=0, le=300, description="Height in cm")
    city: str = Field(default="Unknown")
    smoker: bool = Field(default=False)
```

### Field constraint shortcuts

| Constraint | Meaning |
|-----------|---------|
| `ge=0` | greater than or equal to 0 |
| `gt=0` | strictly greater than 0 |
| `le=100` | less than or equal to 100 |
| `lt=100` | strictly less than 100 |
| `min_length=3` | minimum string length |
| `max_length=50` | maximum string length |
| `pattern=r"^P\d{3}$"` | regex must match |

---

## Validators — Custom Validation Logic

### Field Validators

```python
from pydantic import BaseModel, field_validator

class Patient(BaseModel):
    name: str
    age: int
    weight: float
    height: float

    @field_validator("name")
    @classmethod
    def name_must_be_title_case(cls, v: str) -> str:
        return v.title()   # auto-capitalize names

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be non-negative")
        return v

    @field_validator("weight", "height")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Must be positive")
        return v
```

### Model Validators (cross-field validation)

```python
from pydantic import model_validator

class Patient(BaseModel):
    weight: float
    height: float
    bmi: Optional[float] = None

    @model_validator(mode="after")
    def compute_bmi(self) -> "Patient":
        # Automatically calculate BMI from weight and height
        height_m = self.height / 100
        self.bmi = round(self.weight / (height_m ** 2), 2)
        return self
```

---

## Computed Fields

```python
from pydantic import BaseModel, computed_field

class Patient(BaseModel):
    weight: float   # kg
    height: float   # cm

    @computed_field
    @property
    def bmi(self) -> float:
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)

    @computed_field
    @property
    def bmi_category(self) -> str:
        if self.bmi < 18.5: return "Underweight"
        elif self.bmi < 25:  return "Normal"
        elif self.bmi < 30:  return "Overweight"
        else:                return "Obese"
```

---

## Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    pin_code: str

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None

class Patient(BaseModel):
    id: str
    name: str
    address: Address         # nested model
    contact: ContactInfo     # nested model
    prescriptions: List[str] = []
```

Input JSON:
```json
{
  "id": "P001",
  "name": "Ramesh Kumar",
  "address": {
    "street": "12 MG Road",
    "city": "Mumbai",
    "pin_code": "400001"
  },
  "contact": {
    "email": "ramesh@example.com"
  }
}
```

---

## Serialization — Converting Models Back to Dicts/JSON

```python
patient = Patient(id="P001", name="Ramesh", age=45, ...)

# To dict
patient.model_dump()
# → {"id": "P001", "name": "Ramesh", ...}

# Exclude None values
patient.model_dump(exclude_none=True)

# Exclude specific fields
patient.model_dump(exclude={"password", "internal_notes"})

# Include only specific fields
patient.model_dump(include={"id", "name", "age"})

# To JSON string
patient.model_dump_json()
```

---

## Using Pydantic for ML Feature Schemas

```python
from pydantic import BaseModel, Field
from typing import Literal

class InsuranceInput(BaseModel):
    """Input features for the insurance premium prediction model."""
    
    age: int = Field(ge=18, le=100, description="Age of the insured person")
    sex: Literal["male", "female"] = Field(description="Biological sex")
    bmi: float = Field(ge=10.0, le=60.0, description="Body Mass Index")
    children: int = Field(ge=0, le=10, description="Number of dependent children")
    smoker: bool = Field(description="Whether the person smokes")
    region: Literal["northeast", "northwest", "southeast", "southwest"]

class InsuranceOutput(BaseModel):
    """Response from the insurance premium prediction model."""
    
    prediction: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
```

---

## Topics Not Covered in the Video

### model_config — Global Model Settings

```python
from pydantic import BaseModel, ConfigDict

class Patient(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,   # strip whitespace from all strings
        str_min_length=1,            # all strings must be non-empty
        validate_default=True,       # validate default values too
        frozen=True,                 # make instances immutable (hashable)
        extra="forbid",              # reject unknown fields
    )
    name: str
    age: int
```

### Discriminated Unions — Polymorphic Models

```python
from typing import Union, Annotated
from pydantic import BaseModel, Field

class DogInput(BaseModel):
    type: Literal["dog"]
    breed: str

class CatInput(BaseModel):
    type: Literal["cat"]
    indoor: bool

AnimalInput = Annotated[
    Union[DogInput, CatInput],
    Field(discriminator="type")
]
```

### Pydantic Settings — Environment Variable Parsing

```python
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    database_url: str
    api_key: str
    model_path: str = "artifacts/model.pkl"
    debug: bool = False
    max_workers: int = 4

    class Config:
        env_file = ".env"

settings = AppSettings()  # auto-reads from env or .env file
```

### Schema Export for API Contracts

```python
import json

# Export JSON Schema for documentation / client SDKs
print(json.dumps(InsuranceInput.model_json_schema(), indent=2))
```

---

## Q&A

**Q: Does Pydantic validate data on every access or only at creation?**
> By default, only at creation time (when the model instance is built). Use `model_config = ConfigDict(validate_assignment=True)` to also validate on field assignment.

**Q: What's the difference between `model_dump()` and `dict()`?**
> `dict()` was the Pydantic v1 method. `model_dump()` is the Pydantic v2 method. FastAPI now uses Pydantic v2. Use `model_dump()` in new code.

**Q: How does FastAPI use Pydantic internally?**
> When a POST request comes in with a JSON body, FastAPI passes the raw dict to your Pydantic model's constructor. If validation fails, FastAPI automatically returns a 422 response with detailed error messages. On success, your endpoint function receives a fully validated, typed Python object.

**Q: Should my request schema and response schema be the same Pydantic model?**
> Usually not. Request models often have fewer fields (you don't send the ID when creating) and response models might include computed fields (like BMI) or hide sensitive fields (passwords). Define separate `Input` and `Response` models.

**Q: How do I handle a field that can be multiple types?**
> Use `Union[int, str]` or `Optional[int]` (which is `Union[int, None]`). Pydantic validates against each type in order.

**Q: Can I use Pydantic outside of FastAPI?**
> Absolutely. It's a standalone library. Common uses: config validation, data pipeline validation, LLM structured outputs (LangChain), CLI argument parsing.

---

## Pydantic v1 vs v2 Migration Cheat Sheet

| v1 | v2 |
|----|-----|
| `from pydantic import validator` | `from pydantic import field_validator` |
| `@validator("field")` | `@field_validator("field")` |
| `patient.dict()` | `patient.model_dump()` |
| `patient.json()` | `patient.model_dump_json()` |
| `Patient.schema()` | `Patient.model_json_schema()` |
| `class Config:` | `model_config = ConfigDict(...)` |
| `@root_validator` | `@model_validator(mode="after")` |
