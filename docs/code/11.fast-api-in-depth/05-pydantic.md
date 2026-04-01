---
id: pydantic
title: "05 · Pydantic — Data Validation in Python"
sidebar_label: "05 · Pydantic"
sidebar_position: 5
tags: [pydantic, validation, schemas, types, basemodel, beginner]
---

# Pydantic — Data Validation in Python

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=lRArylZCeOs) · **Series:** FastAPI for ML – CampusX

---

## The Problem Pydantic Solves

Python is dynamically typed. This is great for notebooks. In production ML APIs, it's dangerous:

```python
# In a notebook — you control the inputs, so this is fine
def predict_premium(age, bmi, smoker):
    return model.predict([[age, bmi, int(smoker)]])[0]

# In a production API — clients send whatever they want
data = request.json()
# Client sent: {"age": "thirty-five", "bmi": "fat", "smoker": "sure"}
result = predict_premium(**data)
# Crashes deep inside sklearn:
# ValueError: could not convert string to float: 'thirty-five'
# The client sees: 500 Internal Server Error (unhelpful)
```

**Pydantic** is a data validation library that catches bad data **at the entry point** — before it ever reaches your ML model. It gives clients specific, actionable error messages. It converts compatible types automatically. It documents your expected data schema.

---

## BaseModel — The Foundation

Every Pydantic schema inherits from `BaseModel`. You define fields as class attributes with type annotations.

```python
from pydantic import BaseModel

class Patient(BaseModel):
    name: str       # type annotation — Pydantic uses this to validate
    age: int
    weight: float
    smoker: bool
```

When you create an instance, Pydantic immediately validates:

```python
# ✅ Valid data — creates successfully
p = Patient(name="Ravi Kumar", age=35, weight=78.5, smoker=False)
print(p.name)      # "Ravi Kumar"
print(p.age)       # 35 (int)

# ✅ Compatible types — Pydantic auto-converts
p = Patient(name="Ravi", age="35", weight="78.5", smoker="false")
#                         ^^^^      ^^^^^           ^^^^^^^
#                         strings!  string!         string!
print(p.age)       # 35 (int) — converted from "35"!
print(p.weight)    # 78.5 (float) — converted from "78.5"!
print(p.smoker)    # False (bool) — "false" → False

# ❌ Incompatible — raises ValidationError immediately
p = Patient(name="Ravi", age="thirty-five", weight="normal", smoker="maybe")
# pydantic_core.ValidationError: 3 validation errors for Patient
#   age: Input should be a valid integer [input_value='thirty-five']
#   weight: Input should be a valid number [input_value='normal']
#   smoker: Input should be a valid boolean [input_value='maybe']
```

The ValidationError lists every field that failed — not just the first one.

---

## Field — Adding Constraints and Metadata

`Field()` lets you add validation rules and documentation to individual fields:

```python
from pydantic import BaseModel, Field

class InsuranceInput(BaseModel):
    age: int = Field(
        ge=18,         # ≥ 18 (ge = "greater than or equal")
        le=100,        # ≤ 100 (le = "less than or equal")
        description="Age of the insured person",
        examples=[25, 35, 50]   # shown in Swagger UI
    )
    
    bmi: float = Field(
        gt=10.0,       # > 10 (strict, gt = "greater than")
        lt=65.0,       # < 65 (strict, lt = "less than")
        description="Body Mass Index"
    )
    
    patient_id: str = Field(
        min_length=4,
        max_length=10,
        pattern=r"^P\d{3}$",      # regex: P followed by exactly 3 digits
        description="Patient ID like P001"
    )
    
    smoker: bool = Field(
        default=False,            # optional — defaults to False if not sent
        description="Whether the person smokes"
    )
    
    children: int = Field(
        default=0,
        ge=0,
        le=10
    )
```

### Field Constraint Reference

| Constraint | Applies to | Example | Meaning |
|-----------|-----------|---------|---------|
| `ge=0` | numbers | `age: int = Field(ge=0)` | age ≥ 0 |
| `gt=0` | numbers | `weight: float = Field(gt=0)` | weight > 0 (strictly) |
| `le=100` | numbers | `score: float = Field(le=100)` | score ≤ 100 |
| `lt=100` | numbers | `pct: float = Field(lt=100)` | percentage < 100 |
| `min_length=2` | strings | `name: str = Field(min_length=2)` | at least 2 chars |
| `max_length=100` | strings | `name: str = Field(max_length=100)` | at most 100 chars |
| `pattern=r"..."` | strings | `id: str = Field(pattern=r"^P\d+$")` | must match regex |
| `default=False` | any | `smoker: bool = Field(default=False)` | use if not provided |

---

## Literal and Enum — Restricting to Specific Values

When a field should only accept predefined choices:

```python
from typing import Literal
from enum import Enum

# Option 1: Literal (inline, quick)
class InsuranceInput(BaseModel):
    sex: Literal["male", "female"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]
    smoker: Literal["yes", "no"]

# Option 2: Enum (reusable, with more features)
class Region(str, Enum):
    northeast = "northeast"
    northwest = "northwest"
    southeast = "southeast"
    southwest = "southwest"

class InsuranceInput(BaseModel):
    region: Region = Region.southeast   # default to southeast
```

Both render as **dropdowns in Swagger UI** — the user can only pick from the list. Any other value causes an automatic 422 error.

---

## Optional Fields — Handling Missing Data

```python
from typing import Optional

class PatientCreate(BaseModel):
    name: str                           # Required — must be provided
    age: int = Field(ge=0)              # Required with constraint
    city: str = "Unknown"               # Optional — defaults to "Unknown"
    phone: Optional[str] = None         # Optional — defaults to None
    smoker: bool = False                # Optional — defaults to False
    notes: Optional[str] = None         # Optional nullable string
```

The difference between `Optional[str] = None` and `str = "default"`:
- `Optional[str] = None`: field is absent or explicitly null → Python `None`
- `str = "default"`: field is absent → uses `"default"` string

---

## Computed Fields — Auto-Calculated Values

`@computed_field` creates a field that is calculated from other fields — not provided by the client:

```python
from pydantic import BaseModel, Field, computed_field

class Patient(BaseModel):
    name: str
    weight: float = Field(gt=0, description="Weight in kg")
    height: float = Field(gt=0, description="Height in cm")
    
    @computed_field
    @property
    def bmi(self) -> float:
        """
        BMI formula: weight (kg) / height (m)²
        Must convert height from cm to meters first.
        
        This field appears in model_dump() output and Swagger response schema.
        The client never sends it — the server always computes it.
        """
        height_m = self.height / 100    # cm → meters
        return round(self.weight / (height_m ** 2), 2)
    
    @computed_field
    @property
    def bmi_category(self) -> str:
        """Classify BMI into standard WHO categories."""
        if self.bmi < 18.5:   return "Underweight"
        elif self.bmi < 25.0: return "Normal"
        elif self.bmi < 30.0: return "Overweight"
        else:                  return "Obese"
```

```python
p = Patient(name="Ravi", weight=85, height=175)
print(p.bmi)            # 27.76 (auto-calculated)
print(p.bmi_category)   # "Overweight" (auto-calculated from bmi)

print(p.model_dump())
# {
#   "name": "Ravi",
#   "weight": 85.0,
#   "height": 175.0,
#   "bmi": 27.76,           ← computed field included!
#   "bmi_category": "Overweight"  ← computed field included!
# }
```

---

## Field Validators — Custom Validation Logic

When `Field(ge=..., le=...)` isn't enough, write custom validators:

```python
from pydantic import BaseModel, field_validator

class Patient(BaseModel):
    name: str
    age: int
    blood_type: str
    
    @field_validator("name")
    @classmethod
    def clean_and_validate_name(cls, v: str) -> str:
        """
        @classmethod is required for field validators.
        'v' is the raw value being validated.
        Return the (possibly transformed) valid value.
        Raise ValueError for invalid values.
        """
        v = v.strip()               # remove whitespace from both ends
        if not v:
            raise ValueError("Name cannot be whitespace only")
        if any(char.isdigit() for char in v):
            raise ValueError("Name cannot contain numbers")
        return v.title()            # "ravi kumar" → "Ravi Kumar"
    
    @field_validator("blood_type")
    @classmethod
    def validate_blood_type(cls, v: str) -> str:
        valid = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
        normalized = v.upper().strip()
        if normalized not in valid:
            raise ValueError(f"'{v}' is not a valid blood type. Must be one of: {valid}")
        return normalized
    
    @field_validator("age")
    @classmethod
    def age_must_be_realistic(cls, v: int) -> int:
        if v > 130:
            raise ValueError(f"Age {v} is unrealistically high. Maximum is 130.")
        return v
```

---

## Model Validators — Cross-Field Validation

When validity depends on multiple fields together:

```python
from pydantic import BaseModel, model_validator
from typing import Optional

class PatientAdmission(BaseModel):
    patient_age: int
    parent_name: Optional[str] = None
    admission_date: str
    discharge_date: Optional[str] = None
    
    @model_validator(mode="after")   # "after": all fields are already validated
    def apply_business_rules(self) -> "PatientAdmission":
        """
        'mode="after"' means all individual fields passed validation.
        'self' is the model instance with all fields set.
        Return self (possibly modified) or raise ValueError.
        """
        # Rule: minors must have a parent on record
        if self.patient_age < 18 and not self.parent_name:
            raise ValueError(
                "parent_name is required for patients under 18. "
                "Please provide the legal guardian's name."
            )
        
        # Rule: discharge can't be before admission
        if self.discharge_date and self.discharge_date < self.admission_date:
            raise ValueError(
                f"discharge_date ({self.discharge_date}) must be on or "
                f"after admission_date ({self.admission_date})"
            )
        
        return self
```

---

## `model_dump()` — Converting Back to Dicts

```python
patient = PatientCreate(name="Ravi", age=35, city="Mumbai", weight=78.5, height=175)

# Basic: all fields including computed ones
patient.model_dump()
# → {"name": "Ravi", "age": 35, "city": "Mumbai", "weight": 78.5,
#    "height": 175.0, "smoker": False, "bmi": 25.63}

# Only fields the client explicitly sent (critical for PATCH)
patient.model_dump(exclude_unset=True)
# → Only fields that weren't using their defaults

# Remove None values
patient.model_dump(exclude_none=True)

# Remove specific fields (security: never expose passwords)
patient.model_dump(exclude={"password_hash", "internal_notes"})

# Only specific fields
patient.model_dump(include={"name", "age", "city"})
```

---

## Pydantic for ML Feature Schemas — The Real Use Case

```python
from pydantic import BaseModel, Field
from typing import Literal

class InsuranceInput(BaseModel):
    """
    Input schema for the Insurance Premium Prediction model.
    
    This schema serves three purposes simultaneously:
    1. VALIDATION: rejects bad data before it reaches the model
    2. DOCUMENTATION: Swagger UI shows this as the expected request body
    3. TYPE SAFETY: your code knows exactly what types it's working with
    
    Field names MUST match the training data column names.
    Allowed values MUST match what the model was trained on.
    """
    age: int = Field(ge=18, le=100, description="Age of the primary insured")
    sex: Literal["male", "female"]
    bmi: float = Field(ge=10.0, le=65.0, description="Body Mass Index")
    children: int = Field(default=0, ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]
    
    # This example appears in Swagger UI — helps developers know what to send
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 35, "sex": "male", "bmi": 27.9,
                "children": 2, "smoker": "no", "region": "southeast"
            }]
        }
    }


class PredictionOutput(BaseModel):
    """
    Output schema — what the /predict endpoint returns.
    Separate from input because the output has different fields.
    """
    prediction: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0, description="Prediction confidence")
    model_version: str
```

---

## Pydantic v1 vs v2 Migration

If you read older FastAPI tutorials, you'll see v1 syntax. Here's how to translate:

| Old (Pydantic v1) | New (Pydantic v2) |
|-------------------|-------------------|
| `patient.dict()` | `patient.model_dump()` |
| `patient.json()` | `patient.model_dump_json()` |
| `Patient.schema()` | `Patient.model_json_schema()` |
| `@validator("field")` | `@field_validator("field")` |
| `@root_validator` | `@model_validator(mode="after")` |
| `class Config:` | `model_config = ConfigDict(...)` |
| `from pydantic import validator` | `from pydantic import field_validator` |

---

## Q&A

**Q: Does Pydantic validation run on every API request?**

Yes, but it's extremely fast — microseconds. Pydantic v2 is written in Rust and is one of the fastest data validation libraries in any language. The performance cost is negligible compared to ML inference time.

**Q: What's `exclude_unset=True` for?**

In PATCH operations, you only want to update fields the client explicitly sent. Without `exclude_unset=True`, `model_dump()` returns all fields including those that defaulted to `None`. The `None` values would overwrite existing database values — deleting data accidentally. `exclude_unset=True` returns only what the client actually sent.

**Q: Should my request schema and response schema be the same class?**

Usually not. The request schema (what the client sends) typically excludes server-generated fields (ID, timestamps, computed values). The response schema (what you return) includes those. Define separate classes like `PatientCreate` (input) and `PatientResponse` (output).

**Q: Can Pydantic validate things that aren't coming from an HTTP request?**

Yes! Pydantic is a standalone library. Use it anywhere: config file validation, database query result parsing, ML pipeline input/output contracts, CLI argument validation. FastAPI just happens to use it deeply.

**Q: How do I add a field with no constraints but still want it in the schema?**

Just use a plain type annotation:
```python
class Patient(BaseModel):
    name: str        # no Field() needed — just annotate the type
    notes: str = ""  # with a default
```
`Field()` is only needed when you want constraints, descriptions, or examples.
