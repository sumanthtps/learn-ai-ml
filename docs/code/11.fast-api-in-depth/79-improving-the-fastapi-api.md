---
title: "Improving the FastAPI ML API"
sidebar_position: 79
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 9. Improving the FastAPI API | Video 8 | CampusX
- YouTube video ID: `M17qwKnmG38`
- Transcript pages in the uploaded PDF: 186-206

## Why this lesson matters

A first working API is a milestone, but not the finish line. Many beginner projects stop at "it runs on my machine." Real engineering starts when you ask whether the service is maintainable, reproducible, understandable, and deployable.

This lesson is about that transition.

## What the transcript covers

The transcript explains:

- why the first prediction API version is too basic for real use
- cleaning up the project before deployment steps
- handling dependencies through `requirements.txt`
- organizing files more carefully
- making the ML API more deployment-friendly

## From demo to service: what changes?

A demo focuses on proving that the idea works.

A service must additionally care about:

- code structure
- reproducibility
- clarity of inputs and outputs
- dependency management
- deployment readiness
- maintainability by other engineers

That is the difference between academic success and operational usefulness.

## Why `requirements.txt` matters

A Python project depends on packages.

If those packages are not captured explicitly, you create reproducibility risk.

### Problem
Your API works locally because your laptop happens to have the right packages installed.

Another machine may fail because:

- package is missing
- wrong version is installed
- dependency conflicts exist

### Solution
Record the required packages.

Example:

```text
fastapi
uvicorn
pydantic
scikit-learn
pandas
joblib
```

This becomes even more important before Docker and deployment.

## File and folder organization

As the transcript suggests, code and artifacts should not remain in a chaotic experimental state.

Typical project pieces include:

- application entry point
- model artifact
- preprocessing pipeline
- schema definitions
- utilities
- dependency file

### Example structure

```text
project/
  app/
    main.py
    schemas.py
    services.py
  artifacts/
    model.joblib
    preprocessor.joblib
  requirements.txt
  README.md
```

Good structure reduces confusion during containerization and deployment.

## Why file paths matter so much in ML APIs

This is one of the most common real-world pain points.

Your code may load a model like this:

```python
model = joblib.load('model.pkl')
```

That works only if the current working directory is what you expect.

When the project moves into a container or a different environment, relative paths often break.

### Better mindset
Treat model artifact loading as an explicit part of application setup.

Use stable paths and predictable layout.

## Better response design

A beginner API often returns whatever was easy to return.

A stronger API returns a clear, client-friendly structure.

### Weak response

```json
{
  "prediction": 2
}
```

### Better response

```json
{
  "predicted_premium": 18432.55,
  "risk_band": "high",
  "model_version": "v1"
}
```

This makes integration easier.

## Better error behavior

A demo API often fails messily.

A production-facing API should fail predictably.

Examples:

- invalid request body -> validation error
- missing model artifact -> controlled startup failure or internal error response
- unsupported category value -> clear explanation

The user and client team should not have to infer what went wrong from a stack trace.

## Why cleanup before Docker is smart

Dockerizing messy code preserves mess inside a container.

The transcript's sequence is good:

1. build working API
2. improve structure and dependencies
3. containerize
4. deploy

This prevents the common mistake of using Docker as a bandage over poor organization.

## Example improvement mindset

### Stage 1: quick prototype

```python
@app.post('/predict')
def predict(data: dict) -> dict:
    # ad hoc logic here
    return {'prediction': 2}
```

### Stage 2: improved version

```python
from pydantic import BaseModel

class PredictionInput(BaseModel):
    age: int
    bmi: float
    smoker: bool
    city: str

@app.post('/predict')
def predict(data: PredictionInput) -> dict:
    payload = data.model_dump()
    prediction = 'high'
    return {
        'input': payload,
        'prediction': prediction,
        'model_version': 'v1'
    }
```

That is a more explicit contract and a better base for future improvements.

## What a production-minded ML API usually adds

Even if the transcript does not go deeply into every item, these are the next things engineers typically add:

### 1. Health endpoint
Example:

```python
@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}
```

### 2. Logging
Record request timing, failures, and model version information.

### 3. Configuration handling
Use environment variables or config files instead of hardcoding everything.

### 4. Better exception handling
Convert internal failures into predictable API responses.

### 5. Tests
At least verify the most important routes and validation behavior.

## Why readability matters for future you

In beginner projects, many people optimize only for immediate success.

In real engineering, the person debugging this service next month may also be you.

Readable code reduces operational cost.

That includes:

- clear schema names
- explicit function names
- organized files
- comments only where they add value
- not mixing unrelated logic in one giant route function

## Common mistakes beginners make

### 1. Treating a working local demo as finished
Deployment, reproducibility, and maintainability still remain.

### 2. Forgetting dependency capture
This causes "works on my machine" problems quickly.

### 3. Hardcoding fragile paths
Containers and cloud environments expose these weaknesses fast.

### 4. Returning cryptic outputs
Clients should not need insider knowledge to interpret responses.

### 5. Waiting too long to clean up project structure
The longer disorder remains, the more painful Docker and deployment become.

## Daily engineering additions beyond the transcript

### 1. Add a `.dockerignore`
This avoids copying unnecessary files into the image.

### 2. Separate business logic from route handlers
Routes should stay thin.

### 3. Use response models
They make the output contract more explicit.

### 4. Version critical artifacts
Model version and schema version are both useful.

### 5. Think about secrets early
Do not hardcode credentials or private URLs in the codebase.

## Important Q&A

### 1. Why is the first working API version usually not enough?
Because working locally does not guarantee maintainability, reproducibility, or deployment readiness.

### 2. Why create `requirements.txt` before Dockerizing?
Because the container build needs an explicit dependency list.

### 3. Why can moving files break an ML API?
Because model loading paths and resource assumptions often depend on project structure.

### 4. What is a common serving mistake with model files?
Assuming a relative file path will behave the same everywhere.

### 5. Why is cleanup part of engineering, not just polishing?
Because structure, clarity, and reproducibility directly affect deployment and team productivity.

## Quick revision

- A demo API is not automatically production-friendly.
- Dependency capture matters.
- File structure and path handling matter.
- Better responses and errors improve integration.
- Clean up before containerization and deployment.
