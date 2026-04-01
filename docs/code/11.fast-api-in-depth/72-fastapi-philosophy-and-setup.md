---
title: "FastAPI Philosophy, Strengths, and Setup"
sidebar_position: 72
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 2. FastAPI Philosophy | How to setup FastAPI | Installation and Code Demo | Video 2 | CampusX
- YouTube video ID: `lXx-_1r0Uss`
- Transcript pages in the uploaded PDF: 28-50

## Why this lesson matters

This lesson explains why FastAPI became so popular, especially in Python-heavy teams such as data science, backend engineering, MLOps, and AI product teams. It also introduces the practical setup required to run the first application.

If the previous lesson answered "why APIs?", this lesson answers "why FastAPI for Python APIs?"

## What the transcript covers

The transcript explains:

- what FastAPI is
- why it is called modern and high-performance
- the role of Starlette and Pydantic inside FastAPI
- why type hints matter so much
- why automatic docs are useful
- how to install FastAPI and Uvicorn
- how to run the first application locally

## FastAPI in one line

FastAPI is a modern Python web framework for building APIs with strong validation, high developer productivity, and strong performance.

That line contains three important ideas:

- **modern** - it uses Python type hints heavily
- **validation-focused** - it turns schemas into real runtime checks
- **fast to build and fast to run** - it reduces boilerplate while remaining efficient

## What makes FastAPI feel modern

Earlier Python API frameworks often made you manually do many things:

- extract values from requests
- convert types yourself
- validate JSON fields manually
- document endpoints separately

FastAPI reduces that friction.

When you define function parameters and request schemas properly, FastAPI can infer a lot:

- expected query/path values
- request body schema
- validation rules
- docs structure

That is why it feels expressive rather than verbose.

## The two major building blocks mentioned in the transcript

The transcript points out that FastAPI is built using two important libraries.

### Starlette
Starlette handles much of the web and async foundation:

- receiving HTTP requests
- routing requests to the correct endpoint
- building responses
- ASGI-based web behavior

### Pydantic
Pydantic handles data validation and parsing:

- checking field types
- validating structure
- converting JSON-like data into Python objects
- generating schemas

### Why this combination is powerful
The API layer needs two things all the time:

- web request handling
- data validation

FastAPI gives both in one coherent developer experience.

## Why Python type hints matter here

In normal Python code, many developers treat type hints as optional documentation.

In FastAPI, type hints become much more powerful. They help drive:

- validation
- request parsing
- automatic docs generation
- editor support and readability

### Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/square')
def square(number: int) -> dict:
    return {'square': number * number}
```

What happens here?

- `number: int` tells FastAPI to expect an integer
- if a string that cannot be converted to int is passed, validation fails
- the docs can show the expected parameter type
- the function signature itself becomes part of the API contract

This is one reason FastAPI feels natural to read.

## High performance: what does that really mean?

The word "fast" can be misunderstood.

FastAPI does not make your machine learning model magically faster. Instead, it provides an efficient API framework with modern request handling behavior.

Performance depends on multiple layers:

- web framework overhead
- JSON serialization
- business logic complexity
- database latency
- model inference latency

FastAPI helps reduce framework-side friction, but if your model takes 5 seconds to run, the total response will still be slow.

### Correct understanding
FastAPI improves the serving layer. It does not replace good model optimization, caching, or system design.

## Why FastAPI is attractive to ML engineers

ML engineers often work in Python already:

- training code in Python
- preprocessing code in Python
- model serialization in Python
- experimentation notebooks in Python

So a Python-native serving layer is convenient.

### Practical benefits

- no language switch is needed just to expose a model
- Pydantic helps validate inference inputs
- auto docs help frontend or QA teams test endpoints quickly
- the framework is friendly for prototypes and scalable enough for real services

This is why FastAPI became especially common in ML-serving tutorials and internal tooling.

## Setup: what do you actually install?

A minimal setup usually needs:

- `fastapi`
- `uvicorn`

### Why Uvicorn?
FastAPI defines the application, but something still needs to run that application as a server. Uvicorn is a popular ASGI server used for that purpose.

Think of it like this:

- FastAPI = the application logic and route definitions
- Uvicorn = the server process that listens for requests and executes the app

## Your first FastAPI app

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home() -> dict:
    return {'message': 'Hello from FastAPI'}
```

### Line-by-line explanation

#### `from fastapi import FastAPI`
Imports the main class used to create the application.

#### `app = FastAPI()`
Creates the application object.

#### `@app.get('/')`
Registers a route. When a GET request comes to `/`, this function should run.

#### `def home() -> dict:`
Defines the endpoint logic. The return type here is a Python dictionary.

#### `return {'message': 'Hello from FastAPI'}`
FastAPI serializes the dictionary into JSON automatically.

## How to run it

Typical command:

```bash
uvicorn main:app --reload
```

### Meaning

- `main` = file name without `.py`
- `app` = FastAPI application object inside that file
- `--reload` = restart automatically when code changes during development

If your file is `app.py` and your variable is `api`, then you would run:

```bash
uvicorn app:api --reload
```

## Auto-generated docs: why they matter

FastAPI automatically creates interactive docs when routes and schemas are defined properly.

This is a major productivity boost.

### Why developers love this

- frontend teams can inspect endpoints quickly
- QA teams can test without writing code immediately
- backend engineers can verify request formats quickly
- ML engineers can manually test prediction payloads

### Why this matters in real teams
Documentation is often neglected when written manually. FastAPI reduces that problem by generating useful docs from the code itself.

## Worked example with validation

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/greet')
def greet(name: str, age: int) -> dict:
    return {
        'message': f'Hello {name}',
        'age_next_year': age + 1
    }
```

Request:

```text
GET /greet?name=Aman&age=25
```

Response:

```json
{
  "message": "Hello Aman",
  "age_next_year": 26
}
```

If age cannot be converted to an integer, FastAPI returns a validation error instead of silently accepting bad input.

## Good beginner project structure

Even for small projects, use a clean layout.

```text
project/
  main.py
  requirements.txt
  models/
  routers/
  services/
  data/
```

The transcript starts with a basic setup, which is good for learning. In real projects, code tends to grow quickly, so folder organization becomes important.

## FastAPI vs Flask: the correct comparison mindset

The transcript contrasts FastAPI with older frameworks such as Flask. This should not be read as "one is useless and the other is perfect."

A better comparison is:

### Flask
- simple and flexible
- large ecosystem
- more manual work for validation and docs

### FastAPI
- more batteries included for APIs
- stronger use of type hints
- automatic validation and docs
- especially pleasant for schema-heavy services

The right takeaway is that FastAPI is often a better fit when your primary goal is building APIs quickly and cleanly.

## Common beginner mistakes

### 1. Confusing FastAPI with Uvicorn
FastAPI is the app framework. Uvicorn is the server process that runs the app.

### 2. Not understanding `main:app`
This points to the Python module and variable name, not a random string.

### 3. Assuming `--reload` is for production
It is mainly a development convenience.

### 4. Ignoring type hints
In FastAPI, type hints are not cosmetic. They help define behavior.

### 5. Thinking auto docs remove the need for careful API design
Docs are generated from what you define. If your endpoint design is messy, the docs will also reflect that mess.

## Daily engineering additions beyond the transcript

To use FastAPI well in day-to-day work, also understand these ideas:

### 1. Separation of concerns
Avoid putting all business logic directly inside route functions.

### 2. Configuration management
Environment variables and config files matter once you move beyond local development.

### 3. Dependency management
Keep `requirements.txt` or equivalent tooling updated.

### 4. Error handling
A mature API needs predictable error responses.

### 5. Testing
Even a tiny API should have basic tests for routes and validation.

## Important Q&A

### 1. Why is FastAPI called modern?
Because it uses Python type hints, schema-driven validation, and auto-generated documentation in a very integrated way.

### 2. Why do data science teams like FastAPI?
Because they already work in Python and can expose models without switching languages or writing excessive boilerplate.

### 3. What is the role of Uvicorn?
It is the server that runs the FastAPI application and listens for incoming requests.

### 4. Why are automatic docs so useful?
They reduce friction for testing, collaboration, and understanding the API contract.

### 5. Is FastAPI automatically production-ready by default?
No. It gives a strong foundation, but production quality still requires logging, security, configuration, testing, and deployment discipline.

## Quick revision

- FastAPI is a Python API framework.
- It builds on Starlette and Pydantic.
- Type hints drive validation and docs.
- Uvicorn runs the app.
- Auto docs are a big productivity advantage.
- FastAPI is especially convenient for Python-heavy ML workflows.
