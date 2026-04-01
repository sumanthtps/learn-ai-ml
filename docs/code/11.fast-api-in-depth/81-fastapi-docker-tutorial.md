---
title: "Dockerizing the FastAPI ML API"
sidebar_position: 81
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 11. FastAPI + Docker Tutorial for Beginners | How to Dockerize a FastAPI API Application | CampusX
- YouTube video ID: `jlLs6hfAga4`
- Transcript pages in the uploaded PDF: 251-261

## Why this lesson matters

The previous Docker lesson explained the concepts. This lesson applies those concepts directly to the FastAPI machine learning API built in the playlist.

This is where packaging becomes concrete: code, dependencies, model artifacts, and startup behavior all need to be assembled into one runnable image.

## What the transcript covers

The transcript explains:

- recapping the prediction API project
- why the improved API is ready for containerization
- turning the FastAPI app into a Docker image
- including required files so the app can run elsewhere
- running the container and sharing the image via Docker Hub

## What must go into the image?

To run the API successfully on another machine, the image usually needs:

- application code
- dependency definitions
- model artifact
- preprocessing artifact if separate
- startup command

If any critical piece is missing, the container may start but fail at runtime.

## A practical Dockerfile for FastAPI

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Line-by-line explanation

### `FROM python:3.11-slim`
Choose a lightweight base image with Python installed.

### `WORKDIR /app`
Set the working directory inside the image.

### `COPY requirements.txt .`
Copy dependency file first.

### `RUN pip install --no-cache-dir -r requirements.txt`
Install Python packages.

### `COPY . .`
Copy application code and supporting files.

### `CMD [...]`
Define the default startup command.

## Why copy `requirements.txt` separately?

This is a very useful Docker optimization.

Because Docker builds in layers, keeping dependency installation separate allows caching.

If you change only app code but not dependencies, Docker may reuse the package-installation layer and build faster.

That is why this order is common.

## Why `0.0.0.0` is used inside containers

If the app binds only to `127.0.0.1` inside the container, it may be reachable only from within the container itself.

Binding to `0.0.0.0` allows the app to listen on all network interfaces inside the container, which is necessary for external access through port mapping.

This is a very common containerization detail.

## Model artifact inclusion

For ML APIs, copying only the source code is often not enough.

You may also need:

- trained model file
- preprocessing pipeline file
- label encoder or category mapping file

### Example structure

```text
project/
  main.py
  requirements.txt
  model.joblib
  preprocessor.joblib
  Dockerfile
```

If these files are missing from the image, prediction logic may fail.

## Build and run commands

### Build

```bash
docker build -t insurance-api:v1 .
```

### Run

```bash
docker run -p 8000:8000 insurance-api:v1
```

Then you can access the app at the mapped host port.

## Why Docker Hub is useful in this workflow

The transcript mentions pushing the image so other people or systems can use it.

That is important because local images remain only on your machine.

Pushing to a registry enables:

- QA to run the same app version
- deployment targets to pull the same image
- teammates to test consistent behavior

This is how containerization becomes collaborative instead of local-only.

## Common real-world additions

### `.dockerignore`
Do not copy everything blindly.

Typical exclusions:

- `.git`
- `__pycache__`
- notebooks
- local virtual environments
- large raw datasets not needed for inference

### Environment variables
Some values should be configurable rather than hardcoded.

Examples:

- port
- model path
- log level
- environment name

### Separate directories
Keep app code and artifacts organized so the Docker build is predictable.

## Worked example with a more realistic structure

```text
project/
  app/
    main.py
    schemas.py
    service.py
  artifacts/
    model.joblib
    preprocessor.joblib
  requirements.txt
  Dockerfile
  .dockerignore
```

This layout is easier to maintain than putting everything in one flat folder.

## Why containerization does not equal production readiness

Containerization solves packaging and portability.

It does not automatically provide:

- health monitoring
- scaling
- authentication
- secure secret handling
- production-grade observability

These still need engineering attention.

## Common mistakes beginners make

### 1. Building the image without testing local app correctness first
Dockerizing a broken app only hides the root problem.

### 2. Forgetting the model file
The app starts, but prediction fails.

### 3. Binding the app incorrectly
If the server does not listen on `0.0.0.0`, access from outside the container may fail.

### 4. Copying unnecessary files into the image
This makes images larger and slower to build.

### 5. Not pinning dependencies sensibly
Rebuilds may behave differently over time if package versions drift.

## Daily engineering additions beyond the transcript

### 1. Add a health route and test it inside the container
Do not assume successful start equals healthy service.

### 2. Use explicit image tags
Versioned tags are better than relying only on `latest`.

### 3. Keep build context small
It improves speed and reduces accidental leakage.

### 4. Test container behavior from a clean environment
That is the real portability check.

### 5. Treat the image as a release artifact
Once built and tested, it becomes a deployable unit.

## Important Q&A

### 1. Why copy `requirements.txt` separately in a Dockerfile?
Because it helps Docker cache dependency installation and speeds up rebuilds when only code changes.

### 2. Why do we include model artifacts in the image?
Because the inference service depends on them at runtime.

### 3. Why is `0.0.0.0` commonly used in containerized apps?
Because the app must listen on all interfaces inside the container to be reachable from outside.

### 4. Why is pushing to Docker Hub useful?
Because it lets other systems and people pull and run the exact same packaged application.

### 5. Does a containerized app automatically become production-ready?
No. Packaging is one step; operations and reliability concerns still remain.

## Quick revision

- Dockerizing means packaging app code, dependencies, and artifacts.
- The Dockerfile should be ordered thoughtfully.
- Model files must be included if inference depends on them.
- `0.0.0.0` matters inside containers.
- Registry push makes the image shareable and deployable.
