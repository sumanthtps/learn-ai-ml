---
title: "Docker for Machine Learning"
sidebar_position: 80
---

## Source

- Playlist: FastAPI for Machine Learning | CampusX
- Original video title: 10. Docker for Machine Learning | Docker Crash Course | CampusX
- YouTube video ID: `GToyQTGDOS4`
- Transcript pages in the uploaded PDF: 207-250

## Why this lesson matters

This lesson broadens the playlist from API coding into environment reproducibility. That shift is essential because a model API that only works on one laptop is not a deployable system.

Docker solves a very practical engineering problem: packaging application code and its runtime dependencies into a consistent unit that can run across environments with far less drift.

## What the transcript covers

The transcript explains:

- why Docker is relevant across software roles
- key Docker concepts such as engine, image, container, Dockerfile, and registry
- why Docker matters for ML and MLOps
- the overall workflow of building an image, pushing it, and running it elsewhere

## The core problem Docker solves

Without Docker, applications often fail when moved between environments.

Typical causes:

- missing packages
- different Python versions
- OS-level library differences
- environment configuration drift

This produces the classic problem:

"It works on my machine, but not on yours."

Docker reduces that gap by packaging the app environment in a standard way.

## Image vs container

This is the most fundamental distinction.

### Image
An image is a packaged blueprint.

It contains:

- application code
- dependencies
- runtime instructions
- sometimes model artifacts and config defaults

### Container
A container is a running instance of an image.

Think of it as:

- image = recipe or blueprint
- container = the actual running dish or built house

You can create many containers from one image.

## Dockerfile: how the image is defined

A Dockerfile is the list of instructions used to build the image.

Typical responsibilities:

- choose a base image
- install dependencies
- copy code
- set working directory
- define startup command

### Example

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This says:

- start from a Python base image
- work inside `/app`
- install dependencies
- copy project files
- run the FastAPI server

## Registry: why it matters

Once you build an image locally, you often want other systems to use it.

That is where registries come in.

A registry stores Docker images.

Examples:

- Docker Hub
- cloud-specific registries
- private company registries

### Workflow

1. build image locally
2. tag it
3. push to registry
4. another machine pulls and runs it

This is what makes image distribution practical.

## Why ML teams care about Docker

Machine learning systems often depend on:

- Python packages
- model artifacts
- preprocessing code
- OS-level dependencies
- specific runtime versions

These dependencies can be fragile across machines.

Docker helps package them into a consistent execution unit.

### ML-specific benefits

- inference service runs consistently across environments
- onboarding becomes easier
- deployment pipeline becomes more predictable
- integration testing becomes easier

## Docker does not replace understanding the app

A common beginner misunderstanding is:

"If I put the project into Docker, the engineering problems are solved."

Not true.

Docker helps with packaging and environment consistency. It does not automatically solve:

- poor API design
- broken model logic
- weak validation
- inefficient inference
- security problems
- scaling bottlenecks

It is an enabler, not magic.

## Layered build concept

Docker images are built in layers.

This is useful because repeated builds can reuse unchanged layers.

### Practical consequence
If `requirements.txt` has not changed, Docker may reuse the dependency installation layer instead of reinstalling everything.

That is why Dockerfiles are often ordered carefully.

## Common Docker terms you should know

### Docker Engine
The software that builds and runs containers.

### Image
The packaged blueprint.

### Container
A running instance of the image.

### Dockerfile
The recipe used to build the image.

### Registry
The place where images are stored and shared.

### Tag
A label used to identify image versions.

Example:

```text
my-fastapi-app:v1
```

## Basic workflow for an ML API

1. Build your FastAPI app.
2. Ensure the model artifact is available.
3. Write a Dockerfile.
4. Build the image.
5. Run the container locally.
6. Push the image to a registry.
7. Pull and run it on another machine or cloud host.

This is the bridge between local development and deployment.

## Useful commands

### Build image

```bash
docker build -t my-fastapi-app:v1 .
```

### Run container

```bash
docker run -p 8000:8000 my-fastapi-app:v1
```

### List running containers

```bash
docker ps
```

### Push image

```bash
docker push my-fastapi-app:v1
```

The exact registry naming may vary, but these are the core commands.

## Why port mapping matters

Inside the container, your app may listen on port 8000.

The host machine also needs a way to access it.

That is what `-p 8000:8000` does.

General form:

```text
host_port:container_port
```

Example:

```bash
docker run -p 8080:8000 my-fastapi-app:v1
```

This means:
- host port 8080 forwards to container port 8000

## Containers vs virtual machines

This distinction is often discussed in Docker learning.

### Virtual machine
Includes a full guest OS on top of virtualization.

### Container
Shares the host OS kernel but isolates processes and dependencies.

Containers are typically lighter and faster to start.

That is one reason they are popular for application packaging.

## Common mistakes beginners make

### 1. Thinking image and container are the same thing
They are related but different.

### 2. Forgetting to include required files in the image
If the model file is missing, the app may fail inside the container.

### 3. Using the wrong startup command
The image may build, but the container will not run properly.

### 4. Not understanding port mapping
The service may be healthy inside the container but inaccessible from outside.

### 5. Treating Docker as a replacement for dependency discipline
The underlying project still needs clear requirements and structure.

## Daily engineering additions beyond the transcript

### 1. Use `.dockerignore`
Exclude notebooks, caches, virtual environments, and unnecessary files.

### 2. Keep images smaller where possible
Large images slow build, push, and deployment workflows.

### 3. Use explicit version tags
Avoid relying only on vague labels like `latest`.

### 4. Think about secrets separately
Do not bake sensitive credentials into images.

### 5. Add health checks and logs
A running container is not the same as a healthy application.

## Important Q&A

### 1. What is the difference between a Docker image and a Docker container?
An image is the packaged blueprint; a container is the running instance created from it.

### 2. Why is Docker especially useful in ML projects?
Because ML services often have many dependencies and artifacts that must behave consistently across environments.

### 3. What does a Dockerfile do?
It defines the instructions for building the image.

### 4. Why do registries matter?
They allow images to be stored, shared, and deployed across machines.

### 5. Does Docker solve every deployment problem automatically?
No. It helps with packaging and consistency, but application design and operations still matter.

## Quick revision

- Docker helps make environments reproducible.
- Image is the blueprint; container is the running instance.
- Dockerfile defines how the image is built.
- Registry stores and distributes images.
- Docker is a major bridge from local app to deployable service.
