---
id: docker
title: "10 · Docker — Containers for ML Engineers"
sidebar_label: "10 · Docker"
sidebar_position: 10
tags: [docker, containers, dockerfile, images, devops, mlops, beginner]
---

# Docker — Containers for ML Engineers

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=GToyQTGDOS4) · **Series:** FastAPI for ML – CampusX

---

## The "Works on My Machine" Problem

You build a FastAPI prediction API. It runs beautifully on your MacBook with Python 3.11, scikit-learn 1.5.1, and pandas 2.2.2.

You deploy it to the company's Ubuntu server. It crashes immediately.

Why? The server has Python 3.8, a different numpy version, and pandas isn't even installed.

This is the most common deployment failure in software engineering. It's so famous it has a name: the **"works on my machine" problem**.

Docker solves this completely and permanently.

---

## What Docker Does

Docker packages your application along with **everything it needs to run** — Python version, all libraries, configuration — into a single, portable unit called a **container**. This container runs identically on:

- Your MacBook
- Your colleague's Windows laptop
- An Ubuntu EC2 instance in AWS
- A Kubernetes cluster in GCP

The container brings its own environment. The host machine's Python version, installed libraries, or OS don't matter.

---

## The Physical Shipping Container Analogy

Before shipping containers existed, loading a cargo ship was chaos. Every ship had different storage configurations. Different ports used different cranes. Loading took days. Goods frequently got damaged or lost.

The shipping container standardized everything:

```
Before standard containers:          After standard containers:
────────────────────────────         ──────────────────────────
Cargo: boxes, barrels, crates        Cargo: standardized metal boxes
Loading: custom per ship             Loading: same crane everywhere
Transit: unpredictable               Transit: predictable, reliable
Time: days to load/unload            Time: hours to load/unload
```

Docker containers do the same for software:

```
Before Docker:                        After Docker:
──────────────────────────            ──────────────────────────
App depends on host environment       App + environment in one unit
"Works on my machine"                 Works everywhere identically
Complex deployment scripts            docker run ...
Weeks to onboard new team member      Minutes to run the app
```

---

## Three Core Concepts

### 1. Dockerfile — The Recipe

A text file with step-by-step instructions for building your application image:

```
"Start from Python 3.11 on Ubuntu"
"Copy requirements.txt"
"Install all Python packages"
"Copy my application code"
"When container starts, run uvicorn"
```

### 2. Image — The Built Artifact

Running `docker build` executes your Dockerfile and produces an **image** — a frozen, immutable snapshot of your app and all its dependencies.

Share this image: anyone with Docker can run it identically.

```
Dockerfile ──► docker build ──► Image
(recipe)                        (packaged app — immutable)
```

### 3. Container — The Running Instance

Running `docker run` creates a **container** from an image — a live, running process with isolated filesystem and network.

```
Image ──► docker run ──► Container (running)
           │
           └─► docker run ──► Container 2 (same image, different instance)
           │
           └─► docker run ──► Container 3 (you can run many from one image)
```

---

## Why Docker = Universal (Every Role Needs It)

Docker is truly universal in software:

| Role | Why Docker Matters |
|------|--------------------|
| Data Scientist | Package model + code for reproducible experiments |
| ML Engineer | Deploy models consistently across environments |
| Backend Developer | Run your API anywhere without setup |
| DevOps | Kubernetes, ECS — all built on containers |
| QA Engineer | Isolated test environments in seconds |

---

## Essential Docker Commands

### Building Images

```bash
# Build an image from Dockerfile in the current directory
# -t = tag (name:version)
docker build -t insurance-api:1.0 .

# Build with a specific Dockerfile name
docker build -t insurance-api:1.0 -f Dockerfile.prod .

# See all images on your machine
docker images
# REPOSITORY      TAG    IMAGE ID     CREATED        SIZE
# insurance-api   1.0    a3f8b2c1de   2 minutes ago  487MB
# python          3.11   9d1bfc5a3c   2 weeks ago    1.01GB
```

### Running Containers

```bash
# Basic run (foreground — you see all logs, Ctrl+C stops it)
docker run insurance-api:1.0

# Run in background (detached)
docker run -d insurance-api:1.0

# Full production run with all options
docker run \
  -d \                                  # detached (background)
  --name insurance-api \                # give it a memorable name
  -p 8000:8000 \                        # host_port:container_port
  -e API_KEY=my-secret-key \            # environment variable
  -e LOG_LEVEL=INFO \                   # another env var
  --restart unless-stopped \            # restart if it crashes
  insurance-api:1.0
```

### Monitoring and Debugging

```bash
# List running containers
docker ps
# CONTAINER ID  IMAGE              STATUS         PORTS
# a3f8b2c1     insurance-api:1.0  Up 5 minutes   0.0.0.0:8000->8000/tcp

# All containers (including stopped ones)
docker ps -a

# See live logs
docker logs insurance-api

# Follow logs in real time (like tail -f)
docker logs -f insurance-api

# Open a shell INSIDE the running container
# Useful for debugging — see the filesystem, run Python, check logs
docker exec -it insurance-api /bin/bash

# Stop a container gracefully (waits for cleanup)
docker stop insurance-api

# Force stop immediately (last resort)
docker kill insurance-api

# Remove a stopped container
docker rm insurance-api
```

---

## Port Mapping — The `-p` Flag Explained

Containers are completely isolated by default. No external traffic can reach anything inside. Port mapping creates a deliberate opening:

```
Your laptop (host machine)         Docker Container
──────────────────────────         ────────────────────────────
Port 8000: OPEN   ←─────────────── Port 8000: Uvicorn listening
Port 3306: CLOSED
Port 5432: CLOSED

-p 8000:8000 means:
"Forward any traffic arriving at host port 8000
 to container port 8000"

-p 9000:8000 means:
"Forward host port 9000 to container port 8000"
(useful when host port 8000 is already in use)
```

---

## Environment Variables in Docker

Never bake secrets into your Docker image. Pass them at runtime:

```bash
# Individual variables
docker run -d -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/mldb \
  -e API_KEY=super-secret-key-here \
  -e MODEL_PATH=/app/artifacts/model_v2.pkl \
  insurance-api:1.0

# From a .env file (most convenient for many variables)
docker run -d -p 8000:8000 --env-file .env insurance-api:1.0
# (The .env file stays on your host — never inside the image)
```

---

## Docker Layers — Why Order Matters in Dockerfile

Every instruction in a Dockerfile creates a **layer**. Layers are cached. If a layer's content hasn't changed, Docker reuses the cached version — skipping that instruction in the next build.

```dockerfile
# ❌ SLOW — every code change re-runs pip install (2-5 minutes!)
COPY . .                           # Layer: copy everything
RUN pip install -r requirements.txt  # Layer: install dependencies

# ✅ FAST — pip install only re-runs when requirements change
COPY requirements.txt .            # Layer: copy just requirements
RUN pip install -r requirements.txt  # Layer: install (cached if req unchanged)
COPY . .                           # Layer: copy your code (changes often)
```

With the correct order: changing a Python file means only the `COPY . .` layer is re-run. The `pip install` layer is cached → rebuilds take seconds instead of minutes.

**Rule:** Put things that change rarely (base image, OS packages, Python packages) early. Put things that change often (your code) last.

---

## Docker Networking — Container Communication

Containers don't automatically see each other. Use Docker networks:

```bash
# Create a network for your ML stack
docker network create ml-stack

# Run your API on that network
docker run -d --name api --network ml-stack insurance-api:1.0

# Run a Redis cache on the same network
docker run -d --name redis --network ml-stack redis:7-alpine

# Now: from inside the 'api' container, "redis" resolves to the Redis container's IP
# Code: redis_client = redis.from_url("redis://redis:6379")
#                                              ^^^^^
#                                              container name works as hostname!
```

---

## Docker Cleanup

Unused images and stopped containers accumulate fast:

```bash
# Remove all stopped containers + unused images + unused networks
docker system prune

# With confirmation (just yes it)
docker system prune -f

# Nuclear option: remove EVERYTHING (images, containers, volumes, networks)
docker system prune -a --volumes -f
```

---

## Q&A

**Q: What's the difference between a Docker image and a container?**

An image is static and immutable — like a class definition or a template. You build it once, it never changes. A container is a live, running instance of an image — like an object instantiated from the class. You can run many containers from one image simultaneously.

**Q: Is Docker the same as a virtual machine?**

Similar concept, very different implementation. A Virtual Machine (VM) runs a complete OS with its own kernel. Containers share the host OS kernel but have isolated processes and filesystems. Result: containers start in seconds (VMs take minutes), use far less RAM (containers: tens of MB overhead, VMs: hundreds of MB overhead), and run with near-native performance.

**Q: Why does my container work locally but crash when deployed?**

Most common causes:
1. **Missing `--host 0.0.0.0`** — Uvicorn only listens on localhost inside the container
2. **Environment variables not set** — secrets from your `.env` file aren't in the deployed container
3. **Architecture mismatch** — built on Mac M1 (arm64) but deploying to x86 server. Fix: `docker build --platform linux/amd64 -t my-api:1.0 .`
4. **Image not pushed** — you built locally but the server pulled the old version from the registry

**Q: What is Docker Hub?**

A public registry where Docker images are hosted. When you run `FROM python:3.11-slim` in your Dockerfile, Docker pulls the `python` image tagged `3.11-slim` from Docker Hub. You can also push your own images there to share with your team or deploy to servers.

**Q: How do I reduce image size?**

Use `python:3.11-slim` instead of `python:3.11` (~900MB → ~120MB base). Add a `.dockerignore` file. Use `--no-cache-dir` in pip install. Use multi-stage builds for production.

---

## Key Commands Cheat Sheet

```bash
# Build
docker build -t myapp:1.0 .

# Run
docker run -d --name myapp -p 8000:8000 myapp:1.0

# Debug
docker logs -f myapp              # follow logs
docker exec -it myapp /bin/bash   # open shell inside

# Manage
docker ps                          # list running
docker stop myapp                  # stop
docker rm myapp                    # remove container

# Registry
docker tag myapp:1.0 user/myapp:1.0
docker push user/myapp:1.0
docker pull user/myapp:1.0

# Cleanup
docker system prune -f             # remove stopped containers + dangling images
```
