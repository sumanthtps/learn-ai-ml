---
id: dockerize-fastapi
title: "11 · Dockerizing a FastAPI Application"
sidebar_label: "11 · Dockerize FastAPI"
sidebar_position: 11
tags: [docker, dockerfile, docker-compose, containerization, fastapi, mlops, beginner]
---

# Dockerizing a FastAPI Application

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=jlLs6hfAga4) · **Series:** FastAPI for ML – CampusX

---

## What "Dockerizing" Means in Practice

In Video 10, you learned what Docker is. Now you apply it: write a `Dockerfile` for your FastAPI app, build it into an image, test it locally, and push it to a registry so it can be deployed anywhere.

After this video, your entire application — Python 3.11, all pip packages, your `model.pkl`, your FastAPI code, and Uvicorn — lives in one portable image that runs with one command.

---

## Step 1: Lock Your Dependencies

Before writing the Dockerfile, create a precise `requirements.txt`. Pinning versions prevents "it worked last week" surprises caused by a package releasing a breaking change.

```txt title="requirements.txt"
# Pin ALL versions — no version ranges in production!
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.8.0
pydantic-settings==2.4.0
scikit-learn==1.5.1
joblib==1.4.2
pandas==2.2.2
numpy==1.26.4
```

Generate from your current working environment:
```bash
pip freeze > requirements.txt
# WARNING: pip freeze includes ALL transitive dependencies.
# This is actually what you want for production — it's reproducible.
```

---

## Step 2: The Dockerfile — Every Line Explained

```dockerfile title="Dockerfile"
# ─────────────────────────────────────────────────────────────────
# LINE 1: Choose the base image
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# python:3.11-slim vs python:3.11
# - python:3.11       → 1.01 GB (includes build tools, debuggers, docs)
# - python:3.11-slim  → 120 MB (minimal Debian + Python, nothing extra)
# We use slim because we don't need build tools at runtime.
# The 880 MB difference matters when:
# - Pulling over slow networks
# - Storing many image versions in a registry
# - Kubernetes pods starting quickly

# ─────────────────────────────────────────────────────────────────
# LINE 2-3: Set environment variables
# ─────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# PYTHONDONTWRITEBYTECODE=1:
#   Prevents Python from creating .pyc cache files inside the container.
#   These files cache compiled bytecode but aren't useful in containers
#   (they'd be recreated anyway on next startup, and they clutter the image).

# PYTHONUNBUFFERED=1:
#   Normally Python buffers its output — print() text is collected in memory
#   and sent to stdout in chunks. Inside Docker, this means your log output
#   might not appear in 'docker logs' for a while.
#   With PYTHONUNBUFFERED=1, output is written immediately.
#   This is critical for production — you need to see logs in real time.

# ─────────────────────────────────────────────────────────────────
# LINE 4: Set working directory
# ─────────────────────────────────────────────────────────────────
WORKDIR /app

# Creates /app inside the container and makes it the current directory.
# All subsequent COPY commands are relative to this directory.
# Your files will be at /app/main.py, /app/artifacts/model.pkl, etc.
# Convention: use /app for Python web applications.

# ─────────────────────────────────────────────────────────────────
# LINE 5-6: Install Python dependencies
# ─────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WHY COPY requirements.txt BEFORE COPY . . ?
#
# Docker layer caching. Each instruction = one layer. Layers are cached.
# If nothing changed in that layer, Docker reuses the cache.
#
# With correct order:
#   requirements.txt unchanged → pip install layer is CACHED (skipped!)
#   Your code changes            → only COPY . . layer re-runs
#   Result: rebuild takes 2 seconds instead of 5 minutes
#
# With wrong order (COPY . . first):
#   Any code change              → invalidates the layer
#   pip install re-runs          → 5 minutes every time
#
# --no-cache-dir: don't store pip's download cache inside the image.
#   pip keeps a cache to speed up reinstallation. Inside a container,
#   you'll never reinstall packages, so the cache is pure waste.
#   Omitting this adds ~100MB to your image for no benefit.

# ─────────────────────────────────────────────────────────────────
# LINE 7: Copy application code
# ─────────────────────────────────────────────────────────────────
COPY . .

# Copies everything from your current directory to /app inside the container.
# This layer changes on every code edit — that's fine, it's fast (no pip).
# The .dockerignore file controls what gets copied (see below).

# ─────────────────────────────────────────────────────────────────
# LINE 8-9: Create non-root user (security)
# ─────────────────────────────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# By default, containers run as root. This means:
# - If your app is compromised, the attacker has root access inside
# - If there's a container escape vulnerability, they have root on the host
#
# Best practice: create a non-root user and switch to it.
# --disabled-password: no password (login via key only)
# --gecos "": skip the name/info prompts
# chown: give appuser ownership of /app so they can write files
# USER appuser: all subsequent commands run as this user

# ─────────────────────────────────────────────────────────────────
# LINE 10: Document the port (informational only)
# ─────────────────────────────────────────────────────────────────
EXPOSE 8000

# This does NOT actually open any ports. It's documentation.
# It tells humans and tools: "this container expects traffic on 8000."
# The actual port mapping happens with -p 8000:8000 at runtime.

# ─────────────────────────────────────────────────────────────────
# LINE 11: Health check
# ─────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Docker periodically runs this command to check if the container is healthy.
# If it fails 3 times (--retries=3), Docker marks the container as "unhealthy".
# Orchestrators (Kubernetes, ECS) use this to decide whether to restart it.
#
# --interval=30s: check every 30 seconds
# --timeout=10s: fail if no response in 10 seconds
# --start-period=15s: wait 15 seconds before starting checks
#                     (gives your model time to load before we expect /health to work)
# --retries=3: fail 3 checks in a row before marking unhealthy

# ─────────────────────────────────────────────────────────────────
# LINE 12: Default run command
# ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# This runs when the container starts.
# Array form ["uvicorn", "main:app", ...] vs string form "uvicorn main:app ...":
# - Array form (exec form): directly runs the process — receives signals properly
# - String form (shell form): runs through /bin/sh — PID 1 is the shell, not uvicorn
# Always use array form in production: Ctrl+C, SIGTERM, graceful shutdown all work correctly.
#
# --host 0.0.0.0 is CRITICAL:
# Without it, Uvicorn only listens on 127.0.0.1 (inside the container).
# Docker's port mapping sends traffic to the container's network interface,
# not to 127.0.0.1. Without 0.0.0.0, you'd get "connection refused" despite
# the port being "mapped". Always include --host 0.0.0.0.
```

---

## Step 3: The `.dockerignore` — Don't Copy Garbage

Without `.dockerignore`, `COPY . .` copies your `.git` folder (hundreds of MB), your `.env` secrets, your notebooks, test files, and everything else. This makes your image huge and potentially exposes secrets.

```dockerignore title=".dockerignore"
# Python bytecode — regenerated automatically, no value in image
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/

# Virtual environment — we install fresh inside the container
.venv/
venv/
env/

# Secrets — CRITICAL: never bake secrets into images
.env
.env.*
!.env.example      # exception: template is safe to include

# Development and CI artifacts
.git/
.gitignore
.github/
*.md
Makefile
Dockerfile*        # you can exclude these too — they're not needed at runtime

# Data science (often large, not needed for inference)
notebooks/
*.ipynb
data/
*.csv
experiments/

# Tests (not needed in production image)
tests/
pytest.ini
.pytest_cache/
coverage.xml
htmlcov/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# NOTE: artifacts/ directory is NOT excluded
# Your model.pkl must be in the image for predictions to work!
```

---

## Step 4: Build, Test, Push

```bash
# ─── Build ────────────────────────────────────────────────────────
docker build -t insurance-api:1.0 .

# Watch the output — you'll see each layer being built
# Look for "CACHED" on the pip install layer for fast rebuilds

# ─── Check image size ─────────────────────────────────────────────
docker images | grep insurance-api
# REPOSITORY      TAG  IMAGE ID      SIZE
# insurance-api   1.0  a3f8b2c1de   487MB

# ─── Test locally ────────────────────────────────────────────────
docker run -d \
  --name test-api \
  -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  insurance-api:1.0

# Verify it started correctly
docker logs test-api
# Should see: "Model loaded successfully" and "Application startup complete"

# Test the health endpoint
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0"}

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":"male","bmi":27.9,"children":2,"smoker":"no","region":"southeast"}'
# {"prediction":"medium","confidence":0.72,...}

# Clean up test container
docker stop test-api && docker rm test-api

# ─── Push to Docker Hub ───────────────────────────────────────────
docker login
docker tag insurance-api:1.0 yourusername/insurance-api:1.0
docker tag insurance-api:1.0 yourusername/insurance-api:latest
docker push yourusername/insurance-api:1.0
docker push yourusername/insurance-api:latest
```

---

## Docker Compose — Running Multiple Services

When your application needs more than one service (API + Redis + database), Docker Compose lets you define and run them all together:

```yaml title="docker-compose.yml"
version: "3.9"

services:
  # Your FastAPI application
  api:
    build: .                              # build from Dockerfile in current dir
    container_name: insurance-api
    ports:
      - "8000:8000"                       # host:container
    environment:
      - API_KEY=${API_KEY}               # reads API_KEY from your shell or .env
      - LOG_LEVEL=INFO
      - MODEL_PATH=/app/artifacts/insurance_model.pkl
    volumes:
      - ./artifacts:/app/artifacts       # mount model directory from host
                                          # → update model without rebuilding image
    depends_on:
      redis:
        condition: service_healthy        # don't start until redis is ready
    restart: unless-stopped               # restart if api crashes

  # Redis for prediction caching
  redis:
    image: redis:7-alpine                 # use official image, no Dockerfile needed
    container_name: insurance-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
```

```bash
# Start all services
docker compose up -d

# View all logs together (streaming)
docker compose logs -f

# Rebuild after code changes
docker compose up -d --build

# Stop everything
docker compose down

# Stop and remove volumes (WARNING: deletes persistent data)
docker compose down -v
```

---

## Q&A

**Q: My container starts but I can't reach port 8000. What's wrong?**

Almost always `--host 0.0.0.0` is missing from your Uvicorn command. Check your `CMD` in the Dockerfile. Without it, Uvicorn listens on `127.0.0.1` inside the container — not on the interface Docker exposes to the outside.

**Q: How do I update the model without rebuilding the entire image?**

Use a volume mount:
```bash
docker run -v /host/path/to/artifacts:/app/artifacts insurance-api:1.0
```
Now you can replace `model.pkl` on the host and restart the container — no rebuild. For automatic hot-reload, implement a `/model/reload` endpoint.

**Q: The image is 800MB. How do I make it smaller?**

1. Start with `python:3.11-slim` (or even `python:3.11-alpine` — but Alpine has C library compatibility issues with some ML packages)
2. Add a comprehensive `.dockerignore`
3. Use `pip install --no-cache-dir`
4. Use multi-stage builds (compile/install in one stage, copy only artifacts to slim runtime stage)
5. Remove unnecessary apt packages after installing

**Q: What is Docker Compose for and when should I use it?**

Docker Compose orchestrates multiple containers together on a single host. Use it for local development when your app needs multiple services (API, database, Redis, message queue). For production on multiple machines, use Kubernetes or AWS ECS/Fargate instead.

**Q: Should I `docker push` and `docker pull`, or copy the image file directly?**

Always use a registry (Docker Hub, AWS ECR, GitHub Container Registry). Copying image `.tar` files manually is error-prone and doesn't scale. Registries provide versioning, access control, and CDN distribution.
