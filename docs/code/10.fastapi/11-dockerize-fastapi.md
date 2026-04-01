---
id: dockerize-fastapi
title: "11 · Dockerize a FastAPI API"
sidebar_label: "11 · Dockerize FastAPI"
sidebar_position: 11
tags: [docker, dockerfile, docker-compose, containerization, fastapi, mlops]
---

# Dockerize a FastAPI API Application

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=jlLs6hfAga4) · **Series:** FastAPI for ML – CampusX

---

## What Does "Dockerize" Mean?

Dockerizing your API means:
1. Writing a **Dockerfile** that describes how to build your app image
2. **Building** the image locally
3. **Testing** the container locally
4. **Pushing** the image to a registry (Docker Hub or AWS ECR)
5. Pulling and running on any server

After this, your entire ML API — code, model, dependencies, Python version — is packaged in one portable artifact.

---

## Project Layout Before Dockerizing

```
insurance-api/
├── main.py
├── routers/
│   └── predict.py
├── schemas/
│   └── schemas.py
├── services/
│   └── model_service.py
├── artifacts/
│   └── insurance_model.pkl     ← trained model
├── requirements.txt
└── .env
```

---

## Step 1: `requirements.txt`

Pin your dependencies to avoid "it broke on upgrade":

```txt title="requirements.txt"
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.8.0
pydantic-settings==2.4.0
scikit-learn==1.5.1
joblib==1.4.2
pandas==2.2.2
numpy==1.26.4
python-multipart==0.0.9
```

Generate from your current environment:
```bash
pip freeze > requirements.txt
# Or (cleaner, only direct deps):
pip install pipreqs && pipreqs . --force
```

---

## Step 2: Write the Dockerfile

```dockerfile title="Dockerfile"
# ─── Stage 1: Base Image ─────────────────────────────────────────
FROM python:3.11-slim

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (important for logs)
ENV PYTHONUNBUFFERED=1

# ─── System Dependencies ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working Directory ───────────────────────────────────────────
WORKDIR /app

# ─── Install Python Dependencies ─────────────────────────────────
# Copy requirements first — Docker caches this layer
# If requirements don't change, pip install is skipped on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy Application Code ───────────────────────────────────────
COPY . .

# ─── Non-root User (Security Best Practice) ──────────────────────
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# ─── Expose Port ─────────────────────────────────────────────────
EXPOSE 8000

# ─── Health Check ────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─── Run Command ─────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## Step 3: `.dockerignore`

This prevents unnecessary files from bloating your image:

```dockerignore title=".dockerignore"
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Virtual environment
.venv/
venv/
env/

# Secrets & Config
.env
.env.*
!.env.example

# Development
.git/
.gitignore
*.md
Makefile
notebooks/
tests/

# IDE
.vscode/
.idea/
*.swp

# Data science artifacts (don't include training data)
data/
*.csv
*.ipynb

# Keep: artifacts/model.pkl  ← DO include the model
```

---

## Step 4: Build and Test Locally

```bash
# Build the image
docker build -t insurance-api:1.0 .

# Verify build was successful
docker images | grep insurance-api

# Run locally and test
docker run -d \
  --name insurance-api-test \
  -p 8000:8000 \
  -e API_KEY=test-key \
  insurance-api:1.0

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":30,"sex":"male","bmi":27.5,"children":2,"smoker":"no","region":"southeast"}'

# View logs
docker logs insurance-api-test

# Stop and remove test container
docker stop insurance-api-test && docker rm insurance-api-test
```

---

## Step 5: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag image with your Docker Hub username
docker tag insurance-api:1.0 yourusername/insurance-api:1.0
docker tag insurance-api:1.0 yourusername/insurance-api:latest

# Push
docker push yourusername/insurance-api:1.0
docker push yourusername/insurance-api:latest

# Now anyone can pull and run your API:
docker pull yourusername/insurance-api:1.0
docker run -d -p 8000:8000 yourusername/insurance-api:1.0
```

---

## Docker Compose — Running Multiple Services Locally

When your app needs more than one container (e.g., API + Redis cache), use `docker-compose.yml`:

```yaml title="docker-compose.yml"
version: "3.9"

services:
  api:
    build: .
    image: insurance-api:dev
    container_name: insurance-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/artifacts/insurance_model.pkl
      - LOG_LEVEL=DEBUG
      - API_KEY=${API_KEY}           # read from .env file
    volumes:
      - ./artifacts:/app/artifacts   # mount model directory
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: insurance-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

```bash
# Start all services
docker compose up -d

# View logs from all services
docker compose logs -f

# Rebuild after code changes
docker compose up -d --build

# Tear down
docker compose down

# Tear down + remove volumes
docker compose down -v
```

---

## Topics Not Covered in the Video

### Multi-Stage Build — Smaller Production Images

```dockerfile title="Dockerfile.prod"
# Stage 1: Build dependencies
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Lean runtime image
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy app code
COPY main.py routers/ schemas/ services/ artifacts/ ./

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN adduser --disabled-password appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build sizes comparison:
```
python:3.11 + deps     → ~1.5 GB
python:3.11-slim       → ~350 MB  ← Used in video
Multi-stage build      → ~200 MB  ← Best practice
```

### Build Arguments — Parameterizing the Build

```dockerfile
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG APP_VERSION=unknown
ENV APP_VERSION=${APP_VERSION}
```

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg APP_VERSION=2.1.0 \
  -t insurance-api:2.1.0 .
```

### Pushing to AWS ECR (Used in Next Video)

```bash
# Login to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.ap-south-1.amazonaws.com

# Tag for ECR
docker tag insurance-api:1.0 \
  123456789.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0

# Push
docker push 123456789.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0
```

---

## Q&A

**Q: Why do I need `--host 0.0.0.0` in the CMD?**
> By default, Uvicorn binds to `127.0.0.1` (localhost) — this is only accessible from inside the container. `0.0.0.0` makes it accessible from outside (through Docker's port mapping). Without this, `docker run -p 8000:8000` won't work.

**Q: Why copy `requirements.txt` first before copying the rest of the code?**
> Docker layer caching. If you `COPY . .` first, every code change invalidates the cache and re-runs `pip install`. By copying only `requirements.txt` first, Docker re-runs `pip install` only when requirements change — saving minutes on every build.

**Q: Should I include my `.env` file in the image?**
> **Never.** The `.env` file likely contains secrets (API keys, DB passwords). Add it to `.dockerignore`. Pass environment variables at runtime: `docker run -e KEY=value ...` or use Docker secrets in production.

**Q: My model file is 500MB. Should it be in the Docker image?**
> For simplicity: yes, bake it in. For production: keep the model external (S3 bucket or EFS volume) and download/mount it at startup. This keeps your image small and allows model updates without rebuilding.

**Q: What is the difference between `COPY` and `ADD` in a Dockerfile?**
> `COPY` simply copies files/directories. `ADD` does the same but also auto-extracts `.tar` archives and can fetch URLs. Use `COPY` for everything — `ADD` has surprising behavior. Only use `ADD` when you specifically need its tar-extraction feature.

**Q: How do I run the container with a `.env` file?**
> ```bash
> docker run --env-file .env -p 8000:8000 insurance-api:1.0
> ```
> This reads all variables from `.env` and passes them to the container as environment variables.

---

## Dockerfile Cheat Sheet

```dockerfile
FROM python:3.11-slim          # Base image
WORKDIR /app                   # Set working directory
COPY requirements.txt .        # Copy file(s)
COPY src/ /app/src/            # Copy directory
RUN pip install -r req.txt     # Run shell command (creates layer)
ENV KEY=value                  # Set env variable
ARG BUILD_VAR=default          # Build-time variable
EXPOSE 8000                    # Document exposed port (informational)
HEALTHCHECK ...                # Container health probe
USER appuser                   # Switch to non-root user
VOLUME /data                   # Declare mount point
CMD ["uvicorn", "main:app"]    # Default run command (overridable)
ENTRYPOINT ["uvicorn"]         # Fixed entrypoint (not overridable)
```
