---
id: docker
title: "10 · Docker Crash Course for ML"
sidebar_label: "10 · Docker"
sidebar_position: 10
tags: [docker, containers, dockerfile, images, devops, mlops]
---

# Docker Crash Course for ML Engineers

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=GToyQTGDOS4) · **Series:** FastAPI for ML – CampusX

---

## Why Docker?

The classic ML engineering pain point:

> *"It works on my machine but not on the server."*

Docker solves this by packaging your code + dependencies + runtime into a single, portable unit called a **container**. The container runs identically on:
- Your MacBook
- Your team member's Windows PC
- An Ubuntu EC2 instance
- A Kubernetes pod in GCP

---

## Core Docker Concepts

```
                    ┌─────────────────────────────────┐
                    │         Docker Architecture      │
                    │                                  │
  Dockerfile ──────►│  Docker Engine                  │
  (recipe)          │  ┌────────────────────────────┐ │
                    │  │  Docker Image               │ │◄── docker build
                    │  │  (frozen snapshot)          │ │
                    │  │  ┌──────────────────────┐   │ │
                    │  │  │  Docker Container    │   │ │◄── docker run
                    │  │  │  (running instance)  │   │ │
                    │  │  └──────────────────────┘   │ │
                    │  └────────────────────────────┘ │
                    │                                  │
                    │  Docker Registry (Docker Hub)    │◄── docker push/pull
                    └─────────────────────────────────┘
```

| Concept | Analogy | Description |
|---------|---------|-------------|
| **Dockerfile** | Recipe | Instructions to build the image |
| **Image** | Class / Blueprint | Frozen snapshot of app + dependencies |
| **Container** | Running instance | The live, running application |
| **Registry** | App Store | Central storage for images (Docker Hub, ECR) |
| **Volume** | External hard drive | Persistent storage outside the container |
| **Network** | VPC | How containers communicate |

---

## Installing Docker

```bash
# macOS
brew install --cask docker
# then open Docker Desktop

# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # run docker without sudo

# Verify
docker --version
docker run hello-world
```

---

## Essential Docker Commands

```bash
# ─── Images ────────────────────────────────────────────────────

docker build -t my-ml-api:1.0 .           # Build image from Dockerfile
docker images                              # List local images
docker pull python:3.11-slim               # Download from Docker Hub
docker push username/my-ml-api:1.0         # Push to Docker Hub
docker rmi my-ml-api:1.0                   # Remove image

# ─── Containers ────────────────────────────────────────────────

docker run my-ml-api:1.0                   # Run a container
docker run -d -p 8000:8000 my-ml-api:1.0   # Detached + port mapping
docker run -e API_KEY=secret my-ml-api:1.0 # Pass environment variable
docker run -v /data:/app/data my-ml-api:1.0 # Mount volume

docker ps                                  # List running containers
docker ps -a                               # All containers (inc. stopped)
docker stop <container_id>                 # Stop gracefully
docker kill <container_id>                 # Force stop
docker rm <container_id>                   # Remove container
docker logs <container_id>                 # View container logs
docker logs -f <container_id>              # Follow logs

# ─── Interactive ───────────────────────────────────────────────

docker exec -it <container_id> /bin/bash   # Shell into running container
docker run -it python:3.11-slim /bin/bash  # Interactive new container

# ─── Cleanup ───────────────────────────────────────────────────

docker system prune                        # Remove all unused resources
docker system prune -a --volumes           # Nuclear cleanup
```

---

## Understanding `docker run` Flags

```bash
docker run \
  -d \                          # detached (background)
  --name insurance-api \        # container name
  -p 8000:8000 \                # host_port:container_port
  -e MODEL_PATH=/app/artifacts/model.pkl \  # env variable
  -v $(pwd)/artifacts:/app/artifacts \      # volume mount
  --restart unless-stopped \    # auto-restart on crash
  --memory 512m \               # memory limit
  --cpus 2.0 \                  # CPU limit
  my-ml-api:1.0
```

---

## Docker vs Virtual Machine

| | Virtual Machine | Docker Container |
|-|----------------|----------------|
| **Startup time** | Minutes | Seconds |
| **Size** | GBs (full OS) | MBs (shared kernel) |
| **Isolation** | Complete | Process-level |
| **Performance** | Overhead | Near-native |
| **Use case** | Full OS isolation | App packaging |

---

## Layers — How Docker Images Are Built

Every instruction in a Dockerfile creates a **layer**. Layers are cached — unchanged layers are reused on subsequent builds.

```
FROM python:3.11-slim        ← Layer 1: base OS + Python
WORKDIR /app                 ← Layer 2: set working dir
COPY requirements.txt .      ← Layer 3: copy requirements file
RUN pip install -r req.txt   ← Layer 4: install packages (HEAVY)
COPY . .                     ← Layer 5: copy source code (LIGHT)
CMD ["uvicorn", "main:app"]  ← Layer 6: run command
```

**Key optimization:** Put things that change less frequently earlier in the Dockerfile. This way, the expensive `pip install` layer is cached and not re-run when only your source code changes.

---

## Docker Networking

```bash
# Create a custom network for container-to-container communication
docker network create ml-network

# Run containers on the same network
docker run -d --name api --network ml-network my-ml-api:1.0
docker run -d --name redis --network ml-network redis:7

# Now "api" container can reach "redis" container via hostname "redis"
```

---

## Docker Volumes — Persisting Data

```bash
# Named volume (Docker manages storage)
docker volume create model-data
docker run -v model-data:/app/artifacts my-ml-api:1.0

# Bind mount (you specify the host path)
docker run -v /home/ubuntu/models:/app/artifacts my-ml-api:1.0

# Read-only mount (protect files from being modified)
docker run -v /home/ubuntu/models:/app/artifacts:ro my-ml-api:1.0
```

---

## Topics Not Covered in the Video

### Multi-Stage Builds — Smaller Images

```dockerfile
# Stage 1: Build (heavy)
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime (lean)
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local  # only the installed packages
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This reduces image size significantly by not including build tools in the final image.

### Docker Scout / Image Scanning

```bash
# Check for vulnerabilities in your image
docker scout cves my-ml-api:1.0

# Quick summary
docker scout quickview my-ml-api:1.0
```

### `.dockerignore` — Don't Copy Unnecessary Files

```dockerignore title=".dockerignore"
__pycache__/
*.pyc
*.pyo
.venv/
.env
.git/
.gitignore
*.ipynb
notebooks/
tests/
*.md
Makefile
```

---

## Q&A

**Q: What's the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?**
> `CMD` provides default arguments that can be overridden by `docker run` arguments. `ENTRYPOINT` sets the executable that always runs. Common pattern: `ENTRYPOINT ["uvicorn"]` + `CMD ["main:app", "--host", "0.0.0.0"]`. This lets you override just the arguments while keeping Uvicorn as the entrypoint.

**Q: Why does my container work locally but fail in production?**
> Most common causes: (1) missing `.env` file or environment variables, (2) volume mount paths differ between local and prod, (3) image not pushed after last build, (4) arm64 (Mac M1/M2) image being pulled on x86 server — use `--platform linux/amd64` during build.

**Q: Should I store my ML model inside the Docker image or mount it as a volume?**
> Both approaches are valid: (1) **Baked in**: include `model.pkl` in the image. Simple, self-contained, reproducible. Downside: rebuilding the image for every model update. (2) **Volume-mounted**: mount a model directory. Enables hot-swap without rebuilding. Recommended for frequently updated models.

**Q: How do I reduce my Python Docker image size?**
> (1) Use `python:3.11-slim` instead of `python:3.11`, (2) use multi-stage builds, (3) add `.dockerignore`, (4) use `pip install --no-cache-dir`, (5) uninstall build tools after use.

**Q: What is `docker-compose` and when should I use it?**
> Docker Compose lets you define and run multi-container apps with a YAML file. Use it locally when your app needs multiple services (API + Redis + database). For production, use Kubernetes or ECS instead.
