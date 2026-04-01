---
id: deploy-aws
title: "12 · Deploy FastAPI on AWS"
sidebar_label: "12 · Deploy on AWS"
sidebar_position: 12
tags: [aws, ec2, ecr, ecs, deployment, cloud, mlops]
---

# Deploy a FastAPI API on AWS

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=X0lnToYN21k) · **Series:** FastAPI for ML – CampusX

---

## Deployment Overview

We have a Dockerized FastAPI ML API. Now we deploy it to AWS.

The video covers deploying directly to an **EC2 instance** — the simplest approach. We also cover **ECS Fargate** (the production-grade managed approach) in the extended section.

```
Local Dev → Docker Image → AWS ECR → EC2 / ECS → Public URL
```

---

## Option A: Deploy on EC2 (Video Method)

### Architecture

```
Internet
   │
   ▼
EC2 Instance (Ubuntu 22.04)
   ├── Docker Engine installed
   ├── Port 8000 open in Security Group
   └── Docker container running insurance-api:1.0
         ├── FastAPI + Uvicorn
         └── model.pkl baked in
```

### Step 1: Launch an EC2 Instance

Via AWS Console:
1. Go to **EC2 → Launch Instance**
2. Choose **Ubuntu 22.04 LTS** AMI
3. Instance type: **t2.medium** (2 vCPU, 4 GB RAM) for ML models
4. Key pair: create a new `.pem` key file
5. Security Group — add inbound rules:
   - `SSH` port 22 from your IP
   - `Custom TCP` port 8000 from `0.0.0.0/0` (or your frontend's IP)
6. Launch

### Step 2: SSH into the Instance

```bash
# Change permissions on your key
chmod 400 my-key.pem

# SSH into instance
ssh -i my-key.pem ubuntu@<your-ec2-public-ip>
```

### Step 3: Install Docker on EC2

```bash
# Update package list
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com | sh

# Allow ubuntu user to run docker without sudo
sudo usermod -aG docker ubuntu

# Logout and re-login to apply group change
exit
ssh -i my-key.pem ubuntu@<your-ec2-public-ip>

# Verify
docker --version
```

### Step 4: Pull and Run Your Docker Image

```bash
# Option A: Pull from Docker Hub
docker pull yourusername/insurance-api:1.0

# Option B: Pull from AWS ECR (see below)
# docker pull 123456789.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0

# Run the container
docker run -d \
  --name insurance-api \
  -p 8000:8000 \
  -e API_KEY=your-secret-key \
  --restart unless-stopped \
  yourusername/insurance-api:1.0

# Verify it's running
docker ps
docker logs insurance-api

# Test from EC2
curl http://localhost:8000/health

# Test from your laptop
curl http://<your-ec2-public-ip>:8000/health
```

---

## Option B: Using AWS ECR (Recommended)

ECR (Elastic Container Registry) is AWS's private Docker registry — better for production than Docker Hub.

### Push to ECR

```bash
# On your LOCAL machine:

# 1. Create ECR repository (first time only)
aws ecr create-repository \
  --repository-name insurance-api \
  --region ap-south-1

# 2. Login to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.ap-south-1.amazonaws.com

# 3. Tag your image for ECR
docker tag insurance-api:1.0 \
  <account-id>.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0

# 4. Push
docker push \
  <account-id>.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0
```

### Pull from ECR on EC2

```bash
# On EC2:

# Install AWS CLI
sudo apt-get install -y awscli

# Configure credentials (or use IAM role — preferred)
aws configure

# Login to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.ap-south-1.amazonaws.com

# Pull and run
docker pull <account-id>.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0
docker run -d -p 8000:8000 --restart unless-stopped \
  <account-id>.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0
```

---

## Updating the Deployment

When you push a new model or code change:

```bash
# LOCAL: Build and push new version
docker build -t insurance-api:1.1 .
docker tag insurance-api:1.1 yourusername/insurance-api:latest
docker push yourusername/insurance-api:latest

# EC2: Pull latest and restart
ssh -i my-key.pem ubuntu@<ec2-ip>
docker pull yourusername/insurance-api:latest
docker stop insurance-api
docker rm insurance-api
docker run -d --name insurance-api -p 8000:8000 \
  --restart unless-stopped yourusername/insurance-api:latest
```

---

## Option C: AWS ECS Fargate (Production Grade)

ECS Fargate is serverless container orchestration — you don't manage EC2 servers. AWS handles scaling, health monitoring, and restarts automatically.

```
Client
  │
  ▼
Application Load Balancer (ALB)
  │   port 443 (HTTPS)
  ├── Target Group
  │     └── ECS Service
  │           ├── Task (Container 1) ← insurance-api
  │           ├── Task (Container 2) ← insurance-api
  │           └── Task (Container 3) ← insurance-api
  └── Auto Scaling (min 2, max 10 tasks based on CPU/RPS)
```

### ECS Fargate Setup (Outline)

```bash
# 1. Create ECS Cluster
aws ecs create-cluster --cluster-name ml-api-cluster

# 2. Create Task Definition (describes your container)
# → Set CPU (0.5 vCPU), Memory (1 GB), Image URI, Port, Env vars

# 3. Create ECS Service
# → Links the task definition to the cluster
# → Connects to Application Load Balancer
# → Sets min/max task count for auto-scaling

# 4. Create ALB + Target Group
# → Handles incoming HTTPS traffic
# → Routes to healthy tasks

# 5. Set up Auto Scaling
# → Scale out when CPU > 70%
# → Scale in when CPU < 30%
```

---

## Nginx Reverse Proxy (Production Pattern)

For production on EC2, don't expose Uvicorn directly. Put Nginx in front:

```nginx title="/etc/nginx/sites-available/insurance-api"
server {
    listen 80;
    server_name api.mycompany.com;

    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name api.mycompany.com;

    ssl_certificate /etc/letsencrypt/live/api.mycompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.mycompany.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts (important for ML inference)
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
    }
}
```

```bash
# Install Nginx and Certbot (HTTPS)
sudo apt install -y nginx certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d api.mycompany.com

# Enable site
sudo ln -s /etc/nginx/sites-available/insurance-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## Topics Not Covered in the Video

### IAM Roles — Secure Access to AWS Services

Instead of putting AWS credentials on EC2, attach an **IAM Role**:

```json
// IAM Policy: allow EC2 to pull from ECR + read from S3
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::my-model-bucket/*"
    }
  ]
}
```

Attach the role to your EC2 instance — no credentials needed in your code.

### Load Model from S3 at Startup

```python title="services/model_service.py"
import boto3
import joblib
import io

def load_model_from_s3(bucket: str, key: str):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    model = joblib.load(io.BytesIO(obj["Body"].read()))
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["model"] = load_model_from_s3(
        bucket="my-model-bucket",
        key="insurance/model_v1.0.pkl"
    )
    yield
```

### GitHub Actions CI/CD Pipeline

```yaml title=".github/workflows/deploy.yml"
name: Build and Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, Tag, Push
        run: |
          docker build -t ${{ secrets.ECR_REGISTRY }}/insurance-api:${{ github.sha }} .
          docker push ${{ secrets.ECR_REGISTRY }}/insurance-api:${{ github.sha }}

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster ml-api-cluster \
            --service insurance-api-service \
            --force-new-deployment
```

---

## AWS Cost Estimate (India Region — ap-south-1)

| Service | Config | Monthly Cost |
|---------|--------|-------------|
| EC2 t2.medium | On-demand, 24/7 | ~$35/mo |
| EC2 t2.medium | Reserved 1-year | ~$18/mo |
| ECS Fargate | 0.5 vCPU, 1 GB | ~$15/mo |
| ECR | 10 GB storage | ~$1/mo |
| ALB | 1 LCU/hr | ~$18/mo |
| Data Transfer | 10 GB/mo | ~$1/mo |

For a personal/demo project, EC2 t2.micro (free tier) or Fargate with small tasks is cheapest.

---

## Q&A

**Q: EC2 vs ECS Fargate — which should I use?**
> EC2: simpler, full control, cheaper for constant load, need to manage the OS. Fargate: no server management, auto-scaling, pay only for runtime, more expensive at constant load. For ML projects: start with EC2, migrate to ECS when you need auto-scaling or zero-downtime deployments.

**Q: How do I get HTTPS on my EC2?**
> Use **Certbot** with Let's Encrypt (free) behind Nginx. Or put an **Application Load Balancer** (ALB) in front of EC2 — the ALB handles SSL termination and gives you a managed HTTPS endpoint.

**Q: My model is slow. How do I scale the API?**
> Horizontal scaling: run multiple containers / ECS tasks behind an ALB. Vertical scaling: use a larger EC2 instance (c5.xlarge for CPU-bound ML). For GPU: use `p3` or `g4dn` EC2 instances. For serverless inference: look at AWS SageMaker Endpoints.

**Q: What is the difference between ECR and Docker Hub?**
> Docker Hub is a public registry (can also have private repos). ECR is AWS's private registry — integrated with IAM, VPC, and other AWS services. ECR images are stored in the same region as your compute, reducing latency. Use ECR in production; Docker Hub is fine for learning.

**Q: How do I zero-downtime deploy an update?**
> On ECS: update the task definition → `update-service --force-new-deployment`. ECS will spin up new tasks with the new image, wait for them to pass health checks, then terminate old tasks. On EC2: use a blue-green deployment with two instances behind an ALB.

**Q: What's the cheapest way to host a demo ML API?**
> In order of cheapness: (1) AWS EC2 t2.micro (free tier — 12 months), (2) Railway.app / Render.com (free tier), (3) Google Cloud Run (pay-per-request, scales to zero), (4) AWS Lambda + Mangum adapter.

---

## Deployment Decision Tree

```
Need to deploy ML API?
│
├── Just for demo/learning?
│   └── EC2 t2.micro (free tier) or Render.com
│
├── Team project, moderate traffic?
│   └── EC2 t2.medium + Nginx + ECR
│
├── Production, variable traffic?
│   └── ECS Fargate + ALB + Auto Scaling
│
├── Very high throughput / GPU inference?
│   └── AWS SageMaker Endpoints
│
└── Serverless, infrequent requests?
    └── AWS Lambda + Mangum (FastAPI adapter)
```
