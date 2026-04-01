---
id: deploy-aws
title: "12 · Deploy FastAPI on AWS"
sidebar_label: "12 · Deploy on AWS"
sidebar_position: 12
tags: [aws, ec2, ecr, ecs, deployment, cloud, mlops, beginner]
---

# Deploy FastAPI on AWS

> **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=X0lnToYN21k) · **Series:** FastAPI for ML – CampusX

---

## Visual Reference

![Amazon Web Services logo](https://commons.wikimedia.org/wiki/Special:Redirect/file/Amazon_Web_Services_Logo.svg)

Source: [Wikimedia Commons - Amazon Web Services Logo](https://commons.wikimedia.org/wiki/File:Amazon_Web_Services_Logo.svg)

## From Laptop to the Cloud

Your Docker container runs perfectly locally. Now you want it running on a server in the cloud — publicly accessible 24/7, from anywhere in the world.

AWS (Amazon Web Services) is where most production ML APIs live. We cover two approaches:

- **EC2** (this video) — the simplest. You manage a Linux server, install Docker, run your container. Full control, hands-on learning.
- **ECS Fargate** (extended) — serverless containers. AWS manages the servers. You just say "run my container" and it scales automatically.

---

## Understanding Cloud Deployment

Why can't you just keep running it on your laptop? Three reasons:

1. **Availability:** Your laptop sleeps. Clouds don't.
2. **Reachability:** Your laptop's IP address changes. Cloud servers have fixed public IPs (or load balancer URLs).
3. **Scale:** Your laptop's resources are shared with your other work. A cloud server is dedicated to your API.

---

## Option A: Deploying to EC2

### What is EC2?

EC2 (Elastic Compute Cloud) is a virtual Linux server you rent by the hour. You get root access, install Docker, and run your container — identical to running it on your laptop, just on a machine in Amazon's data center.

### Step 1: Launch an EC2 Instance

In the AWS Console:
1. EC2 → **Launch Instance**
2. **Name:** `insurance-api-server`
3. **AMI:** Ubuntu Server 22.04 LTS (free tier eligible)
4. **Instance type:** `t2.medium` (2 vCPU, 4GB RAM) — needed for ML models. `t2.micro` (free tier) works for the lightweight examples but may run out of memory when loading sklearn models.
5. **Key pair:** Create new → download the `.pem` file → keep it safe
6. **Security Group:** Configure inbound rules:
   - SSH: Port 22, Source: My IP (for your terminal access)
   - Custom TCP: Port 8000, Source: 0.0.0.0/0 (allows anyone to call the API)
7. **Launch**

### Step 2: Connect to Your Server

```bash
# Fix permissions on the key (required on Linux/macOS — SSH refuses keys that are too open)
chmod 400 ~/Downloads/my-keypair.pem

# Connect
ssh -i ~/Downloads/my-keypair.pem ubuntu@YOUR-EC2-PUBLIC-IP

# You should see:
# ubuntu@ip-172-31-xx-xx:~$
# You're now INSIDE the EC2 instance!
```

### Step 3: Install Docker on EC2

```bash
# On the EC2 instance:

# Update package list
sudo apt-get update

# Install Docker using the official script (easiest method)
curl -fsSL https://get.docker.com | sh

# Allow the ubuntu user to run docker without sudo
# Without this, every docker command needs 'sudo'
sudo usermod -aG docker ubuntu

# IMPORTANT: Log out and back in for group change to take effect
exit

# Reconnect
ssh -i ~/Downloads/my-keypair.pem ubuntu@YOUR-EC2-PUBLIC-IP

# Verify Docker works
docker --version    # Docker version 25.0.0
docker run hello-world  # should print "Hello from Docker!"
```

### Step 4: Deploy Your Container

```bash
# On the EC2 instance:

# Pull your image from Docker Hub
docker pull yourusername/insurance-api:1.0

# Run the container
docker run -d \
  --name insurance-api \
  -p 8000:8000 \
  -e API_KEY=your-production-api-key \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \      # auto-restart if the container crashes
  yourusername/insurance-api:1.0

# Verify it's running
docker ps
docker logs insurance-api

# Test from INSIDE the EC2 instance
curl http://localhost:8000/health

# 🎉 Test from YOUR LAPTOP (the public internet)
curl http://YOUR-EC2-PUBLIC-IP:8000/health
```

**Your API is now live on the internet!**

---

## Option B: AWS ECR — Private Container Registry

Docker Hub is public. For private images, use AWS ECR (Elastic Container Registry) — integrated with IAM and other AWS services.

```bash
# On your LOCAL machine:

# 1. Create a repository in ECR (one time)
aws ecr create-repository \
  --repository-name insurance-api \
  --region ap-south-1

# 2. Get login credentials (valid for 12 hours)
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  YOUR-ACCOUNT-ID.dkr.ecr.ap-south-1.amazonaws.com

# 3. Tag your image for ECR
docker tag insurance-api:1.0 \
  YOUR-ACCOUNT-ID.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0

# 4. Push to ECR
docker push \
  YOUR-ACCOUNT-ID.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0
```

On EC2, pull from ECR:
```bash
# On EC2 (after configuring AWS credentials or attaching an IAM role):
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  YOUR-ACCOUNT-ID.dkr.ecr.ap-south-1.amazonaws.com

docker pull YOUR-ACCOUNT-ID.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0

docker run -d -p 8000:8000 --restart unless-stopped \
  YOUR-ACCOUNT-ID.dkr.ecr.ap-south-1.amazonaws.com/insurance-api:1.0
```

---

## Adding HTTPS — Your API Needs HTTPS

Currently your API is accessible at `http://YOUR-IP:8000`. For production, you need HTTPS (port 443). Use Nginx as a reverse proxy with a free Let's Encrypt certificate.

Prerequisites: a domain name (e.g., `api.yourdomain.com`) pointing to your EC2 IP address.

```bash
# On EC2:
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Get a free SSL certificate
sudo certbot --nginx -d api.yourdomain.com
# Follow prompts — Certbot handles everything automatically
```

```nginx title="/etc/nginx/sites-available/insurance-api"
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    # Certbot fills in these paths automatically
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        # Forward all traffic to your Docker container
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for ML inference (which can take a few seconds)
        proxy_read_timeout 120s;
        proxy_connect_timeout 60s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

```bash
sudo nginx -t              # test config
sudo systemctl reload nginx
```

Now your API is at `https://api.yourdomain.com/predict` — professional and secure.

---

## Updating Your Deployment

When you push new code or a new model:

```bash
# 1. Build new image on your local machine
docker build -t yourusername/insurance-api:1.1 .
docker push yourusername/insurance-api:1.1

# 2. Update on EC2 (takes ~10 seconds of downtime)
ssh -i my-keypair.pem ubuntu@YOUR-EC2-IP

docker pull yourusername/insurance-api:1.1
docker stop insurance-api
docker rm insurance-api
docker run -d \
  --name insurance-api \
  -p 8000:8000 \
  -e API_KEY=your-production-key \
  --restart unless-stopped \
  yourusername/insurance-api:1.1
```

For zero-downtime deployments, use ECS or Kubernetes (discussed in Advanced Topics).

---

## Option C: ECS Fargate — Serverless Containers (Advanced)

ECS Fargate is AWS's managed container service. You define what to run (CPU, memory, image, environment), and AWS handles the servers, scaling, and health monitoring.

```
Your Docker Image in ECR
         │
         ▼
ECS Task Definition
(how much CPU/RAM, which image, env vars, ports)
         │
         ▼
ECS Service
(how many copies to run, auto-restart on failure)
         │
         ▼
Application Load Balancer
(distributes traffic, SSL termination, health checks)
         │
         ▼
Auto Scaling
(add more tasks when CPU > 70%, remove when < 30%)
```

Fargate vs EC2:

| | EC2 | ECS Fargate |
|-|-----|------------|
| **Management** | You manage OS, Docker | AWS manages everything |
| **Scaling** | Manual | Automatic |
| **Pricing** | Fixed hourly (even when idle) | Per CPU-second (pay for what you use) |
| **Best for** | Learning, constant traffic, cost-sensitive | Production, variable traffic, scale |
| **Setup complexity** | Low | Medium |

---

## AWS Cost Estimates

| Service | Configuration | Monthly cost |
|---------|--------------|-------------|
| EC2 t2.micro | Free tier (12 months) | $0 |
| EC2 t2.medium | On-demand, 24/7 | ~$33/month |
| ECS Fargate | 0.5 vCPU, 1GB RAM | ~$15/month |
| ECR | 10 GB storage | ~$1/month |
| Elastic IP | Static IP address | ~$3/month |
| Let's Encrypt SSL | Via Certbot | Free |

For a personal ML project: start with EC2 t2.micro (free tier) or t2.medium ($33/month). For a team product: ECS Fargate + ALB.

---

## Q&A

**Q: My API is on port 8000 but I can't access it from the internet. What's wrong?**

Check your EC2 Security Group. By default, only port 22 (SSH) is open. You must manually add an inbound rule: Custom TCP, Port 8000, Source 0.0.0.0/0. Security Groups are EC2's firewall.

**Q: What's the difference between a public IP and an Elastic IP?**

A public IP is assigned when the instance starts and **changes** every time you stop and restart the instance. An Elastic IP is a fixed static IP you can reserve — it stays the same even after restarts. For production, always use an Elastic IP (or a load balancer URL) so your clients' configurations don't break.

**Q: How do I SSH in without entering the key file path every time?**

Add it to your `~/.ssh/config`:
```
Host my-api-server
    HostName YOUR-EC2-PUBLIC-IP
    User ubuntu
    IdentityFile ~/Downloads/my-keypair.pem
```
Then: `ssh my-api-server`

**Q: What if my EC2 instance runs out of memory?**

ML models can use significant RAM. `t2.micro` has 1GB RAM — too little for most sklearn models loaded alongside Python and FastAPI. `t2.medium` (4GB) is more appropriate. If you're running out of memory, you'll see the container being killed with exit code 137 (OOM Killed). Check `docker inspect insurance-api` for exit codes.

**Q: Is there a free option for deploying ML APIs?**

Yes: EC2 t2.micro with AWS Free Tier (12 months). Also: Railway.app (free tier), Render.com (free tier), Google Cloud Run (pay-per-request, very cheap for low volume). For learning, all are fine. For production, invest in proper infrastructure.
