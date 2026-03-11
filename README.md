# DevOps Quiz AI — Assignment 8 Starter

A fully-working AI-powered quiz app (Gemini 2.5 Flash). Your task is to
containerise it and build the CI/CD pipeline that delivers it to production.

## Running locally (without Docker)

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in GEMINI_KEY and STUDENT_ID
python app.py
# open http://localhost:5000
```

## Building and running with Docker

**Note:** On macOS, port 5000 is used by AirPlay Receiver. Use port 5001 instead.

```bash
# Build the Docker image
docker build -t quiz-ai .

# Run the container with environment variables
docker run -d -p 5001:5000 --name quiz-ai --env-file .env quiz-ai

# Test the application
open http://localhost:5001
# OR
curl http://localhost:5001

# View container logs
docker logs quiz-ai

# Stop and remove the container
docker stop quiz-ai && docker rm quiz-ai
```

## What you need to create (not included in this starter)

1. ~~`Dockerfile`~~ — **Already provided** (containerises the app)
2. `.github/workflows/ci-cd.yml` — **YOU MUST CREATE THIS** (the CI/CD pipeline)

## Environment variables

| Variable | Where to set |
|---|---|
| `GEMINI_KEY` | `.env` locally · Render Environment Variables in production |
| `STUDENT_ID` | `.env` locally · Render Environment Variables in production |
| `DOCKERHUB_USERNAME` | GitHub Secret |
| `DOCKERHUB_TOKEN` | GitHub Secret |
| `RENDER_DEPLOY_URL` | GitHub Secret |
