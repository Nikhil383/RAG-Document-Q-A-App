# Docker Deployment Guide

This guide explains how to build and run the RAG Document Q&A application using Docker.

## Prerequisites

- Docker installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (included with Docker Desktop)

## Quick Start with Docker Compose

The easiest way to run the application is using Docker Compose:

```bash
# Build and start the application
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

The application will be available at `http://localhost:5000`

To stop the application:
```bash
docker-compose down
```

## Manual Docker Commands

### Build the Docker Image

```bash
docker build -t rag-document-qa .
```

### Run the Container

```bash
docker run -d \
  --name rag-app \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  rag-document-qa
```

**For Windows PowerShell:**
```powershell
docker run -d `
  --name rag-app `
  -p 5000:5000 `
  -v ${PWD}/uploads:/app/uploads `
  rag-document-qa
```

### Useful Docker Commands

```bash
# View running containers
docker ps

# View application logs
docker logs rag-app

# Follow logs in real-time
docker logs -f rag-app

# Stop the container
docker stop rag-app

# Start the container
docker start rag-app

# Remove the container
docker rm rag-app

# Remove the image
docker rmi rag-document-qa
```

## Docker Configuration Details

### Dockerfile Features

- **Base Image**: Python 3.11 slim (lightweight)
- **Optimized Layers**: Requirements installed separately for better caching
- **Production Ready**: Environment variables configured for production
- **Port**: Exposes port 5000

### Volume Mounting

The `uploads/` directory is mounted as a volume to persist uploaded documents between container restarts.

### Environment Variables

You can customize the application by setting environment variables:

```bash
docker run -d \
  --name rag-app \
  -p 5000:5000 \
  -e FLASK_ENV=development \
  -v $(pwd)/uploads:/app/uploads \
  rag-document-qa
```

## Troubleshooting

### Container won't start
```bash
# Check logs for errors
docker logs rag-app
```

### Port already in use
Change the host port mapping:
```bash
docker run -d --name rag-app -p 8080:5000 rag-document-qa
```
Then access at `http://localhost:8080`

### Out of memory
The ML models require sufficient RAM. Increase Docker's memory limit in Docker Desktop settings (recommended: 4GB+).

## Production Deployment

For production deployment, consider:

1. **Using a production WSGI server** (add `gunicorn` to requirements.txt):
   ```dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
   ```

2. **Setting up a reverse proxy** (nginx) for SSL/TLS

3. **Implementing persistent storage** for the FAISS index

4. **Resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```
