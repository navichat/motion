# Docker Usage Guide

This guide explains how to use Docker with the Motion package for development, testing, and deployment.

## Quick Start

### Prerequisites

- Docker 20.10+ 
- Docker Compose 2.0+
- 4GB+ RAM available for containers
- 10GB+ disk space for images and builds

### Running the Application

```bash
# Start development environment
docker-compose up motion-dev

# Access the application
open http://localhost:8080

# Stop the environment
docker-compose down
```

## Docker Commands

### Building Images

```bash
# Build all images
./scripts/build-docker.sh

# Build specific target
docker build --target testing -t motion:test .

# Build for specific platform
docker buildx build --platform linux/amd64,linux/arm64 .
```

### Running Containers

```bash
# Development mode (with source mounting)
docker-compose up motion-dev

# Testing mode
docker run --rm motion:test npm run test:unit

# Production mode
docker run -p 8080:8080 motion:production
```

### Managing Containers

```bash
# List running containers
docker ps

# View logs
docker logs motion-dev

# Execute commands in container
docker exec -it motion-dev bash

# Stop all containers
docker-compose down
```

## Development Workflow

### Live Development

The development container supports live reloading:

1. **Start development container**:
   ```bash
   docker-compose up motion-dev
   ```

2. **Edit source code** - changes are automatically reflected

3. **View logs**:
   ```bash
   docker-compose logs -f motion-dev
   ```

### Testing Changes

Run tests without rebuilding:

```bash
# Run all tests
docker-compose run --rm motion-test npm test

# Run specific test suite
docker-compose run --rm motion-test npm run test:unit

# Run with debug output
docker-compose run --rm -e DEBUG=1 motion-test npm run test:unit
```

## Test Environments

### Unit Testing

```bash
# Using docker-compose
docker-compose --profile unit-testing up

# Using script
./scripts/test-docker.sh unit

# Manual run
docker run --rm \
  -v $(pwd)/test-results:/app/test-results \
  motion:test \
  npm run test:unit
```

### E2E Testing

```bash
# Full E2E suite
docker-compose --profile e2e-testing up

# Specific E2E tests
./scripts/test-docker.sh e2e

# With browser visible (for debugging)
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  motion:test \
  npm run test:e2e:root-smokes
```

### Performance Testing

```bash
# Performance benchmarks
docker-compose --profile perf-testing up

# Task manager performance
./scripts/test-docker.sh taskmanager

# Custom performance test
docker run --rm motion:test npm run test:perf
```

## Production Deployment

### Building Production Image

```bash
# Build optimized production image
docker build --target production -t motion:prod .

# With version tag
./scripts/build-docker.sh v1.2.3
```

### Running Production Container

```bash
# Simple run
docker run -p 8080:8080 motion:prod

# With health checks and restart policy
docker run -d \
  --name motion-production \
  --restart unless-stopped \
  -p 8080:8080 \
  --health-cmd="curl -f http://localhost:8080/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  motion:prod

# Using docker-compose
docker-compose -f docker-compose.production.yml up -d
```

### Production Monitoring

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View production logs
docker logs motion-production --tail 100 -f

# Monitor resource usage
docker stats motion-production
```

## Advanced Usage

### Custom Builds

Build with custom configurations:

```bash
# Development with custom port
docker build \
  --target development \
  --build-arg PORT=3000 \
  -t motion:dev-custom .

# Testing with specific Node version
docker build \
  --build-arg NODE_VERSION=18.19.0 \
  --target testing \
  -t motion:test-node18 .
```

### Multi-Platform Builds

```bash
# Setup buildx (one time)
docker buildx create --name motion-builder --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target production \
  --push \
  -t ghcr.io/navichat/motion:latest .
```

### Volume Mounting

For development and debugging:

```bash
# Mount source code for live editing
docker run -it \
  -v $(pwd):/app \
  -v /app/node_modules \
  -p 8080:8080 \
  motion:dev

# Mount test results directory
docker run --rm \
  -v $(pwd)/test-results:/app/test-results \
  -v $(pwd)/playwright-report:/app/playwright-report \
  motion:test npm test
```

## Docker Compose Profiles

Use profiles to run specific configurations:

```bash
# Development profile (default)
docker-compose up

# Testing profiles
docker-compose --profile testing up
docker-compose --profile unit-testing up
docker-compose --profile ml-testing up
docker-compose --profile e2e-testing up
docker-compose --profile perf-testing up

# Web viewer profile
docker-compose --profile web-viewer up

# Multiple profiles
docker-compose --profile testing --profile web-viewer up
```

## Environment Variables

### Development

```bash
# Set environment variables
export NODE_ENV=development
export DEBUG=motion:*
export PORT=3000

# Run with custom environment
docker-compose up motion-dev
```

### Testing

```bash
# CI environment
export CI=true
export NODE_ENV=test

# Test-specific variables
export RUN_REAL_INFERENCE=1
export TIMEOUT=900
```

### Production

```bash
# Production configuration
export NODE_ENV=production
export LOG_LEVEL=info
export HEALTH_CHECK_INTERVAL=30s
```

## Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check what's using port 8080
   lsof -i :8080
   
   # Use different port
   docker run -p 8081:8080 motion:dev
   ```

2. **Out of disk space**:
   ```bash
   # Clean up unused containers
   docker container prune
   
   # Clean up unused images
   docker image prune
   
   # Clean up everything (use with caution)
   docker system prune -a
   ```

3. **Permission issues**:
   ```bash
   # Fix ownership of mounted volumes
   sudo chown -R $USER:$USER test-results/
   
   # Run as current user
   docker run --rm \
     --user $(id -u):$(id -g) \
     -v $(pwd):/app \
     motion:test npm test
   ```

4. **Memory issues**:
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Advanced
   
   # Or run with memory limit
   docker run --memory=4g motion:test npm test
   ```

5. **Network issues**:
   ```bash
   # Reset Docker networks
   docker network prune
   
   # Check container connectivity
   docker exec motion-dev ping google.com
   ```

### Debug Mode

Enable detailed logging:

```bash
# Docker build debug
DOCKER_BUILDKIT=1 docker build --progress=plain .

# Container debug
docker run -it --entrypoint=/bin/bash motion:dev

# Compose debug
docker-compose --verbose up
```

### Log Analysis

```bash
# Follow logs from all services
docker-compose logs -f

# Logs from specific service
docker-compose logs -f motion-dev

# Export logs to file
docker logs motion-dev > container.log 2>&1

# Search logs
docker logs motion-dev 2>&1 | grep ERROR
```

## Performance Tips

### Build Optimization

```bash
# Use build cache
export DOCKER_BUILDKIT=1

# Parallel builds
docker buildx build --platform linux/amd64,linux/arm64

# Cache mount for npm
docker build --target development \
  --mount=type=cache,target=/root/.npm \
  .
```

### Runtime Optimization

```bash
# Limit resource usage
docker run \
  --memory=2g \
  --cpus=1.0 \
  --pids-limit=100 \
  motion:prod

# Use init system for proper signal handling
docker run --init motion:prod
```

### Image Size Optimization

```bash
# Check image size
docker images motion

# Analyze layers
docker history motion:prod

# Use multi-stage builds (already implemented)
# Use .dockerignore to exclude unnecessary files
```

## Security

### Container Security

```bash
# Run as non-root user (production images do this)
docker run --user 1001:1001 motion:prod

# Read-only filesystem
docker run --read-only motion:prod

# No new privileges
docker run --security-opt=no-new-privileges motion:prod

# Drop capabilities
docker run --cap-drop=ALL motion:prod
```

### Secrets Management

```bash
# Use Docker secrets (Swarm mode)
echo "secret_value" | docker secret create my_secret -

# Or environment files
docker run --env-file .env.production motion:prod

# Or mount secrets
docker run -v /host/secrets:/secrets:ro motion:prod
```

## Maintenance

### Regular Tasks

```bash
# Update base images
docker pull node:18-bullseye
docker pull node:18-alpine

# Rebuild with latest base
docker build --no-cache --target production .

# Clean up old images
docker image prune -a

# Update docker-compose
docker-compose pull
docker-compose up -d
```

### Backup and Restore

```bash
# Export container
docker export motion-dev > motion-container.tar

# Save image
docker save motion:prod > motion-image.tar

# Load image
docker load < motion-image.tar

# Import container
docker import motion-container.tar motion:restored
```

---

**Last Updated**: October 2024  
**Version**: 1.0.0