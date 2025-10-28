# Motion Package CI/CD Setup

This document describes the continuous integration and continuous deployment (CI/CD) setup for the Motion package, including Docker containerization, testing, and deployment workflows.

## Overview

The Motion package uses a modern CI/CD pipeline built with:
- **GitHub Actions** for workflow orchestration
- **Docker** for containerization and consistent environments
- **Multi-stage builds** for optimized images
- **Parallel testing** for faster feedback
- **Automated deployment** to staging and production

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Source Code   │───▶│  GitHub Actions  │───▶│   Docker Hub    │
│   (GitHub)      │    │   (CI/CD)        │    │  (Container     │
└─────────────────┘    └──────────────────┘    │   Registry)     │
                                │               └─────────────────┘
                                ▼
                       ┌──────────────────┐
                       │  Docker Images   │
                       │  - Development   │
                       │  - Testing       │
                       │  - Production    │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Deployments    │
                       │  - Staging       │
                       │  - Production    │
                       └──────────────────┘
```

## Docker Setup

### Multi-Stage Dockerfile

The project uses a multi-stage Dockerfile (`/Dockerfile`) with the following stages:

1. **base**: Common dependencies (Node.js, Python, system packages)
2. **development**: Development environment with full source code
3. **testing**: Testing environment with Xvfb for headless browser testing
4. **ci**: CI-optimized environment for GitHub Actions
5. **production**: Minimal production image

### Docker Images

Images are built and pushed to GitHub Container Registry:

- `ghcr.io/navichat/motion:dev-{version}` - Development environment
- `ghcr.io/navichat/motion:test-{version}` - Testing environment  
- `ghcr.io/navichat/motion:ci-{version}` - CI environment
- `ghcr.io/navichat/motion:prod-{version}` - Production environment
- `ghcr.io/navichat/motion:latest` - Latest production build

## GitHub Actions Workflows

### Main CI/CD Workflow (`.github/workflows/ci-cd.yml`)

The main workflow includes the following jobs:

#### 1. Docker Build (`docker-build`)
- Builds multi-platform Docker images (AMD64, ARM64)
- Pushes images to GitHub Container Registry
- Uses BuildKit cache for faster builds
- Runs on every push and pull request

#### 2. Test in Docker (`test-in-docker`)
- Runs tests in isolated Docker containers
- Matrix strategy for parallel test execution:
  - Unit tests
  - Animation tests  
  - Web tests
  - ML serverless tests
  - E2E smoke tests
  - Task manager tests
- Uploads test artifacts

#### 3. Legacy Tests (`test`)
- Maintains backward compatibility
- Direct testing without Docker
- Will be phased out in favor of Docker-based testing

### Test Suites

The following test suites are available:

```bash
# Unit Tests
npm run test:unit                    # Basic unit tests
npm run test:unit:animation          # Animation system tests
npm run test:unit:web               # Web viewer tests

# ML/AI Tests  
npm run test:ml:serverless          # ML inference tests (serverless)
npm run test:ml:ort-smoke           # ONNX Runtime smoke tests

# E2E Tests
npm run test:e2e:root-smokes        # Core E2E smoke tests
npm run test:e2e:ultimate:smoke     # Ultimate conversation tests

# Performance Tests
npm run test:perf                   # Performance benchmarks
npm run test:taskmanager            # Task manager performance
```

## Docker Compose

### Development (`docker-compose.yml`)

For local development with hot reloading:

```bash
# Start development environment
docker-compose up motion-dev

# Access the application
open http://localhost:8080
```

Profiles available:
- `testing` - Test environment
- `unit-testing` - Unit tests only
- `ml-testing` - ML/AI tests only  
- `e2e-testing` - E2E tests only
- `perf-testing` - Performance tests only
- `web-viewer` - Python web viewer server

### CI (`docker-compose.ci.yml`)

Optimized for continuous integration:

```bash
# Run all CI tests
docker-compose -f docker-compose.ci.yml up

# Run specific test suite
docker-compose -f docker-compose.ci.yml up test-unit
```

## Scripts

### Build Script (`scripts/build-docker.sh`)

Builds and pushes Docker images:

```bash
# Build all images with latest tag
./scripts/build-docker.sh

# Build with specific version
./scripts/build-docker.sh v1.2.3

# Build for specific platform
PLATFORM=linux/amd64 ./scripts/build-docker.sh
```

### Test Script (`scripts/test-docker.sh`)

Runs tests in Docker containers:

```bash
# Run all tests
./scripts/test-docker.sh

# Run specific test suite
./scripts/test-docker.sh unit
./scripts/test-docker.sh ml
./scripts/test-docker.sh e2e

# Run with docker-compose
./scripts/test-docker.sh compose
```

### Deploy Script (`scripts/deploy.sh`)

Deploys to different environments:

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh production v1.2.3

# Start development environment
./scripts/deploy.sh development
```

## Environment Configuration

### GitHub Secrets

Configure the following secrets in your GitHub repository:

```
GITHUB_TOKEN         # Automatically provided by GitHub
STAGING_URL          # Staging environment URL
PRODUCTION_URL       # Production environment URL
```

### Environment Variables

Runtime environment variables:

```bash
# Development
NODE_ENV=development
DISPLAY=:99
PYTHONPATH=/app

# Testing  
NODE_ENV=test
CI=true
DISPLAY=:99

# Production
NODE_ENV=production
ENVIRONMENT=production
```

## Local Development

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for web viewer)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/navichat/motion.git
   cd motion
   ```

2. **Start development environment**:
   ```bash
   docker-compose up motion-dev
   ```

3. **Run tests**:
   ```bash
   ./scripts/test-docker.sh unit
   ```

4. **Build production image**:
   ```bash
   ./scripts/build-docker.sh production
   ```

### Development Workflow

1. **Make changes** to source code
2. **Run tests locally**:
   ```bash
   docker-compose up motion-test-unit
   ```
3. **Build and test** Docker image:
   ```bash
   ./scripts/build-docker.sh dev
   ./scripts/test-docker.sh unit
   ```
4. **Push changes** - CI/CD pipeline will automatically:
   - Build Docker images
   - Run comprehensive tests
   - Deploy to staging (if on develop branch)
   - Deploy to production (if on main branch)

## Testing Strategy

### Test Pyramid

```
    ┌─────────────────┐
    │       E2E       │  ← Fewer, high-value integration tests
    │   (Playwright)  │
    ├─────────────────┤
    │   Integration   │  ← API and component integration
    │     Tests       │
    ├─────────────────┤
    │   Unit Tests    │  ← Many, fast, isolated tests
    │   (Jest/Mocha)  │
    └─────────────────┘
```

### Test Types

1. **Unit Tests**: Fast, isolated component testing
2. **Integration Tests**: API and service integration
3. **E2E Tests**: Full user journey testing with Playwright
4. **Performance Tests**: Load and performance benchmarks
5. **Visual Regression**: Screenshot-based UI testing

### Test Environment

All tests run in Docker containers with:
- Headless Chrome/Firefox via Playwright
- Xvfb for display server
- Consistent Node.js and Python environments
- Isolated file systems
- Parallel execution capabilities

## Deployment

### Environments

1. **Development**: Local development with hot reloading
2. **Staging**: Pre-production testing environment
3. **Production**: Live environment with optimized builds

### Deployment Process

1. **Automatic Triggers**:
   - Staging: Push to `develop` branch
   - Production: Push to `main` branch

2. **Manual Deployment**:
   ```bash
   # Deploy specific version to staging
   ./scripts/deploy.sh staging v1.2.3
   
   # Deploy to production (requires confirmation)
   ./scripts/deploy.sh production v1.2.3
   ```

3. **Rollback**:
   ```bash
   # Rollback to previous version
   ./scripts/deploy.sh production v1.2.2
   ```

### Health Checks

All deployed services include health checks:
- HTTP endpoint: `/health`
- Docker health check with retries
- Automatic container restart on failure

## Monitoring and Observability

### Metrics

- Build success/failure rates
- Test execution times
- Deployment frequency
- Container resource usage

### Logs

- Application logs via Docker logs
- Build logs in GitHub Actions
- Test output artifacts
- Performance metrics

### Alerts

- Failed builds notify via GitHub
- Health check failures trigger restarts
- Performance regression detection

## Troubleshooting

### Common Issues

1. **Build Failures**:
   ```bash
   # Check build logs
   docker-compose logs motion-dev
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

2. **Test Failures**:
   ```bash
   # Run specific test with verbose output
   ./scripts/test-docker.sh unit
   
   # Check test artifacts
   ls -la test-results/ playwright-report/
   ```

3. **Deployment Issues**:
   ```bash
   # Check deployment logs
   docker-compose -f docker-compose.production.yml logs
   
   # Verify health status
   curl http://localhost:8080/health
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Enable debug in docker-compose
export DEBUG=motion:*
docker-compose up motion-dev

# Run tests with debug output  
DEBUG=1 ./scripts/test-docker.sh unit
```

## Performance Optimization

### Build Optimization

- Multi-stage builds reduce image size
- BuildKit cache speeds up rebuilds
- Parallel test execution
- Layer caching in CI

### Runtime Optimization

- Production images use Alpine Linux
- Minimal dependencies in production
- Resource limits in docker-compose
- Health checks for reliability

## Security

### Container Security

- Non-root user in production
- Minimal attack surface
- Regular base image updates
- Security scanning in CI

### Secrets Management

- GitHub secrets for sensitive data
- Environment-specific configuration
- No secrets in Docker images
- Secure registry authentication

## Contributing

### Adding Tests

1. Create test files following naming convention
2. Add test scripts to `package.json`
3. Update test matrix in GitHub Actions
4. Document test purpose and requirements

### Modifying CI/CD

1. Test changes in feature branch
2. Update documentation
3. Verify all environments work
4. Get team review before merging

## Support

For issues with the CI/CD setup:

1. Check the [troubleshooting section](#troubleshooting)
2. Review GitHub Actions logs
3. Check Docker container logs
4. Open an issue with detailed information

---

**Last Updated**: October 2024  
**Version**: 1.0.0