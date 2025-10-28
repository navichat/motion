# Multi-stage Docker build for Motion Package

# Base image with Node.js and Python
FROM node:18-bullseye as base

# Install system dependencies required for testing and AI/ML workloads
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    git \
    # Browser testing dependencies
    xvfb \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    libgtk-3-0 \
    libgconf-2-4 \
    # AI/ML dependencies
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install timeout command if not available
RUN which timeout || (apt-get update && apt-get install -y coreutils)

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --include=dev

# Install Playwright browsers with all dependencies
RUN npx playwright install --with-deps

# Development stage
FROM base as development

# Copy entire source code
COPY . .

# Set environment variables for development
ENV NODE_ENV=development
ENV DISPLAY=:99
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p test-results playwright-report

# Expose ports for development servers
EXPOSE 8080 3000 3001 3002

# Start Xvfb for headless browser testing and development server
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & npm run serve"]

# Testing stage
FROM base as testing

# Copy entire source code
COPY . .

# Set environment variables for testing
ENV NODE_ENV=test
ENV DISPLAY=:99
ENV CI=true
ENV PYTHONPATH=/app

# Create test output directories
RUN mkdir -p test-results playwright-report ai-inference-results

# Set proper permissions
RUN chmod +x validate-fixes.sh run-ai-inference-test.sh

# Health check for testing environment
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD pgrep Xvfb > /dev/null || exit 1

# Default command runs comprehensive tests
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm test"]

# Specific test runners for different test suites
FROM testing as test-unit
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm run test:unit"]

FROM testing as test-ml
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm run test:ml:serverless"]

FROM testing as test-e2e
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm run test:e2e:root-smokes"]

FROM testing as test-performance
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm run test:perf"]

# CI stage - optimized for continuous integration
FROM testing as ci

# Install additional CI tools
RUN npm install -g npm-check-updates

# Copy CI-specific configurations
COPY .github/ ./.github/

# Validate package security
RUN npm audit --audit-level moderate || true

# Default CI command
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm run test:unit && npm run test:ml:serverless"]

# Production stage - minimal image for deployment
FROM node:18-alpine as production

# Install Python and system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl

# Create app user for security
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set working directory
WORKDIR /app

# Copy production files
COPY --chown=appuser:appgroup package*.json ./
COPY --chown=appuser:appgroup *.js ./
COPY --chown=appuser:appgroup *.md ./
COPY --chown=appuser:appgroup dist-web_viewer/ ./dist-web_viewer/

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Switch to non-root user
USER appuser

# Expose default port
EXPOSE 8080

# Health check for production
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Production server command
CMD ["npm", "run", "serve"]