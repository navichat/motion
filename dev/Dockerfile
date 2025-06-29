# Multi-stage Docker build for Motion Viewer testing environment

# Base image with Node.js and Python
FROM node:18-bullseye as base

# Install system dependencies for testing
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
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
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy package files
COPY dev/tests/package*.json ./tests/
COPY dev/viewer/package*.json ./viewer/
COPY dev/server/package*.json ./server/

# Install dependencies
RUN cd tests && npm ci
RUN cd viewer && npm ci
RUN cd server && npm ci

# Install Playwright browsers
RUN cd tests && npx playwright install

# Development stage
FROM base as development

# Copy source code
COPY dev/ ./

# Set environment variables
ENV NODE_ENV=development
ENV DISPLAY=:99

# Expose ports
EXPOSE 3000 3001 3002

# Start Xvfb for headless testing
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & npm run dev"]

# Testing stage
FROM base as testing

# Copy source code
COPY dev/ ./

# Set environment variables
ENV NODE_ENV=test
ENV DISPLAY=:99
ENV CI=true

# Create test directories
RUN mkdir -p /app/tests/reports /app/tests/screenshots /app/tests/coverage

# Start Xvfb and run tests
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & npm run test:all"]

# Production build stage
FROM base as builder

# Copy source code
COPY dev/ ./

# Set production environment
ENV NODE_ENV=production

# Build viewer
RUN cd viewer && npm run build

# Build server
RUN cd server && npm run build

# Production stage
FROM node:18-alpine as production

# Install only production dependencies
RUN apk add --no-cache python3 py3-pip

# Create app user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy built applications
COPY --from=builder --chown=nextjs:nodejs /app/viewer/dist ./viewer/dist
COPY --from=builder --chown=nextjs:nodejs /app/server/dist ./server/dist
COPY --from=builder --chown=nextjs:nodejs /app/server/package*.json ./server/

# Install production dependencies only
RUN cd server && npm ci --only=production && npm cache clean --force

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["node", "server/dist/server.js"]
