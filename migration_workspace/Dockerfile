FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY deploy_api_server.py .
COPY test_mojo_validation.py .

# Create non-root user for security
RUN useradd -m -u 1000 motionapi && chown -R motionapi:motionapi /app
USER motionapi

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV HOST=0.0.0.0
ENV PORT=5000

# Start command
CMD ["python", "deploy_api_server.py"]
