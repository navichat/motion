#!/bin/bash
"""
Production Deployment Script for Motion Inference API

This script deploys the Python bridge as a production-ready API server
with multiple deployment options.
"""

set -e

echo "🚀 Motion Inference API Deployment"
echo "=================================="

# Configuration
PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-4}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-development}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "No virtual environment detected"
        print_status "Activating virtual environment..."
        source /home/barberb/motion/.venv/bin/activate
    fi
    
    # Check required packages
    python3 -c "import flask, numpy, onnxruntime" 2>/dev/null || {
        print_error "Required packages not installed. Run: pip install -r requirements.txt"
        exit 1
    }
    
    print_success "Dependencies verified"
}

# Validate models
validate_models() {
    print_status "Validating ONNX models..."
    
    models_dir="models/onnx"
    required_models=("deephase.onnx" "stylevae_encoder.onnx" "stylevae_decoder.onnx" "deepmimic_actor.onnx" "deepmimic_critic.onnx")
    
    for model in "${required_models[@]}"; do
        if [[ -f "$models_dir/$model" ]]; then
            print_success "✓ $model found"
        else
            print_error "✗ $model missing"
            exit 1
        fi
    done
    
    print_success "All models validated"
}

# Test API server
test_server() {
    print_status "Testing API server..."
    
    # Start server in background
    python3 deploy_api_server.py &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Test health endpoint
    if curl -s "http://localhost:$PORT/health" > /dev/null; then
        print_success "Server is responding"
        
        # Run validation tests
        print_status "Running validation tests..."
        python3 test_mojo_validation.py
        
    else
        print_error "Server not responding"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    
    # Stop test server
    kill $SERVER_PID 2>/dev/null
    print_success "Test completed successfully"
}

# Deploy development server
deploy_development() {
    print_status "Starting development server..."
    print_status "Server will be available at: http://localhost:$PORT"
    print_status "Press Ctrl+C to stop"
    
    export FLASK_ENV=development
    export FLASK_DEBUG=True
    python3 deploy_api_server.py
}

# Deploy production server with Gunicorn
deploy_production() {
    print_status "Starting production server with Gunicorn..."
    
    # Create gunicorn configuration
    cat > gunicorn.conf.py << EOF
# Gunicorn configuration for Motion Inference API
bind = "$HOST:$PORT"
workers = $WORKERS
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2
user = $(whoami)
group = $(whoami)
tmp_upload_dir = "/tmp"
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
loglevel = "info"
EOF
    
    # Create logs directory
    mkdir -p logs
    
    # Start with gunicorn
    print_status "Server starting at: http://$HOST:$PORT"
    print_status "Workers: $WORKERS"
    print_status "Logs: logs/gunicorn_*.log"
    
    exec gunicorn --config gunicorn.conf.py deploy_api_server:app
}

# Deploy with Docker
deploy_docker() {
    print_status "Building Docker image..."
    
    docker build -t motion-inference-api .
    
    print_status "Starting Docker container..."
    
    docker run -d \
        --name motion-api \
        --restart unless-stopped \
        -p $PORT:5000 \
        -v $(pwd)/models:/app/models:ro \
        -v $(pwd)/logs:/app/logs \
        -e FLASK_ENV=production \
        motion-inference-api
    
    print_success "Docker container started"
    print_status "Container name: motion-api"
    print_status "API available at: http://localhost:$PORT"
    print_status "View logs: docker logs motion-api"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    print_status "Starting with Docker Compose..."
    
    docker-compose up -d motion-api
    
    print_success "Docker Compose deployment started"
    print_status "API available at: http://localhost:$PORT"
    print_status "View logs: docker-compose logs motion-api"
}

# Main deployment logic
main() {
    print_status "Deployment mode: $DEPLOYMENT_MODE"
    
    # Always check dependencies and validate models
    check_dependencies
    validate_models
    
    case "$DEPLOYMENT_MODE" in
        "development"|"dev")
            deploy_development
            ;;
        "production"|"prod")
            deploy_production
            ;;
        "docker")
            deploy_docker
            ;;
        "docker-compose"|"compose")
            deploy_docker_compose
            ;;
        "test")
            test_server
            ;;
        *)
            print_error "Unknown deployment mode: $DEPLOYMENT_MODE"
            echo "Available modes: development, production, docker, docker-compose, test"
            exit 1
            ;;
    esac
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode|-m)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --host|-h)
            HOST="$2"
            shift 2
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --test|-t)
            DEPLOYMENT_MODE="test"
            shift
            ;;
        --help)
            echo "Motion Inference API Deployment Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --mode MODE        Deployment mode (development, production, docker, docker-compose, test)"
            echo "  -p, --port PORT        Port to bind to (default: 5000)"
            echo "  -h, --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  -w, --workers WORKERS  Number of worker processes for production (default: 4)"
            echo "  -t, --test            Run tests only"
            echo "      --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --mode development                    # Start development server"
            echo "  $0 --mode production --workers 8        # Start production server with 8 workers"
            echo "  $0 --mode docker --port 8080            # Deploy with Docker on port 8080"
            echo "  $0 --test                               # Run tests only"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
