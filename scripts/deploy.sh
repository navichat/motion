#!/bin/bash

# Motion Package Deployment Script
# This script deploys the Motion package to different environments

set -e

# Configuration
REGISTRY="ghcr.io"
REPOSITORY="navichat/motion"
ENVIRONMENT=${1:-"staging"}
VERSION=${2:-"latest"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to validate environment
validate_environment() {
    local env=$1
    
    case $env in
        "staging"|"production"|"development")
            return 0
            ;;
        *)
            log_error "Invalid environment: $env"
            log_info "Valid environments: staging, production, development"
            exit 1
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check registry authentication
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "$GITHUB_TOKEN" | docker login $REGISTRY -u $GITHUB_ACTOR --password-stdin
    fi
    
    log_success "Prerequisites check passed"
}

# Function to pull latest image
pull_image() {
    local env=$1
    local version=$2
    
    log_info "Pulling latest image for $env environment..."
    
    local image_tag
    case $env in
        "development")
            image_tag="dev-${version}"
            ;;
        "staging")
            image_tag="test-${version}"
            ;;
        "production")
            image_tag="prod-${version}"
            ;;
    esac
    
    docker pull "${REGISTRY}/${REPOSITORY}:${image_tag}"
    
    if [ $? -eq 0 ]; then
        log_success "Image pulled successfully"
    else
        log_error "Failed to pull image"
        exit 1
    fi
}

# Function to run pre-deployment tests
run_predeploy_tests() {
    local env=$1
    local version=$2
    
    log_info "Running pre-deployment tests for $env..."
    
    # Run smoke tests
    docker run --rm \
        -e NODE_ENV=$env \
        "${REGISTRY}/${REPOSITORY}:prod-${version}" \
        npm --version
    
    if [ $? -eq 0 ]; then
        log_success "Pre-deployment tests passed"
    else
        log_error "Pre-deployment tests failed"
        exit 1
    fi
}

# Function to deploy to staging
deploy_staging() {
    local version=$1
    
    log_info "Deploying to staging environment..."
    
    # Create staging docker-compose override
    cat > docker-compose.staging.yml << EOF
version: '3.8'
services:
  motion-app:
    image: ${REGISTRY}/${REPOSITORY}:test-${version}
    container_name: motion-staging
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=staging
      - ENVIRONMENT=staging
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
networks:
  default:
    external: true
    name: motion-staging
EOF
    
    # Deploy with docker-compose
    docker-compose -f docker-compose.staging.yml up -d
    
    # Wait for service to be healthy
    log_info "Waiting for staging service to be healthy..."
    timeout 120 sh -c 'until docker-compose -f docker-compose.staging.yml ps | grep -q "healthy"; do sleep 5; done'
    
    if [ $? -eq 0 ]; then
        log_success "Staging deployment successful"
        log_info "Staging URL: http://localhost:8080"
    else
        log_error "Staging deployment failed or service is unhealthy"
        docker-compose -f docker-compose.staging.yml logs
        exit 1
    fi
}

# Function to deploy to production
deploy_production() {
    local version=$1
    
    log_info "Deploying to production environment..."
    
    # Additional safety checks for production
    if [ -z "$PRODUCTION_APPROVED" ]; then
        log_warning "Production deployment requires approval"
        read -p "Are you sure you want to deploy to production? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Production deployment cancelled"
            exit 0
        fi
    fi
    
    # Create production docker-compose override
    cat > docker-compose.production.yml << EOF
version: '3.8'
services:
  motion-app:
    image: ${REGISTRY}/${REPOSITORY}:prod-${version}
    container_name: motion-production
    ports:
      - "80:8080"
      - "443:8080"
    environment:
      - NODE_ENV=production
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
networks:
  default:
    external: true
    name: motion-production
EOF
    
    # Deploy with docker-compose
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for service to be healthy
    log_info "Waiting for production service to be healthy..."
    timeout 180 sh -c 'until docker-compose -f docker-compose.production.yml ps | grep -q "healthy"; do sleep 10; done'
    
    if [ $? -eq 0 ]; then
        log_success "Production deployment successful"
        log_info "Production URL: http://localhost"
    else
        log_error "Production deployment failed or service is unhealthy"
        docker-compose -f docker-compose.production.yml logs
        exit 1
    fi
}

# Function to deploy development environment
deploy_development() {
    local version=$1
    
    log_info "Starting development environment..."
    
    # Use the main docker-compose file for development
    docker-compose up -d motion-dev
    
    log_success "Development environment started"
    log_info "Development URL: http://localhost:8080"
}

# Function to run post-deployment tests
run_postdeploy_tests() {
    local env=$1
    local url=$2
    
    log_info "Running post-deployment tests for $env..."
    
    # Wait for service to be available
    sleep 10
    
    # Basic health check
    if command -v curl &> /dev/null; then
        curl -f "$url/health" || {
            log_warning "Health check endpoint not responding"
        }
    fi
    
    log_success "Post-deployment tests completed"
}

# Function to cleanup old deployments
cleanup_old_deployments() {
    local env=$1
    
    log_info "Cleaning up old deployments for $env..."
    
    # Remove old containers
    docker container prune -f
    
    # Remove old images (keep last 3 versions)
    docker images "${REGISTRY}/${REPOSITORY}" --format "table {{.Tag}}\t{{.CreatedAt}}" | \
        tail -n +4 | \
        awk '{print $1}' | \
        xargs -r docker rmi "${REGISTRY}/${REPOSITORY}:" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    local env=$1
    local version=$2
    
    log_info "Starting deployment to $env environment (version: $version)..."
    
    # Validate inputs
    validate_environment "$env"
    
    # Check prerequisites
    check_prerequisites
    
    # Pull latest image
    pull_image "$env" "$version"
    
    # Run pre-deployment tests
    run_predeploy_tests "$env" "$version"
    
    # Deploy based on environment
    case $env in
        "development")
            deploy_development "$version"
            run_postdeploy_tests "$env" "http://localhost:8080"
            ;;
        "staging")
            deploy_staging "$version"
            run_postdeploy_tests "$env" "http://localhost:8080"
            ;;
        "production")
            deploy_production "$version"
            run_postdeploy_tests "$env" "http://localhost"
            ;;
    esac
    
    # Cleanup old deployments
    cleanup_old_deployments "$env"
    
    log_success "Deployment to $env completed successfully!"
}

# Help function
show_help() {
    cat << EOF
Motion Package Deployment Script

USAGE:
    $0 [ENVIRONMENT] [VERSION] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT Environment to deploy to (default: staging)
                Options: development, staging, production
    VERSION     Version to deploy (default: latest)

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN         GitHub token for registry authentication
    GITHUB_ACTOR         GitHub username for registry authentication
    PRODUCTION_APPROVED  Set to 'true' to skip production confirmation

EXAMPLES:
    $0                        # Deploy latest to staging
    $0 production v1.2.3      # Deploy v1.2.3 to production
    $0 development            # Start development environment

EOF
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"