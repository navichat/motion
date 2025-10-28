#!/bin/bash

# Motion Package Docker Build Script
# This script builds Docker images for different environments

set -e

# Configuration
REGISTRY="ghcr.io"
REPOSITORY="navichat/motion"
VERSION=${1:-"latest"}
PLATFORM=${PLATFORM:-"linux/amd64,linux/arm64"}

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

# Function to build and tag image
build_image() {
    local target=$1
    local tag=$2
    local description=$3
    
    log_info "Building $description image..."
    
    docker buildx build \
        --platform $PLATFORM \
        --target $target \
        --tag "${REGISTRY}/${REPOSITORY}:${tag}" \
        --tag "${REGISTRY}/${REPOSITORY}:${target}-${VERSION}" \
        --cache-from type=gha \
        --cache-to type=gha,mode=max \
        --push \
        .
        
    if [ $? -eq 0 ]; then
        log_success "$description image built successfully"
    else
        log_error "Failed to build $description image"
        exit 1
    fi
}

# Function to test image
test_image() {
    local tag=$1
    local description=$2
    
    log_info "Testing $description image..."
    
    # Run a quick test to ensure the image works
    docker run --rm \
        -e CI=true \
        "${REGISTRY}/${REPOSITORY}:${tag}" \
        npm --version
        
    if [ $? -eq 0 ]; then
        log_success "$description image test passed"
    else
        log_error "$description image test failed"
        exit 1
    fi
}

# Main build process
main() {
    log_info "Starting Motion Package Docker build process..."
    
    # Check if Docker buildx is available
    if ! docker buildx version > /dev/null 2>&1; then
        log_error "Docker buildx is not available. Please install it first."
        exit 1
    fi
    
    # Create buildx builder if it doesn't exist
    if ! docker buildx ls | grep -q motion-builder; then
        log_info "Creating Docker buildx builder..."
        docker buildx create --name motion-builder --use
    else
        docker buildx use motion-builder
    fi
    
    # Log in to registry if credentials are provided
    if [ -n "$GITHUB_TOKEN" ]; then
        log_info "Logging in to GitHub Container Registry..."
        echo "$GITHUB_TOKEN" | docker login $REGISTRY -u $GITHUB_ACTOR --password-stdin
    fi
    
    # Build different target images
    build_image "development" "dev-${VERSION}" "Development"
    build_image "testing" "test-${VERSION}" "Testing"
    build_image "ci" "ci-${VERSION}" "CI"
    build_image "production" "prod-${VERSION}" "Production"
    
    # Tag latest if building main branch
    if [ "$VERSION" = "latest" ] || [ "$GITHUB_REF" = "refs/heads/main" ]; then
        docker buildx build \
            --platform $PLATFORM \
            --target production \
            --tag "${REGISTRY}/${REPOSITORY}:latest" \
            --cache-from type=gha \
            --push \
            .
        log_success "Latest tag created"
    fi
    
    # Test images (only for single platform builds in CI)
    if [ "$CI" = "true" ] && [[ "$PLATFORM" != *","* ]]; then
        test_image "test-${VERSION}" "Testing"
        test_image "prod-${VERSION}" "Production"
    fi
    
    log_success "All images built successfully!"
    
    # Output image information
    echo ""
    log_info "Built images:"
    echo "  - ${REGISTRY}/${REPOSITORY}:dev-${VERSION}"
    echo "  - ${REGISTRY}/${REPOSITORY}:test-${VERSION}"
    echo "  - ${REGISTRY}/${REPOSITORY}:ci-${VERSION}"
    echo "  - ${REGISTRY}/${REPOSITORY}:prod-${VERSION}"
    if [ "$VERSION" = "latest" ] || [ "$GITHUB_REF" = "refs/heads/main" ]; then
        echo "  - ${REGISTRY}/${REPOSITORY}:latest"
    fi
}

# Help function
show_help() {
    cat << EOF
Motion Package Docker Build Script

USAGE:
    $0 [VERSION] [OPTIONS]

ARGUMENTS:
    VERSION     Version tag for the images (default: latest)

ENVIRONMENT VARIABLES:
    PLATFORM        Target platforms (default: linux/amd64,linux/arm64)
    GITHUB_TOKEN    GitHub token for registry authentication
    GITHUB_ACTOR    GitHub username for registry authentication
    CI              Set to 'true' when running in CI environment

EXAMPLES:
    $0                      # Build with 'latest' tag
    $0 v1.2.3              # Build with specific version
    $0 dev                 # Build development version

    # CI environment
    PLATFORM=linux/amd64 $0 ci-build

EOF
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"