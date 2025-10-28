#!/bin/bash

# Motion Package Docker Test Runner
# This script runs tests in Docker containers

set -e

# Configuration
REGISTRY="ghcr.io"
REPOSITORY="navichat/motion"
VERSION=${VERSION:-"latest"}
TEST_SUITE=${1:-"all"}

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

# Function to run tests in Docker
run_test_suite() {
    local suite=$1
    local description=$2
    local image_tag=${3:-"test-${VERSION}"}
    
    log_info "Running $description tests..."
    
    # Create directories for test outputs
    mkdir -p test-results playwright-report ai-inference-results coverage
    
    # Run the test suite
    docker run --rm \
        -e CI=true \
        -e NODE_ENV=test \
        -e DISPLAY=:99 \
        -e PYTHONPATH=/app \
        -v "$(pwd)/test-results:/app/test-results" \
        -v "$(pwd)/playwright-report:/app/playwright-report" \
        -v "$(pwd)/ai-inference-results:/app/ai-inference-results" \
        -v "$(pwd)/coverage:/app/coverage" \
        "${REGISTRY}/${REPOSITORY}:${image_tag}" \
        sh -c "Xvfb :99 -screen 0 1920x1080x24 & sleep 2 && npm run test:${suite}"
        
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "$description tests passed"
    else
        log_error "$description tests failed (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to run tests with docker-compose
run_compose_tests() {
    local profile=$1
    local description=$2
    
    log_info "Running $description tests with docker-compose..."
    
    docker-compose -f docker-compose.ci.yml --profile $profile up --build --abort-on-container-exit
    local exit_code=$?
    
    # Clean up
    docker-compose -f docker-compose.ci.yml --profile $profile down
    
    if [ $exit_code -eq 0 ]; then
        log_success "$description tests passed"
    else
        log_error "$description tests failed"
        return $exit_code
    fi
}

# Function to run parallel tests
run_parallel_tests() {
    log_info "Running parallel test suites..."
    
    # Define test suites to run in parallel
    local test_suites=(
        "unit:Unit Tests"
        "unit:animation:Animation Tests" 
        "ml:serverless:ML Serverless Tests"
        "taskmanager:Task Manager Tests"
    )
    
    local pids=()
    local failed_tests=()
    
    # Start all test suites in background
    for suite_info in "${test_suites[@]}"; do
        IFS=':' read -r suite description <<< "$suite_info"
        (
            run_test_suite "$suite" "$description"
            echo $? > "test_result_${suite//[:\/]/_}.tmp"
        ) &
        pids+=($!)
    done
    
    # Wait for all tests to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    # Check results
    local overall_result=0
    for suite_info in "${test_suites[@]}"; do
        IFS=':' read -r suite description <<< "$suite_info"
        result_file="test_result_${suite//[:\/]/_}.tmp"
        if [ -f "$result_file" ]; then
            result=$(cat "$result_file")
            rm "$result_file"
            if [ "$result" -ne 0 ]; then
                failed_tests+=("$description")
                overall_result=1
            fi
        fi
    done
    
    if [ $overall_result -eq 0 ]; then
        log_success "All parallel tests passed"
    else
        log_error "Some tests failed: ${failed_tests[*]}"
    fi
    
    return $overall_result
}

# Function to generate test report
generate_report() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local report_file="test-report-${timestamp}.md"
    
    log_info "Generating test report: $report_file"
    
    cat > "$report_file" << EOF
# Motion Package Test Report

**Generated:** $timestamp  
**Version:** $VERSION  
**Test Suite:** $TEST_SUITE  

## Test Results

EOF

    # Add test results if available
    if [ -d "test-results" ]; then
        echo "### Test Output Files" >> "$report_file"
        find test-results -name "*.xml" -o -name "*.json" | while read -r file; do
            echo "- \`$file\`" >> "$report_file"
        done
        echo "" >> "$report_file"
    fi
    
    if [ -d "playwright-report" ]; then
        echo "### Playwright Report" >> "$report_file"
        echo "Browser test results available in \`playwright-report/\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    if [ -d "coverage" ]; then
        echo "### Coverage Report" >> "$report_file"
        echo "Code coverage results available in \`coverage/\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    log_success "Test report generated: $report_file"
}

# Main test execution
main() {
    log_info "Starting Motion Package test execution..."
    
    case $TEST_SUITE in
        "all")
            log_info "Running all test suites..."
            run_parallel_tests
            ;;
        "unit")
            run_test_suite "unit" "Unit"
            ;;
        "animation")
            run_test_suite "unit:animation" "Animation"
            ;;
        "web")
            run_test_suite "unit:web" "Web"
            ;;
        "ml")
            run_test_suite "ml:serverless" "ML Serverless"
            ;;
        "e2e")
            run_test_suite "e2e:root-smokes" "E2E"
            ;;
        "perf")
            run_test_suite "perf" "Performance"
            ;;
        "taskmanager")
            run_test_suite "taskmanager" "Task Manager"
            ;;
        "compose")
            log_info "Running tests with docker-compose..."
            docker-compose -f docker-compose.ci.yml up --build --abort-on-container-exit
            docker-compose -f docker-compose.ci.yml down
            ;;
        *)
            log_error "Unknown test suite: $TEST_SUITE"
            echo "Available test suites: all, unit, animation, web, ml, e2e, perf, taskmanager, compose"
            exit 1
            ;;
    esac
    
    local exit_code=$?
    
    # Generate test report
    generate_report
    
    if [ $exit_code -eq 0 ]; then
        log_success "All tests completed successfully!"
    else
        log_error "Some tests failed. Check the logs above for details."
    fi
    
    exit $exit_code
}

# Help function
show_help() {
    cat << EOF
Motion Package Docker Test Runner

USAGE:
    $0 [TEST_SUITE] [OPTIONS]

ARGUMENTS:
    TEST_SUITE  Test suite to run (default: all)
                Options: all, unit, animation, web, ml, e2e, perf, taskmanager, compose

ENVIRONMENT VARIABLES:
    VERSION     Docker image version to use (default: latest)

EXAMPLES:
    $0                  # Run all tests
    $0 unit            # Run only unit tests
    $0 ml              # Run only ML tests
    $0 compose         # Run tests using docker-compose

EOF
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"