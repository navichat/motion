#!/usr/bin/env bash
set -euo pipefail

echo "Running serverless component tests (shell timeout)"
NO_WEBSERVER=1 timeout 600s npx playwright test --project=component-tests

echo "Running web_viewer smoke (shell timeout)"
timeout 600s npx playwright test --project=web_viewer-root-e2e

echo "All checks passed."
