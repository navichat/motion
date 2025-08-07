#!/bin/bash

# Enhanced AI Inference Results Collection Script
echo "🤖 Starting Enhanced Avatar AI Inference Collection..."

# Create results directory
mkdir -p ./ai-inference-results

# Run the enhanced test with full output capture
echo "⚙️ Running comprehensive AI model inference test..."
timeout 400s npx playwright test --project=chromium-webgpu dev/web_viewer/e2e-workload-test.spec.js > ./ai-inference-results/test-output.log 2>&1

# Check if results files were created
echo "📁 Checking for generated result files..."
ls -la ./ai-inference-results/

echo "📊 Result files created:"
find ./ai-inference-results/ -name "*.json" | while read file; do
    echo "   📄 $file ($(wc -l < "$file") lines)"
done

echo "✅ Results collection complete. Check the ai-inference-results directory for:"
echo "   - test-output.log: Complete test output"
echo "   - avatar-ai-results-*.json: Complete inference results" 
echo "   - avatar-ai-summary-*.json: Summary of model counts"

echo ""
echo "🔍 Quick summary from latest results file:"
latest_summary=$(ls -t ./ai-inference-results/avatar-ai-summary-*.json 2>/dev/null | head -1)
if [ -f "$latest_summary" ]; then
    echo "📋 Model counts from $latest_summary:"
    cat "$latest_summary" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
print(f'Total Results: {data[\"totalResults\"]}')
print(f'Execution Time: {data[\"executionTime\"]:.1f}s')
print()
for category, models in data['modelCounts'].items():
    print(f'{category}:')
    for model, count in models.items():
        print(f'  {model}: {count}')
    print()
"
else
    echo "❌ No summary files found. Check test-output.log for details."
fi
