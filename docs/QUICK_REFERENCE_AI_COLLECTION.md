# 🚀 Quick Reference: Avatar AI Model Inference Collection

## One-Command Test Execution

```bash
# Complete AI model capture (recommended)
cd /home/barberb/motion/dev/web_viewer && python3 serve_with_headers.py 8081 &
cd /home/barberb/motion && npx playwright test capture-ai-results.spec.js --project=chromium-webgpu
```

## 📊 Expected Results Summary

### ✅ Success Indicators
- **Total completed tasks**: 40+ tasks
- **Unique job types**: 16+ types  
- **Test duration**: 3-5 minutes
- **Success rate**: 95%+
- **JSON files**: Auto-generated in `ai-inference-results/`

### 🎯 19+ AI Model Types Expected

| Category | Models | Count |
|----------|--------|-------|
| **Language** | TinyLlama, DiabloGPT | 2 |
| **Audio** | Whisper, VAD, Kokoro, SpeechT5 | 4 |
| **Motion** | RSMT, DeepMimic, FaceFormer, Audio2Gesture | 4 |
| **Compute** | WASMMatrix, WASMPrime, WASMFractal, WebGPU* | 6 |
| **KNN** | CloseVector, HNSW, UnifiedKNN | 3 |
| **Total** | | **19+** |

## 🔍 Result Inspection Commands

```bash
# View captured results
ls -la ai-inference-results/

# Check job type counts
cat ai-inference-results/job-summary-*.json | jq '.jobTypeCounts'

# View execution details
cat ai-inference-results/complete-ai-results-*.json | jq '.completedTasks[0]'

# Count unique model types
cat ai-inference-results/job-summary-*.json | jq '.uniqueJobTypes'
```

## ⚡ Alternative Test Methods

### Interactive Web Interface
```bash
open http://localhost:8081/task-manager-demo.html
# Click: "🚀 Real WASM/GPU/WebNN Workload"
```

### Extended E2E Test (20 minutes)
```bash
npx playwright test dev/web_viewer/e2e-workload-test.spec.js --project=chromium-webgpu --timeout=1200000
```

## 🛠️ Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| Port 8081 in use | `pkill -f serve_with_headers && sleep 2` |
| No results captured | Check TaskManager state in browser console |
| ONNX errors | Verify conservative session options are applied |
| Missing models | Ensure all script dependencies loaded |

### Success Validation
```bash
# Check if test passed
echo "Expected: 16+ job types, got: $(cat ai-inference-results/job-summary-*.json | jq '.uniqueJobTypes')"

# Verify no errors
grep -i error ai-inference-results/complete-ai-results-*.json || echo "No errors found"
```

## 📁 Output Files Structure

```
ai-inference-results/
├── complete-ai-results-TIMESTAMP.json    # Full task details
└── job-summary-TIMESTAMP.json            # Summary statistics
```

## 📚 Full Documentation
- **Complete Guide**: [docs/AI_INFERENCE_COLLECTION.md](./AI_INFERENCE_COLLECTION.md)
- **Main README**: [docs/README.md](./README.md)

---

**Status**: ✅ Production Ready - Captures 19+ AI model types with automated JSON export
