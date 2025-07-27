# Avatar AI Inference Collection System

A comprehensive system for collecting, testing, and exporting AI model inference results for avatar driving applications. This system supports 13 different AI models across language processing, audio synthesis, motion generation, and computational tasks.

## 🤖 System Overview

The Avatar AI System provides:
- **Comprehensive AI Model Support**: 13 different AI models for complete avatar functionality
- **Real-time Inference Collection**: Live collection of AI model outputs during task execution
- **Multi-format Export**: Export results to production-ready file formats (BVH, WAV, JSON, etc.)
- **Playwright Testing**: Automated testing with comprehensive data collection
- **Production Integration**: Ready for use in avatar driving applications

## 🎭 Supported AI Models

### Language Models (Avatar Conversation & Reasoning)
- **TinyLlama**: Lightweight language model for avatar conversation
- **DiabloGPT**: Personality-driven text generation for avatar character

### Audio Processing (Avatar Voice & Listening)
- **Whisper**: Speech recognition for avatar input processing
- **VAD (Voice Activity Detection)**: Voice activity detection for avatar listening
- **Kokoro**: Text-to-speech synthesis for avatar voice output
- **SpeechT5**: Advanced voice synthesis for natural avatar speech

### Motion Models (Avatar Animation & Movement)
- **RSMT**: Real-time stylized motion transitions for avatar movement
- **DeepMimic**: Physics-based motion learning for realistic avatar animation
- **FaceFormer**: Facial animation generation for avatar expressions
- **Audio2Gesture**: Audio-driven gesture generation for avatar body language

### Computational Models (Avatar Physics & Algorithms)
- **WASMMatrix**: Matrix computations for avatar physics calculations
- **WASMPrime**: Mathematical computations for avatar algorithms
- **WASMFractal**: Visual generation for avatar environments and effects

## 🚀 Quick Start

### 1. Start the Development Server
```bash
cd /home/barberb/motion/dev/web_viewer
python3 -m http.server 8000
```

### 2. Run Inference Collection Test
```bash
# Basic collection test
npx playwright test e2e-workload-test.spec.js --headed

# Full data export test
npx playwright test e2e-avatar-data-export.spec.js --headed
```

### 3. View Results
- **Live Demo**: `http://localhost:8000/task-manager-demo.html`
- **Export Directory**: `avatar-data-exports/export-TIMESTAMP/`

## 📊 Test Specifications

### Avatar AI Inference Collection Test (`e2e-workload-test.spec.js`)
**Purpose**: Comprehensive collection and analysis of all 13 AI model inference results

**Features**:
- 4-minute timeout for complete model collection
- Real-time progress monitoring with 15-second status updates
- Detailed model-by-model result tracking
- Avatar readiness assessment
- Comprehensive logging and debugging

**Expected Results**:
- Language Models: TinyLlama + DiabloGPT outputs
- Audio Processing: Whisper + VAD + Kokoro + SpeechT5 outputs  
- Motion Models: RSMT + DeepMimic + FaceFormer + Audio2Gesture outputs
- Compute Models: Matrix + Prime + Fractal outputs

### Avatar Data Export Test (`e2e-avatar-data-export.spec.js`)
**Purpose**: Export AI inference results to production-ready file formats

**Features**:
- 5-minute timeout for collection and export
- Multiple file format generation
- Realistic data synthesis for missing models
- Comprehensive file organization
- Export summary and metadata generation

**Output Formats**:
- **Text Files (.txt)**: Language model outputs, prime numbers
- **Audio Files (.wav)**: Synthesized speech and TTS outputs
- **Motion Files (.bvh)**: Motion capture data for 3D animation
- **Data Files (.csv)**: Matrix computations and tabular data
- **Image Files (.png)**: Fractal visualizations and graphics
- **JSON Files (.json)**: Structured metadata and configurations

## 📁 Export Directory Structure

```
avatar-data-exports/export-TIMESTAMP/
├── EXPORT_SUMMARY.txt                    # Comprehensive collection summary
├── export_metadata.json                  # Collection statistics and metadata
│
├── Language Models/
│   ├── tinyllama_1_taskId.txt            # Conversation text outputs
│   └── diablogpt_1_taskId.txt            # Personality responses
│
├── Audio Processing/
│   ├── whisper_transcript_1_taskId.json  # Speech recognition data
│   ├── whisper_transcript_1_taskId.txt   # Plain text transcripts
│   ├── vad_detection_1_taskId.json       # Voice activity detection
│   ├── kokoro_tts_1_taskId.wav           # Text-to-speech audio
│   ├── kokoro_metadata_1_taskId.json     # TTS metadata
│   ├── speecht5_synthesis_1_taskId.wav   # Voice synthesis audio
│   └── speecht5_metadata_1_taskId.json   # Synthesis metadata
│
├── Motion Models/
│   ├── rsmt_motion_1_taskId.bvh          # Motion transition data
│   ├── rsmt_metadata_1_taskId.json       # Motion metadata
│   ├── deepmimic_motion_1_taskId.bvh     # Physics-based motion
│   ├── deepmimic_metadata_1_taskId.json  # Physics metadata
│   ├── faceformer_keypoints_1_taskId.json # Facial keypoints (68 points)
│   ├── faceformer_blendshapes_1_taskId.txt # Facial blend shapes
│   ├── audio2gesture_1_taskId.bvh        # Audio-driven gestures
│   └── audio2gesture_metadata_1_taskId.json # Gesture metadata
│
└── Compute Models/
    ├── matrix_data_1_taskId.csv          # Matrix computations
    ├── matrix_metadata_1_taskId.json     # Matrix metadata
    ├── prime_numbers_1_taskId.txt        # Prime number sequences
    ├── prime_metadata_1_taskId.json      # Mathematical metadata
    ├── fractal_image_1_taskId.png        # Fractal visualizations
    └── fractal_metadata_1_taskId.json    # Image generation metadata
```

## 🎯 File Format Details

### BVH Motion Files
- **Standard BVH format** with proper skeleton hierarchy
- **120 frames** at 30fps (4 seconds of motion)
- **Compatible** with Blender, Maya, MotionBuilder
- **Realistic joint rotations** based on model type

### WAV Audio Files  
- **Standard WAV format** with proper headers
- **44.1kHz sample rate**, 16-bit mono
- **2-3 second duration** per synthesis
- **Compatible** with all audio software

### JSON Metadata Files
- **Structured data** with execution metrics
- **Model parameters** and inference details
- **Timestamp and task tracking** information
- **Compatible** with any JSON processor

### CSV Data Files
- **Standard CSV format** with headers
- **10x10 matrix data** for computational results
- **Compatible** with Excel, Google Sheets, data analysis tools

### PNG Image Files
- **Standard PNG format** with proper headers
- **64x64 pixel fractal visualizations**
- **Gradient-based color schemes**
- **Compatible** with all image viewers

## 🔧 Technical Architecture

### Worker System
- **CPU Workers**: Handle WASM computations and basic AI tasks
- **GPU Workers**: Execute WebGPU-accelerated AI models
- **WebNN Workers**: Run ONNX models with hardware acceleration
- **WASM Workers**: Perform mathematical and computational tasks

### Task Management
- **Priority Queue**: Efficient task scheduling and execution
- **Resource Management**: Worker allocation and load balancing
- **Progress Tracking**: Real-time execution monitoring
- **Error Handling**: Comprehensive error recovery and reporting

### Data Collection
- **Real-time Parsing**: Live extraction of AI model outputs
- **Format Detection**: Automatic identification of output types
- **Synthetic Generation**: Fallback data creation for missing outputs
- **Quality Assurance**: Validation and verification of collected data

## 🎨 Integration with Avatar Systems

### 3D Animation Software
```javascript
// Import BVH files into Blender/Maya
const bvhData = fs.readFileSync('rsmt_motion_1_taskId.bvh', 'utf8');
// Standard BVH format - direct import supported
```

### Audio Processing
```javascript
// Load WAV files for avatar speech
const audioContext = new AudioContext();
const audioBuffer = await audioContext.decodeAudioData(wavData);
// Standard WAV format - direct playback supported
```

### Real-time Systems
```javascript
// Use JSON metadata for parameter configuration
const metadata = JSON.parse(fs.readFileSync('export_metadata.json', 'utf8'));
const executionTime = metadata.collectionDuration;
const totalResults = metadata.totalFiles;
```

## 🧪 Testing and Validation

### Comprehensive Model Testing
```bash
# Test all models with detailed logging
npx playwright test e2e-workload-test.spec.js --headed --reporter=html

# Export test with file generation
npx playwright test e2e-avatar-data-export.spec.js --headed --reporter=html
```

### Individual Model Testing
```bash
# Test specific model categories
AVATAR_MODELS="language,audio" npx playwright test --grep "Avatar AI"
AVATAR_MODELS="motion,compute" npx playwright test --grep "Avatar AI"
```

### Performance Benchmarking
```bash
# Run multiple iterations for performance analysis
for i in {1..5}; do
  npx playwright test e2e-workload-test.spec.js --reporter=json > results_$i.json
done
```

## 📈 Results Analysis

### Model Performance Metrics
- **Execution Time**: Per-model inference duration
- **Success Rate**: Model completion percentage
- **Output Quality**: Data validation metrics
- **Resource Usage**: CPU/GPU/Memory utilization

### Avatar Readiness Assessment
- **Language Processing**: ✅/❌ Conversation capability
- **Audio Processing**: ✅/❌ Voice synthesis/recognition
- **Motion Processing**: ✅/❌ Animation generation
- **Compute Processing**: ✅/❌ Physics calculations

### Export Statistics
- **Total Files Created**: Count of generated files
- **Format Distribution**: Breakdown by file type
- **Data Volume**: Total size of exported data
- **Collection Duration**: Time to gather all results

## 🛠️ Configuration Options

### Environment Variables
```bash
# Set specific models to test
export AVATAR_MODELS="language,audio,motion,compute"

# Configure collection timeout
export COLLECTION_TIMEOUT=300000  # 5 minutes

# Set export directory
export EXPORT_DIR="custom-exports"
```

### Test Configuration
```javascript
// In test files
test.setTimeout(300000); // 5 minutes for comprehensive testing
const maxRetries = 3;    // Retry failed model executions
const exportFormats = ['bvh', 'wav', 'json', 'csv', 'png', 'txt'];
```

## 🔍 Troubleshooting

### Common Issues

**SpeechT5 Not Working**:
- ✅ **Fixed**: SpeechT5 now fully integrated across all worker types
- Check worker implementations in `webnn-worker-simple.js`
- Verify model paths in `model-loader-webnn.js`

**WebNN/WebGPU Not Available**:
- System falls back to CPU simulation automatically
- Advanced models may show as "not available" - this is normal
- All 13 models can run with CPU fallback

**Export Directory Permissions**:
```bash
chmod 755 avatar-data-exports/
chown -R $USER:$USER avatar-data-exports/
```

**Browser Compatibility**:
- Chrome 90+ recommended for WebGPU support
- Firefox 85+ for WASM support
- Safari 14+ for basic functionality

### Debug Logging
```javascript
// Enable detailed logging in tests
console.log('🤖 Collected results:', avatarInferenceResults.metadata.totalResults);
console.log('📊 Model breakdown:', {
  language: totalLanguageResults,
  audio: totalAudioResults,
  motion: totalMotionResults,
  compute: totalComputeResults
});
```

## 🚀 Future Enhancements

### Planned Features
- [ ] Real-time streaming export during collection
- [ ] WebRTC integration for remote avatar control
- [ ] Custom model plugin system
- [ ] Advanced motion blending algorithms
- [ ] GPU-accelerated export processing

### Integration Roadmap
- [ ] Unity3D avatar system integration
- [ ] Unreal Engine motion pipeline
- [ ] VRChat avatar compatibility
- [ ] WebXR immersive experiences
- [ ] Mobile avatar applications

## 📝 License and Usage

This system is part of the Motion Workspace project and follows the same licensing terms. See individual model licenses for specific usage restrictions.

For commercial avatar applications, ensure compliance with:
- Model-specific licensing (TinyLlama, Whisper, etc.)
- Audio synthesis usage rights
- Motion capture data permissions
- Export format compatibility requirements

---

**Last Updated**: July 27, 2025  
**Version**: 2.0.0  
**Compatibility**: Chrome 90+, Firefox 85+, Node.js 16+
