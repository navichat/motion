# WebNN/WebGPU/WASM Avatar System

This directory contains a comprehensive web-based avatar system supporting WebNN, WebGPU, and WASM technologies. The codebase has been completely reorganized for systematic testing and development of individual components.

## 🗂️ Organized Project Structure

**RECENTLY REORGANIZED**: All files have been systematically organized into a clean component-based structure:

```
src/
├── models/motion/              # Motion model implementations
│   ├── audio2gesture/          # Audio to gesture conversion
│   ├── rsmt/                   # Realtime stylized motion transition  
│   ├── deepmimic/              # DeepMimic humanoid animations
│   └── faceformer/             # Facial animation from audio
├── components/animation/       # Animation components
│   ├── vrm/                    # VRM avatar system (25 files)
│   └── timeline/               # Animation timeline system
├── testing/                    # All test files consolidated
│   ├── demos/                  # Demo applications
│   ├── e2e/                    # End-to-end tests
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
└── utils/                      # Utilities and tools
    ├── kokoro.js/              # Audio processing
    └── workers/                # Web workers
```

## 🤖 Avatar AI Inference System

Complete AI inference collection system for avatar applications with multi-modal AI processing:

### 13 Supported AI Models:
- **Language Models**: TinyLlama, DiabloGPT for conversation and personality
- **Audio Processing**: Whisper, VAD, Kokoro, **SpeechT5** for speech recognition and synthesis  
- **Motion Models**: RSMT, DeepMimic, FaceFormer, Audio2Gesture for animation generation
- **Compute Models**: WASMMatrix, WASMPrime, WASMFractal for physics and visual effects

### Key Features:
- **Individual Component Testing**: Each motion model can be tested separately
- **Systematic Organization**: Clean separation between models, components, and testing
- **Comprehensive Testing**: Automated Playwright tests with 4-5 minute collection cycles
- **Multi-format Export**: Export to BVH, WAV, JSON, CSV, PNG, TXT formats (132+ files per test)
- **Production Ready**: Real-time inference with export to avatar-compatible file formats
- **Fixed SpeechT5**: Advanced voice synthesis now fully functional across all worker types

📚 **[Complete Avatar AI Documentation](AVATAR_AI_SYSTEM.md)**

### Quick Start Avatar AI:
```bash
# Navigate to the organized structure
cd dev/web_viewer

# Start development server
python3 -m http.server 8000

# Run comprehensive AI inference collection
npx playwright test src/testing/e2e/e2e-workload-test.spec.js --headed

# Run data export test (creates 132+ files in avatar-data-exports/)
npx playwright test src/testing/e2e/e2e-avatar-data-export.spec.js --headed

# View live demo
open http://localhost:8000/index.html
```

## 🎯 Individual Component Testing

**NEW CAPABILITY**: With the reorganized structure, you can now test each component individually:

### Test Motion Models Separately:
```bash
# Test Audio2Gesture model
open src/models/motion/audio2gesture/

# Test RSMT transitions  
open src/models/motion/rsmt/

# Test DeepMimic animations
open src/models/motion/deepmimic/

# Test FaceFormer facial animation
open src/models/motion/faceformer/
```

### Test VRM Avatar Components:
```bash
# Test VRM avatar system (25 files)
open src/components/animation/vrm/

# Test timeline system
open src/components/animation/timeline/
```

### Run Organized Test Suites:
```bash
# Run specific test categories
npx playwright test src/testing/unit/
npx playwright test src/testing/integration/
npx playwright test src/testing/performance/
npx playwright test src/testing/demos/
```

## ⚠️ Important Note About Transitions

**The current showcase demonstrates BVH motion playback and basic interpolation, NOT the actual RSMT neural network system described in the paper.**

### What This Showcase Actually Does:
- ✅ Visualizes authentic 100STYLE dataset animations
- ✅ Shows different motion styles (emotional, character, energy)
- ✅ Provides interactive 3D skeleton visualization
- ⚠️ Uses **basic linear interpolation** for "transitions" (NOT real RSMT)

### What Real RSMT Does:
The actual RSMT system uses sophisticated neural networks:
- **DeepPhase**: Encodes temporal motion patterns on a 2D phase manifold
- **StyleVAE**: Encodes motion style in latent space  
- **TransitionNet**: Generates natural transitions using learned representations

For details, see: [RSMT_TRANSITION_EXPLANATION.md](RSMT_TRANSITION_EXPLANATION.md)

## Available Viewers

### 1. RSMT Showcase (`rsmt_showcase.html`)
**Primary demonstration viewer with style interpolation**
- 9 professional animations from 100STYLE dataset
- Interactive controls for different motion styles
- Basic style blending (simplified transitions)
- Real-time 3D skeleton visualization

**Features:**
- Emotional styles: Neutral, Elated, Angry, Depressed
- Character styles: Proud, Robot, Elderly
- Energy levels: Rushed, Strutting
- Preset transition sequences
- Manual animation selection

### 2. Motion Viewer (`motion_viewer.html`) 
**Original working motion viewer**
- Single animation playback
- Proven BVH parsing and skeleton rendering
- Reference implementation for 3D visualization

### 3. Index Page (`index.html`)
**Landing page with viewer descriptions and navigation**

## Quick Start

1. **Start HTTP Server:**
   ```bash
   cd /path/to/web_viewer
   python3 -m http.server 8080
   ```

2. **Open in Browser:**
   - Main showcase: `http://localhost:8080/rsmt_showcase.html`
   - Index page: `http://localhost:8080/index.html`

3. **Try the Demo:**
   - Click "Preload All Animations" first
   - Try different animation styles
   - Use preset sequences for style progression

## Animation Catalog

| Animation | Duration | Style | Description |
|-----------|----------|-------|-------------|
| `neutral_reference.bvh` | 92s | Emotional | Natural walking baseline |
| `elated_reference.bvh` | 62s | Emotional | Happy, bouncy movement |
| `angry_reference.bvh` | 58s | Emotional | Aggressive, tense motion |
| `depressed_reference.bvh` | 113s | Emotional | Slow, heavy movements |
| `proud_reference.bvh` | 77s | Character | Confident, upright posture |
| `robot_reference.bvh` | 189s | Character | Mechanical, precise movements |
| `old_reference.bvh` | 187s | Character | Careful, deliberate motion |
| `rushed_reference.bvh` | 43s | Energy | Fast, hurried movements |
| `strutting_reference.bvh` | 105s | Energy | Confident, rhythmic walk |

All animations use forward walking (FW) variants from the 100STYLE dataset for natural motion flow.

## Technical Implementation

### BVH Processing
- **Parser**: Custom JavaScript BVH parser with error handling
- **Skeleton**: Fixed 72-channel mapping to match 100STYLE format
- **Rendering**: Three.js-based 3D visualization
- **Animation**: Real-time frame interpolation and playback

### Style Interpolation (Current)
```javascript
// NOTE: This is simplified interpolation, NOT real RSMT!
function createTransitionFrames(fromBvh, toBvh) {
    // Linear interpolation between frame data
    // Creates "dragging" motion between positions
    // Missing: phase encoding, style encoding, neural networks
}
```

### What's Missing for Real RSMT
1. **Trained Neural Models**: DeepPhase, StyleVAE, TransitionNet
2. **Phase Processing**: Temporal pattern encoding
3. **Style Encoding**: Latent space representations
4. **Physics Constraints**: Foot contact preservation
5. **Real-time Inference**: GPU-accelerated neural network execution

## Files Structure

```
dev/web_viewer/
├── src/                           # 🗂️ ORGANIZED SOURCE CODE
│   ├── models/motion/             # Motion model implementations
│   │   ├── audio2gesture/         # Neural audio to gesture conversion
│   │   │   ├── Audio2GestureBVHConverter.js
│   │   │   ├── audio2gesture_step_fixed.onnx
│   │   │   └── ... (40+ files)
│   │   ├── rsmt/                  # Realtime stylized motion transition
│   │   │   ├── RSMTBVHConverter.js
│   │   │   ├── deepphase.onnx
│   │   │   ├── stylevae.onnx
│   │   │   └── ... (20+ files)
│   │   ├── deepmimic/             # DeepMimic humanoid animations
│   │   │   ├── DeepMimicBVHConverter.js
│   │   │   ├── compatible_humanoid3d_*.onnx
│   │   │   └── ... (50+ files)
│   │   └── faceformer/            # Facial animation from audio
│   │       ├── FaceFormerBVHConverter.js
│   │       ├── faceformer_core_step.onnx
│   │       └── ... (80+ files)
│   ├── components/animation/      # Animation components
│   │   ├── vrm/                   # VRM avatar system
│   │   │   ├── VRMBVHAdapter.js   # Core VRM integration
│   │   │   ├── conversation/      # Conversation interfaces
│   │   │   ├── diagnostics/       # Debug and validation
│   │   │   └── ... (25 files total)
│   │   └── timeline/              # Animation timeline system
│   │       ├── BVHTimeline.js
│   │       └── ... (6 files)
│   ├── testing/                   # 🧪 ALL TESTS CONSOLIDATED
│   │   ├── demos/                 # Demo applications
│   │   ├── e2e/                   # End-to-end Playwright tests
│   │   ├── unit/                  # Component unit tests
│   │   ├── integration/           # Integration tests
│   │   ├── performance/           # Performance benchmarks
│   │   └── legacy/                # Legacy test files
│   └── utils/                     # 🔧 UTILITIES & TOOLS
│       ├── kokoro.js/             # Audio processing utilities
│       ├── workers/               # Web workers
│       ├── debug/                 # Debug scripts
│       └── serve_with_headers.py  # Development server
├── assets/                        # Static assets
├── config/                        # Configuration files  
├── docs/                          # Documentation
├── index.html                     # Main entry point
└── README.md                      # This documentation
```

## Development Notes

### ✅ Recently Completed Reorganization
- **Complete File Organization**: All scattered files moved to systematic structure
- **Individual Component Testing**: Each motion model now testable separately  
- **Consolidated Testing**: All test types organized in `src/testing/`
- **Clean Module Boundaries**: Clear separation between models, components, and utilities
- **No Stray Files**: All files properly organized with clear locations

### Technical Capabilities
- **WebNN Support**: Neural network inference acceleration
- **WebGPU Integration**: GPU-accelerated processing for real-time performance
- **WASM Optimization**: WebAssembly modules for compute-intensive operations
- **VRM Avatar System**: Complete 25-file VRM integration with conversation interfaces
- **Multi-Modal AI**: 13 different AI models working together

### Current Development Focus
- ✅ Systematic component organization complete
- ✅ Individual testing capability achieved
- 🔄 Playwright test suite integration ongoing
- 🔄 ES6 module conversion for full compatibility
- 📋 Performance optimization for WebNN/WebGPU pipeline

### Next Development Steps
1. **Component Integration Testing**: Validate that all moved components work together
2. **Performance Benchmarking**: Test individual vs. integrated performance
3. **WebNN/WebGPU Optimization**: Optimize neural network execution
4. **Documentation Updates**: Update all component-specific documentation

## Citation

If referencing this visualization work, please cite the original RSMT paper:

```bibtex
@inproceedings{tang2023rsmt,
  title={RSMT: Real-time Stylized Motion Transition for Characters},
  author={Tang, Xiangjun and Wu, Linjun and Wang, He and Hu, Bo and Gong, Xu and Liao, Yuchen and Li, Songnan and Kou, Qilong and Jin, Xiaogang},
  booktitle={SIGGRAPH '23 Conference Proceedings},
  year={2023},
  publisher={ACM}
}
```

## Questions or Issues

For questions about:
- **This visualization**: Check browser console for detailed logs
- **Real RSMT system**: Refer to the main repository documentation
- **100STYLE dataset**: Visit https://www.ianxmason.com/100style/
