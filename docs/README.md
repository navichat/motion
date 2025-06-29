# Motion Workspace Documentation

## 🚀 Development Status Update

### ✅ Phase 1 COMPLETED: Foundation Migration
**Location**: `/home/barberb/motion/dev/`

The Chat Interface's 3D avatar viewer has been successfully migrated to a new development environment with enhanced capabilities:

- **3D Viewer**: Modern React + Three.js implementation 
- **Avatar Support**: VRM character loading with caching
- **Animations**: JSON format playback with advanced controls
- **Environments**: Classroom, stage, studio, and outdoor scenes
- **API Server**: FastAPI with REST endpoints
- **Documentation**: Comprehensive guides and examples

**Quick Start:**
```bash
cd /home/barberb/motion/dev/server
./start_dev.sh
# Open http://localhost:8081
```

### 🔄 Phase 2 NEXT: RSMT Integration
Upcoming neural network integration for advanced motion synthesis:

- RSMT PyTorch models (DeepPhase, StyleVAE, TransitionNet)
- 100STYLE dataset support
- Real-time style transfer and motion generation
- WebSocket streaming for live updates

---

This workspace contains multiple projects related to motion capture, character animation, and AI-driven movement synthesis. The projects work together to provide a complete pipeline from motion data processing to real-time character animation.

## Repository Overview

**Repository:** `motion` (Owner: navichat)  
**Current Branch:** `main`  
**Last Updated:** June 28, 2025

## Project Structure

```
motion/
├── docs/                                    # Documentation (this folder)
├── BvhToDeepMimic/                         # BVH to DeepMimic converter
├── pytorch_DeepMimic/                      # PyTorch implementation of DeepMimic
├── RSMT-Realtime-Stylized-Motion-Transition/  # Real-time stylized motion transitions
└── chat/                                   # Web-based character animation chat interface
```

## Projects Summary

### 1. BvhToDeepMimic
**Purpose:** Converts BVH (Biovision Hierarchy) motion capture files to DeepMimic format for reinforcement learning training.

**Key Features:**
- Converts motion capture data from standard BVH format to DeepMimic-compatible motion files
- Supports custom reference motions for training DeepMimic skills
- Compatible with SFU Motion Capture Database
- Configurable joint mapping and scaling

**Technologies:** Python 3.6+, PyQuaternion, NumPy

### 2. pytorch_DeepMimic
**Purpose:** PyTorch implementation of DeepMimic for learning imitation policies from reference motions.

**Key Features:**
- Reinforcement learning for character motion imitation
- Translation from original TensorFlow implementation to PyTorch
- Proximal Policy Optimization (PPO) algorithm
- Training and inference capabilities for humanoid characters

**Technologies:** PyTorch 1.12, PyBullet, OpenAI Gym, MPI4Py

### 3. RSMT (Real-time Stylized Motion Transition)
**Purpose:** Real-time generation of stylized motion transitions for character animation.

**Key Features:**
- Real-time motion transition generation
- Style-aware motion synthesis
- Phase manifold learning for motion timing
- Integration with 100STYLE dataset
- Deep learning-based motion generation

**Technologies:** PyTorch, PyTorch3D, PyTorch Lightning, NumPy, Matplotlib

### 4. Chat Interface
**Purpose:** Web-based application for real-time character animation and interaction.

**Key Features:**
- Real-time character animation rendering
- Web-based chat interface with animated avatars
- Server-client architecture for multiplayer support
- Integration with motion animation systems
- Account management and session handling

**Technologies:** Node.js, Koa, WebSockets, Mithril, MySQL, CloudKit

## Detailed Project Documentation

- [BvhToDeepMimic Details](./BvhToDeepMimic.md)
- [PyTorch DeepMimic Details](./pytorch_DeepMimic.md)
- [RSMT Details](./RSMT.md)
- [Chat Interface Details](./chat_interface.md)
- [Installation Guide](./installation.md)
- [Usage Examples](./usage_examples.md)

## Quick Start

1. **For BVH Conversion:**
   ```bash
   cd BvhToDeepMimic
   pip install bvhtodeepmimic
   python example_script.py
   ```

2. **For DeepMimic Training:**
   ```bash
   cd pytorch_DeepMimic/deepmimic
   python DeepMimic_Optimizer.py --arg_file train_humanoid3d_walk_args.txt
   ```

3. **For RSMT:**
   ```bash
   cd RSMT-Realtime-Stylized-Motion-Transition
   pip install -r requirements.txt
   python process_dataset.py --preprocess
   ```

4. **For Chat Interface:**
   ```bash
   cd chat/webapp
   npm install
   npm run build
   ```

## Workflow Integration

The projects in this workspace are designed to work together:

1. **Data Preparation:** Use BvhToDeepMimic to convert motion capture data
2. **Model Training:** Train imitation policies with pytorch_DeepMimic
3. **Real-time Synthesis:** Generate smooth transitions with RSMT
4. **Interactive Application:** Deploy characters in the chat interface

## Assets and Resources

- **Animations:** Located in `chat/assets/animations/`
- **Avatars:** Character models in `chat/assets/avatars/`
- **Scenes:** Environment assets in `chat/assets/scenes/`
- **Example Data:** Sample BVH files and conversions in respective project folders

## Requirements

### System Requirements
- **Operating System:** Linux (primary), macOS, Windows
- **Python:** 3.6+ (3.7+ recommended)
- **Node.js:** 16+ (for chat interface)
- **GPU:** CUDA-compatible GPU recommended for training

### Hardware Recommendations
- **Memory:** 16GB+ RAM for large dataset processing
- **Storage:** 50GB+ for datasets and model files
- **GPU:** NVIDIA GPU with 8GB+ VRAM for optimal training performance

## Contributing

Each project has its own contribution guidelines and testing procedures. See individual project documentation for specific requirements.

## License

Projects have individual licenses:
- BvhToDeepMimic: MIT License
- pytorch_DeepMimic: Custom License (see project)
- RSMT: Custom License (see project)
- Chat Interface: Custom License (see project)

## Support and Contact

For technical support and questions, refer to individual project documentation or create issues in the respective repositories.
