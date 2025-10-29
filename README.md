# Motion - Ichika VRM Classroom System

[![Ultimate Conversation E2E](https://github.com/navichat/motion/actions/workflows/e2e-ultimate-conversation.yml/badge.svg)](https://github.com/navichat/motion/actions/workflows/e2e-ultimate-conversation.yml)

An interactive 3D video game with VRM characters, BVH animation, and neural network-powered AI avatars. Features the Ichika VRM classroom system with real-time conversation, walking animations, and gesture controls.

## 🚀 Quick Start

Get started with the Ichika VRM classroom system in under 2 minutes:

```bash
# Clone and setup
git clone https://github.com/navichat/motion.git
cd motion
npm install

# Start development server and open VRM classroom demo
npm run dev
```

This will:
- Start the development server on http://localhost:8080
- Automatically open the working Ichika VRM classroom system
- Load the 3D classroom environment with walking VRM character

## 📋 Available Commands

### Core Commands

```bash
# Development mode - starts server and opens main VRM demo
npm run dev

# Production server - serves the application
npm run serve  

# Build and prepare assets (if needed)
npm run build
```

### VRM Classroom Demos

```bash
# Launch specific VRM classroom demos
npm run demo:classroom          # Main classroom system
npm run demo:classroom:real     # Real asset loading version
npm run demo:voice              # Voice conversation system
```

### Testing

```bash
# Run all tests
npm test

# VRM and animation tests
npm run test:vrm               # VRM integration tests  
npm run test:animation         # Animation system tests
npm run test:conversation      # Voice conversation tests
```

## 🎮 What You'll See

The main demo showcases:

### **Real VRM Character System**
- **Ichika VRM Model**: Full 3D anime character with detailed features
- **Walking System**: Character moves between classroom positions (center → blackboard → teacher desk → front)
- **Gesture Animations**: Wave, bow, point gestures with smooth transitions
- **Interactive Controls**: Click buttons to control character movement and actions

### **3D Classroom Environment**  
- **Realistic Classroom**: Desks, blackboard, windows, proper lighting
- **Collision Detection**: Character navigates around obstacles
- **Camera System**: Mouse controls, zoom, and follow modes
- **60 FPS Rendering**: Smooth real-time 3D graphics

### **Infrastructure Integration**
- **AdvancedVRMLoader**: Loads real VRM files with progress tracking
- **ClassroomGLBLoader**: Handles 3D classroom environment loading  
- **VRMBVHAdapter**: Maps BVH motion capture to VRM animations
- **BVHTimeline**: Manages animation sequences and blending
- **PathfindingBVHPlanner**: Controls intelligent movement paths

## 🛠️ System Architecture

### **Asset Loading**
- **VRM Character**: `ichika.vrm` (16MB) - Full anime character model
- **Classroom Scene**: `classroom.glb` (21MB) - 3D environment with furniture
- **Animation Data**: BVH motion capture files for realistic movement
- **CDN Fallbacks**: Automatic fallback when CDN resources unavailable

### **Neural Network Integration**
- **Speech Recognition**: Whisper ASR for voice input
- **Text-to-Speech**: Multiple TTS backends (Kokoro, SpeechT5, Speech API)
- **Gesture Generation**: AI-driven gesture synthesis from audio
- **Motion Planning**: Neural network pathfinding for character movement

## 📁 Project Structure

```
motion/
├── dev/web_viewer/                    # Main VRM system
│   ├── demos/                         # Interactive demos
│   │   ├── real_working_ichika_vrm_classroom_system.html  # Main demo
│   │   ├── ichika_voice_conversation_demo.html           # Voice chat
│   │   └── working_ichika_vrm_classroom_complete.html    # Complete system
│   ├── src/components/animation/vrm/  # VRM infrastructure
│   │   ├── AdvancedVRMLoader.js       # VRM model loading
│   │   ├── ClassroomGLBLoader.js      # 3D environment loading
│   │   ├── VRMBVHAdapter.js           # Animation mapping
│   │   ├── BVHTimeline.js             # Animation timeline
│   │   └── PathfindingBVHPlanner.js   # Movement planning
│   ├── assets/                        # VRM models and scenes
│   │   ├── avatars/ichika.vrm         # Main character model
│   │   └── scenes/classroom.glb       # Classroom environment
│   └── tests/                         # Comprehensive test suite
├── docs/                              # Documentation
├── pytorch_DeepMimic/                 # DeepMimic RL training
├── RSMT-Realtime-Stylized-Motion-Transition/  # Motion synthesis
└── BvhToDeepMimic/                   # BVH conversion tools
```

## 🎯 Features

### **Interactive 3D Game**
- ✅ Real VRM character rendering with bone mapping
- ✅ Interactive walking system between classroom positions
- ✅ BVH animation integration (wave, bow, point gestures)
- ✅ 3D classroom environment with lighting and shadows
- ✅ Camera controls with mouse interaction and zoom
- ✅ 60 FPS real-time rendering with performance monitoring

### **Voice Conversation**
- ✅ Microphone input with voice activity detection
- ✅ Real-time speech recognition (Whisper ASR)
- ✅ Multiple TTS backends (Kokoro, SpeechT5, Speech API)
- ✅ Audio-driven facial animations and gestures
- ✅ Multi-turn conversation support with state management

### **AI & Neural Networks**
- ✅ 19+ AI model types across language, audio, motion, and compute
- ✅ WebGPU, WebNN, and WASM acceleration
- ✅ ONNX Runtime compatibility layer
- ✅ Real-time inference with performance metrics
- ✅ Automated result capture and JSON export

## 🔧 Development

### **Requirements**
- Node.js 16+  
- Python 3.7+
- Modern browser with WebGL 2.0 support
- Optional: CUDA-compatible GPU for AI model training

### **Setup**
```bash
# Install dependencies
npm install

# Prepare AI models (optional)
npm run prepare:models

# Start development server
npm run dev
```

### **Testing**
```bash
# Run all tests
npm test

# Animation and VRM tests
npm run test:animation
npm run test:vrm

# Voice conversation tests  
npm run test:conversation

# Performance tests
npm run test:perf
```

## 📚 Documentation

- **[System Overview](./docs/README.md)** - Complete system architecture
- **[VRM Integration Guide](./dev/web_viewer/docs/readmes/ICHIKA_CLASSROOM_VRM_IMPLEMENTATION_PLAN.md)** - VRM implementation details
- **[Testing Guide](./dev/web_viewer/docs/TESTING_INFRASTRUCTURE.md)** - Comprehensive testing documentation
- **[API Reference](./dev/web_viewer/README.md)** - Technical API documentation

## 🌟 Demos

### **Main Classroom Demo**
**URL**: `http://localhost:8080/dev/web_viewer/demos/real_working_ichika_vrm_classroom_system.html`

Features the complete VRM classroom system with walking character, gesture controls, and 3D environment.

### **Voice Conversation Demo**  
**URL**: `http://localhost:8080/dev/web_viewer/demos/ichika_voice_conversation_demo.html`

Interactive voice chat with the Ichika character using microphone input and TTS responses.

### **Complete Interactive System**
**URL**: `http://localhost:8080/dev/web_viewer/demos/working_ichika_vrm_classroom_complete.html`

Full-featured demo combining all systems: VRM rendering, voice interaction, and classroom environment.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project contains multiple components with different licenses:
- Main system: MIT License
- DeepMimic components: Custom License (see respective directories)
- Third-party libraries: See individual package licenses

## 🎊 Credits

Built with modern web technologies:
- **Three.js** - 3D graphics rendering
- **VRM** - 3D character models and animation
- **Playwright** - End-to-end testing
- **WebGL/WebGPU** - Hardware-accelerated graphics
- **Web Audio API** - Real-time audio processing