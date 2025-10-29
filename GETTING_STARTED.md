# Getting Started with Ichika VRM Classroom System

This guide will get you up and running with the Ichika VRM classroom system in just a few minutes.

## Prerequisites

Make sure you have the following installed:

- **Node.js 16+** - [Download here](https://nodejs.org/)
- **Python 3.7+** - [Download here](https://python.org/)
- **Git** - [Download here](https://git-scm.com/)
- **Modern web browser** with WebGL 2.0 support (Chrome, Firefox, Safari, Edge)

## Quick Setup (2 minutes)

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/navichat/motion.git
cd motion

# Install dependencies
npm install
```

### 2. Start the System

```bash
# Start development server and open VRM classroom demo
npm run dev
```

This command will:
- Start the development server on http://localhost:8080
- Automatically open the working Ichika VRM classroom system in your browser
- Display the 3D classroom with walking VRM character

### 3. Explore the Demos

Once running, you can explore different demos:

```bash
# Main classroom with VRM character walking
npm run demo:classroom

# Real asset loading version (16MB VRM + 21MB classroom)
npm run demo:classroom:real

# Voice conversation with microphone input
npm run demo:voice
```

## What You'll See

### Main VRM Classroom Demo

![VRM Classroom Screenshot](https://github.com/user-attachments/assets/d562a55a-e9de-4020-8f09-c9457b87df43)

**Features:**
- ✅ **3D Classroom Environment**: Realistic classroom with desks, blackboard, windows
- ✅ **Ichika VRM Character**: Full anime-style 3D character with detailed animations
- ✅ **Walking System**: Character moves between positions (center, blackboard, teacher desk, front)
- ✅ **Interactive Controls**: Click buttons to control movement and gestures
- ✅ **Real-time Rendering**: 60 FPS smooth 3D graphics with lighting and shadows

### Voice Conversation Demo

**Features:**
- 🎤 **Microphone Input**: Speak to the character using your microphone
- 🗣️ **Speech Recognition**: Real-time speech-to-text using Whisper ASR
- 🎵 **Text-to-Speech**: Multiple TTS backends (Kokoro, SpeechT5, Speech API)
- 🤖 **AI Responses**: Conversational AI with personality
- 💃 **Audio-driven Animation**: Gestures and expressions from audio analysis

## Directory Structure

```
motion/
├── dev/web_viewer/                    # Main VRM system
│   ├── demos/                         # Interactive demos
│   │   ├── real_working_ichika_vrm_classroom_system.html  # 👈 Main demo
│   │   ├── ichika_voice_conversation_demo.html           # 👈 Voice chat
│   │   └── working_ichika_vrm_classroom_complete.html    # 👈 Complete system
│   ├── src/components/animation/vrm/  # VRM infrastructure components
│   ├── assets/                        # VRM models and 3D scenes
│   └── tests/                         # Test suite
├── docs/                              # Documentation
└── README.md                          # Project overview
```

## Key Demo URLs

With server running on http://localhost:8080:

| Demo | URL | Description |
|------|-----|-------------|
| **Main Classroom** | `/dev/web_viewer/demos/real_working_ichika_vrm_classroom_system.html` | Complete VRM classroom with walking character |
| **Voice Chat** | `/dev/web_viewer/demos/ichika_voice_conversation_demo.html` | Interactive voice conversation system |
| **Complete System** | `/dev/web_viewer/demos/working_ichika_vrm_classroom_complete.html` | Full-featured demo with all systems |

## Interactive Controls

### Classroom Demo Controls

**Character Movement:**
- **Walk to Center** - Move character to classroom center
- **Walk to Blackboard** - Move character to blackboard position  
- **Walk to Teacher Desk** - Move character to teacher's desk
- **Walk to Front** - Move character to front of classroom
- **Start Walking Demo** - Automated walking tour of classroom

**Gesture Controls:**
- **Wave Gesture** - Character waves hello
- **Bow Gesture** - Character bows politely
- **Point Gesture** - Character points at something
- **Follow Ichika** - Camera follows character movement

### Voice Chat Demo Controls

**Voice Interaction:**
- **Start Microphone** - Begin voice input
- **Listen and Reply** - Single turn conversation
- **Conversation Loop** - Multi-turn conversation
- **Stop All** - End conversation

**Settings:**
- **Backend Selection**: Choose TTS engine (Speech, Kokoro, SpeechT5, Beeps)
- **Audio Playback**: Enable/disable audio output
- **ASR Engine**: Choose speech recognition (Whisper, Speech, Fake)
- **VRM Rendering**: Enable/disable 3D character

## Troubleshooting

### Server Won't Start

```bash
# Check if Python is installed
python3 --version

# Check if port 8080 is available
lsof -ti:8080

# Kill process using port 8080 if needed
kill -9 $(lsof -ti:8080)

# Try alternative port
PORT=8081 python3 dev/web_viewer/serve_with_headers.py
```

### Demo Won't Load

1. **Clear browser cache** and reload
2. **Check browser console** for errors (F12 Developer Tools)
3. **Verify server is running** at http://localhost:8080
4. **Try a different browser** (Chrome recommended)

### Character Not Appearing

1. **Check WebGL support** in your browser
2. **Verify assets are loading** in browser Network tab
3. **Try demo with fallback** rendering mode
4. **Check for JavaScript errors** in browser console

### Voice Chat Not Working

1. **Allow microphone permissions** in browser
2. **Check microphone is working** in system settings
3. **Try different TTS backend** (beeps for simple testing)
4. **Verify audio output** is enabled

## System Requirements

### Minimum Requirements
- **Browser**: Chrome 80+, Firefox 75+, Safari 14+, Edge 80+
- **RAM**: 4GB available memory
- **GPU**: Any WebGL 2.0 compatible graphics card
- **Network**: Internet connection for CDN assets (with fallbacks)

### Recommended Requirements  
- **Browser**: Latest Chrome or Firefox
- **RAM**: 8GB available memory
- **GPU**: Dedicated graphics card with WebGL 2.0
- **Network**: Stable broadband connection
- **Audio**: Microphone and speakers/headphones for voice features

## Next Steps

### Explore Advanced Features

1. **Run Tests**: `npm test` - Verify system functionality
2. **Voice Conversation**: Try the microphone-enabled demos
3. **Customize Character**: Modify VRM models and animations
4. **Extend System**: Add new gestures, environments, or AI models

### Development

1. **Edit Demos**: Modify HTML files in `/dev/web_viewer/demos/`
2. **Add Components**: Extend VRM infrastructure in `/dev/web_viewer/src/components/`
3. **Create Tests**: Add new tests in `/dev/web_viewer/tests/`
4. **Documentation**: Update documentation in `/docs/`

### Learn More

- **[Full Documentation](./docs/README.md)** - Complete system overview
- **[VRM Implementation Guide](./dev/web_viewer/docs/readmes/ICHIKA_CLASSROOM_VRM_IMPLEMENTATION_PLAN.md)** - Technical details
- **[Testing Guide](./dev/web_viewer/docs/TESTING_INFRASTRUCTURE.md)** - Testing framework
- **[API Reference](./dev/web_viewer/README.md)** - Technical API documentation

## Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Review browser console** for error messages
3. **Open GitHub issue** with detailed error information
4. **Include system specs** (browser, OS, hardware) when reporting issues

Happy exploring with the Ichika VRM classroom system! 🎉