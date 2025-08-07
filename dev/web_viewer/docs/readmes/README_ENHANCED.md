# Enhanced VRM Avatar AI Conversation System

This enhanced implementation integrates advanced audio-to-audio conversational features from the `conversational-webgpu` reference implementation into a VRM avatar chat interface.

## Key Features Integrated

### 🎤 **Real-time Voice Activity Detection (VAD)**
- Uses Silero VAD for accurate speech detection
- Handles noise suppression and echo cancellation
- Intelligent buffering with overflow management
- Configurable sensitivity thresholds

### 🗣️ **High-quality Speech-to-Text**
- Whisper-base model for accurate transcription
- Support for WebGPU acceleration
- Real-time audio processing with AudioWorklets
- Automatic silence detection and chunking

### 🧠 **Advanced Conversational AI**
- SmolLM2-1.7B-Instruct for natural conversations
- Streaming response generation
- Context-aware conversation history
- Configurable system prompts for different personalities

### 🔊 **High-quality Text-to-Speech**
- Kokoro TTS for natural voice synthesis
- Multiple voice options (af_heart, am_adam, etc.)
- Streaming audio output
- Fallback to Web Speech API

### 🎵 **Advanced Audio Pipeline**
- Custom AudioWorklet processors for low-latency processing
- Separate VAD and playback worklets
- Sample rate conversion and buffering
- Real-time audio visualization feedback

### 🎮 **Interactive UI**
- Visual feedback for listening/speaking states
- Voice selection dropdown
- Real-time conversation display
- Call duration tracking
- Debug information panel

## File Structure

```
dev/web_viewer/
├── vrm_ai_conversation_enhanced.html    # Main enhanced interface
├── conversation-demo.html               # Simple demo/test interface
├── js/
│   ├── conversationWorker.js           # Main AI conversation worker
│   ├── vad-processor.js                # Voice Activity Detection worklet
│   ├── play-worklet.js                 # Audio playback worklet
│   ├── constants.js                    # Shared configuration constants
│   ├── vrm_mesh_classroom.js           # VRM and 3D scene management
│   ├── VRMLightingManager.js           # Lighting system
│   └── VRMBVHAdapter.js                # Animation system
└── README_ENHANCED.md                  # This documentation
```

## Technical Implementation

### Audio Processing Pipeline

1. **Input**: Microphone → AudioContext (16kHz)
2. **VAD**: Real-time voice activity detection
3. **Buffering**: Smart audio chunking with overflow handling
4. **STT**: Whisper model transcription
5. **AI**: SmolLM2 conversation generation
6. **TTS**: Kokoro voice synthesis
7. **Output**: AudioContext (24kHz) → Speakers

### Model Configuration

- **VAD**: `onnx-community/silero-vad`
- **STT**: `onnx-community/whisper-base`
- **LLM**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **TTS**: `onnx-community/Kokoro-82M-v1.0-ONNX`

### Key Constants

```javascript
INPUT_SAMPLE_RATE = 16000      // Input audio sample rate
OUTPUT_SAMPLE_RATE = 24000     // TTS output sample rate
SPEECH_THRESHOLD = 0.3         // VAD speech detection threshold
MIN_SILENCE_DURATION = 400ms   // Silence before ending speech
MIN_SPEECH_DURATION = 250ms    // Minimum speech length to process
```

## Usage

### Basic Demo
1. Open `conversation-demo.html` in a modern browser
2. Click "Start Conversation" and allow microphone access
3. Speak naturally - the system will detect speech and respond

### Full VRM Avatar Interface
1. Open `vrm_ai_conversation_enhanced.html`
2. Load VRM character and classroom environment
3. Start conversation and interact with your 3D avatar

### API Integration

```javascript
// Initialize conversation worker
const worker = new Worker('./js/conversationWorker.js', { type: 'module' });

// Handle worker messages
worker.addEventListener('message', ({ data }) => {
    switch (data.type) {
        case "status":
            // Handle system status updates
            break;
        case "output":
            // Handle AI responses (text + audio)
            break;
        case "info":
            // Handle debug information
            break;
    }
});

// Send audio data
worker.postMessage({ type: "audio", buffer: audioBuffer });

// Control conversation
worker.postMessage({ type: "start_call" });
worker.postMessage({ type: "end_call" });
worker.postMessage({ type: "set_voice", voice: "af_heart" });
```

## Browser Requirements

- **WebGPU Support**: Required for optimal performance (Chrome 113+)
- **AudioWorklet Support**: Required for real-time audio processing
- **Microphone Access**: Required for voice input
- **Modern JavaScript**: ES modules, async/await, Web Workers

## Fallback Support

- Falls back to WASM if WebGPU unavailable
- Falls back to Web Speech API if Kokoro TTS fails
- Graceful degradation for older browsers

## Performance Optimizations

- **Streaming Generation**: Real-time response generation
- **Audio Chunking**: Efficient memory usage
- **Model Caching**: Persistent conversation context
- **WebGPU Acceleration**: GPU-accelerated inference
- **Worklet Processing**: Low-latency audio processing

## Debugging

Enable debug information to monitor:
- Model loading progress
- Audio buffer status
- VAD detection results
- Transcription accuracy
- Response generation
- Audio playback status

## Integration with VRM Avatars

The enhanced system integrates seamlessly with:
- **VRM Character Loading**: 3D avatar visualization
- **Animation Sync**: Lip sync and gesture animations
- **Emotion Expression**: Visual feedback based on conversation context
- **Scene Management**: Classroom environments and lighting
- **Camera Controls**: Interactive 3D scene navigation

## Future Enhancements

- **Multi-language Support**: Additional language models
- **Emotion Recognition**: Sentiment analysis for avatar expressions
- **Custom Voice Training**: Personalized TTS voices
- **Video Input**: Visual conversation context
- **AR/VR Integration**: Immersive avatar interactions
