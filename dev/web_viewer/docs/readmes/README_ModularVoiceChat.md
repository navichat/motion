# Modular Voice Chat Interface

A sophisticated, memory-efficient voice chat system with automatic model management and event-driven architecture.

## 🏗️ Architecture

The system is divided into specialized modules that handle different aspects of voice chat:

### Core Modules

1. **VoiceChatInterface** - Main orchestrator module
   - Coordinates all other modules
   - Manages conversation flow
   - Handles memory pressure and model eviction
   - Event-driven architecture

2. **WhisperModule** - Speech-to-text processing
   - Supports multiple Whisper model sizes
   - Automatic model loading/unloading
   - WebGPU/WASM device support

3. **KokoroModule** - Text-to-speech synthesis
   - High-quality neural TTS
   - Multiple voice options
   - Audio chunking for streaming

4. **LlamaModule** - Language model for conversation
   - Multiple LLM options with fallbacks
   - Conversation history management
   - Quantized model support

5. **VoiceActivityDetector** - Real-time voice detection
   - Web Audio API based VAD
   - Configurable sensitivity
   - Automatic speech segmentation

6. **AudioQueue** - Audio playback management
   - Seamless audio chunk playback
   - Queue management
   - Volume control and crossfading

## 🧠 Memory Management

The system implements intelligent memory management:

- **Automatic Model Eviction**: Models are unloaded when memory usage exceeds configurable thresholds
- **LRU-based Eviction**: Least recently used models are evicted first
- **Lazy Loading**: Models are only loaded when needed
- **Cache Timeout**: Unused models are automatically unloaded after a timeout period

## 🔄 Event Loop & Voice Activity Detection

1. **Voice Activity Detection**: Continuously monitors audio input for speech
2. **Speech Segmentation**: Automatically detects speech start/end
3. **Processing Chain**: 
   - Speech detected → Whisper transcription → LLM response → Kokoro synthesis → Audio playback
4. **Chunked Processing**: Long responses are split into chunks for better user experience

## 📁 File Structure

```
modules/
├── VoiceChatInterface.js     # Main orchestrator
├── WhisperModule.js          # Speech-to-text
├── KokoroModule.js           # Text-to-speech
├── LlamaModule.js            # Language model
├── VoiceActivityDetector.js  # Voice detection
└── AudioQueue.js             # Audio playback

voiceChatExample.js           # Usage example
voiceChatDemo.html           # Demo interface
```

## 🚀 Usage

### Basic Setup

```javascript
import { VoiceChatInterface } from './modules/VoiceChatInterface.js';

const voiceChat = new VoiceChatInterface({
    memoryThresholdMB: 512,
    vadSensitivity: 0.6,
    whisperModel: 'tiny',
    llamaModel: 'Xenova/TinyLlama-1.1B-Chat-v0.4',
    voice: 'af_heart'
});

// Initialize
await voiceChat.initialize();

// Start listening
await voiceChat.startListening();
```

### Event Handling

```javascript
// Model loading events
voiceChat.addEventListener('modelLoaded', (event) => {
    console.log(`${event.detail.module} model loaded`);
});

// Conversation events
voiceChat.addEventListener('transcription', (event) => {
    console.log('User said:', event.detail.text);
});

voiceChat.addEventListener('response', (event) => {
    console.log('AI responded:', event.detail.output);
});

// Memory management events
voiceChat.addEventListener('memoryPressure', (event) => {
    console.log(`Memory usage: ${event.detail.usedMB}MB`);
});
```

### Text Input (Bypass Voice)

```javascript
await voiceChat.processTextInput("Hello, how are you?");
```

## ⚙️ Configuration Options

### VoiceChatInterface Options

```javascript
{
    // Memory management
    memoryThresholdMB: 512,        // Evict models when memory exceeds this
    modelCacheTimeout: 30000,      // Unload models after 30s of inactivity
    
    // Voice Activity Detection
    vadSensitivity: 0.6,           // VAD sensitivity (0-1)
    minSpeechDuration: 300,        // Minimum speech duration (ms)
    maxSpeechDuration: 8000,       // Maximum speech duration (ms)
    silenceThreshold: 150,         // Silence to end speech (ms)
    
    // Audio
    audioSampleRate: 22050,        // Audio sample rate
    chunkSizeMs: 1000,            // Audio chunk size
    
    // Models
    whisperModel: 'tiny',          // 'tiny', 'base', 'small'
    llamaModel: 'Xenova/TinyLlama-1.1B-Chat-v0.4',
    kokoroModelPath: './Kokoro-82M-v1.0-ONNX/',
    voice: 'af_heart',             // Kokoro voice
    
    // Generation
    maxTokens: 150,                // Max LLM tokens
    temperature: 0.8,              // LLM temperature
    topP: 0.9,                    // LLM top-p
    
    // System prompt
    systemPrompt: "You are a helpful AI assistant..."
}
```

## 🎯 Key Features

### 1. Memory Efficiency
- Models are automatically loaded/unloaded based on usage
- Memory pressure detection and automatic eviction
- Configurable memory thresholds

### 2. Event-Driven Architecture
- All modules communicate through events
- Easy to extend and customize
- Loose coupling between components

### 3. Voice Activity Detection
- Real-time speech detection
- Automatic speech segmentation
- Configurable sensitivity

### 4. Audio Chunking
- Long responses are split into manageable chunks
- Seamless playback with audio queue
- Better user experience with streaming audio

### 5. Fallback Support
- Multiple model options with automatic fallbacks
- Graceful degradation when models fail
- Built-in conversation responses

### 6. Device Adaptation
- Automatic WebGPU/WASM device selection
- Quantized models for better performance
- Cross-platform compatibility

## 🔧 Integration with Existing Systems

To integrate with your existing VRM avatar system:

```javascript
// Replace the monolithic TTS/STT functions
const voiceChat = new VoiceChatInterface(options);

// Connect to existing avatar animation
voiceChat.addEventListener('response', (event) => {
    // Trigger avatar emotion based on response
    const emotion = extractEmotion(event.detail.output);
    if (vrmCharacter) {
        triggerAvatarEmotion(emotion);
    }
});

voiceChat.addEventListener('speaking', (event) => {
    // Sync avatar mouth movements
    if (event.detail.status === 'started') {
        startMouthAnimation();
    } else {
        stopMouthAnimation();
    }
});
```

## 🐛 Debugging

The system provides comprehensive debug information:

```javascript
// Get current status
const status = voiceChat.getStatus();
console.log(status);

// Listen for debug events
voiceChat.addEventListener('status', (event) => {
    console.log('Status:', event.detail.message);
});
```

## 🔄 Migration from Monolithic System

1. **Replace TTS functions**: Use `voiceChat.processTextInput()` instead of `speakText()`
2. **Replace STT**: Voice input is automatically handled by VAD
3. **Replace conversation**: LLM responses are automatic
4. **Update event handlers**: Use the new event system
5. **Remove manual model management**: The system handles this automatically

## 🚦 Performance Considerations

- **Model Size**: Smaller models (tiny Whisper, small LLMs) for better performance
- **Memory Limits**: Set appropriate memory thresholds for your device
- **Audio Quality**: Balance between quality and performance
- **Chunk Size**: Optimize chunk size for your use case

## 🔮 Future Enhancements

- **Streaming LLM**: Real-time token streaming
- **Custom VAD**: More sophisticated voice detection
- **Multi-language**: Support for multiple languages
- **Cloud Fallback**: Fallback to cloud APIs when local models fail
- **Voice Cloning**: Custom voice synthesis
- **Context Awareness**: Better conversation memory
