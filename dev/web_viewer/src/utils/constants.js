/**
 * Configuration constants for the voice chat system
 * Based on patterns from conversational-webgpu example
 */

// Audio configuration
export const INPUT_SAMPLE_RATE = 16000; // Standard for speech recognition
export const OUTPUT_SAMPLE_RATE = 24000; // Standard for TTS
export const INPUT_SAMPLE_RATE_MS = INPUT_SAMPLE_RATE / 1000;

// Voice Activity Detection
export const SPEECH_THRESHOLD = 0.3; // Probabilities above this are speech
export const EXIT_THRESHOLD = 0.1; // Below this exits speech mode
export const MIN_SILENCE_DURATION_MS = 400; // Wait between speech chunks
export const MIN_SILENCE_DURATION_SAMPLES = MIN_SILENCE_DURATION_MS * INPUT_SAMPLE_RATE_MS;
export const SPEECH_PAD_MS = 80; // Padding around speech
export const SPEECH_PAD_SAMPLES = SPEECH_PAD_MS * INPUT_SAMPLE_RATE_MS;
export const MIN_SPEECH_DURATION_SAMPLES = 250 * INPUT_SAMPLE_RATE_MS; // 250ms minimum

// Audio buffering
export const NEW_BUFFER_SIZE = 512;
export const MAX_BUFFER_DURATION = 30; // seconds
export const MAX_NUM_PREV_BUFFERS = 10;
export const MIN_CHUNK_SIZE = 512;

// Model configuration
export const DEFAULT_MODELS = {
  whisper: "onnx-community/whisper-base",
  llama: "HuggingFaceTB/SmolLM2-1.7B-Instruct", 
  kokoro: "onnx-community/Kokoro-82M-v1.0-ONNX",
  vad: "onnx-community/silero-vad"
};

// Device preferences by capability
export const DEVICE_DTYPE_CONFIGS = {
  webgpu: {
    encoder_model: "fp32",
    decoder_model_merged: "fp32",
    llm_dtype: "q4f16",
    tts_dtype: "fp32"
  },
  wasm: {
    encoder_model: "fp32", 
    decoder_model_merged: "q8",
    llm_dtype: "q8",
    tts_dtype: "fp32"
  }
};

// System prompts
export const SYSTEM_PROMPTS = {
  default: "You're a helpful and conversational voice assistant. Keep your responses short, clear, and casual.",
  detailed: "You are a helpful AI assistant. Keep responses concise and engaging. Include emotion markers like [happy], [thoughtful], [excited] in your responses.",
  technical: "You are a technical assistant. Provide accurate, concise information with practical examples when helpful."
};

// Audio processing
export const AUDIO_WORKLET_OPTIONS = {
  numberOfInputs: 1,
  numberOfOutputs: 0,
  channelCount: 1,
  channelCountMode: "explicit",
  channelInterpretation: "discrete"
};

// Memory management
export const MEMORY_THRESHOLDS = {
  warning: 256, // MB
  critical: 512, // MB
  eviction: 768 // MB
};

// Timeouts and intervals
export const TIMEOUTS = {
  modelCache: 30000, // 30 seconds
  audioContext: 5000, // 5 seconds
  workerResponse: 10000, // 10 seconds
  microphoneAccess: 8000 // 8 seconds
};

// Error retry limits
export const RETRY_LIMITS = {
  modelLoad: 2,
  audioContext: 3,
  microphone: 3,
  generation: 2
};
