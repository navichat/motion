/**
 * Constants for the audio-to-audio conversation system
 * Based on conversational-webgpu reference implementation
 */

/**
 * Sample rate of the input audio.
 * Coincidenally, this is the same for both models (Moonshine and Silero VAD)
 */
export const INPUT_SAMPLE_RATE = 16000;
const INPUT_SAMPLE_RATE_MS = INPUT_SAMPLE_RATE / 1000;

/**
 * Probabilities ABOVE this value are considered as SPEECH
 */
export const SPEECH_THRESHOLD = 0.3;

/**
 * If current state is SPEECH, and the probability of the next state
 * is below this value, it is considered as NON-SPEECH.
 */
export const EXIT_THRESHOLD = 0.1;

/**
 * After each speech chunk, wait for at least this amount of silence
 * before considering the next chunk as a new speech chunk
 */
export const MIN_SILENCE_DURATION_MS = 400;
export const MIN_SILENCE_DURATION_SAMPLES =
  MIN_SILENCE_DURATION_MS * INPUT_SAMPLE_RATE_MS;

/**
 * Pad the speech chunk with this amount each side
 */
export const SPEECH_PAD_MS = 80;
export const SPEECH_PAD_SAMPLES = SPEECH_PAD_MS * INPUT_SAMPLE_RATE_MS;

/**
 * Final speech chunks below this duration are discarded
 */
export const MIN_SPEECH_DURATION_SAMPLES = 250 * INPUT_SAMPLE_RATE_MS; // 250 ms

/**
 * Maximum duration of audio that can be handled by Moonshine/Whisper
 */
export const MAX_BUFFER_DURATION = 30;

/**
 * Size of the incoming buffers
 */
export const NEW_BUFFER_SIZE = 512;

/**
 * The number of previous buffers to keep, to ensure the audio is padded correctly
 */
export const MAX_NUM_PREV_BUFFERS = Math.ceil(
  SPEECH_PAD_SAMPLES / NEW_BUFFER_SIZE,
);

/**
 * TTS output sample rate (Kokoro TTS default)
 */
export const OUTPUT_SAMPLE_RATE = 24000;

/**
 * Device configurations for different AI models
 */
export const DEVICE_DTYPE_CONFIGS = {
  webgpu: {
    encoder_model: "fp32",
    decoder_model_merged: "fp32",
  },
  wasm: {
    encoder_model: "fp32",
    decoder_model_merged: "q8",
  },
};

/**
 * Model configurations
 */
export const MODEL_CONFIGS = {
  VAD_MODEL: "onnx-community/silero-vad",
  WHISPER_MODEL: "onnx-community/whisper-base",
  LLM_MODEL: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  KOKORO_MODEL: "onnx-community/Kokoro-82M-v1.0-ONNX",
};

/**
 * System prompts for different conversation contexts
 */
export const SYSTEM_PROMPTS = {
  DEFAULT: "You're a helpful and conversational AI avatar assistant. Keep your responses short, clear, and casual. You are speaking through voice, so be natural and engaging. Express emotions appropriately.",
  FRIENDLY: "You're a cheerful and friendly AI avatar companion. Be warm, encouraging, and supportive in your responses. Keep conversations light and positive.",
  PROFESSIONAL: "You're a professional AI assistant providing helpful information. Be concise, accurate, and courteous in your responses.",
  EDUCATIONAL: "You're an educational AI tutor. Help explain concepts clearly and encourage learning. Ask follow-up questions to ensure understanding.",
};

/**
 * Audio processing settings
 */
export const AUDIO_SETTINGS = {
  ECHO_CANCELLATION: true,
  AUTO_GAIN_CONTROL: true,
  NOISE_SUPPRESSION: true,
  CHANNEL_COUNT: 1,
  LATENCY_HINT: "interactive",
};

/**
 * UI animation settings
 */
export const UI_ANIMATIONS = {
  LISTENING_PULSE_DURATION: 2000,
  SPEAKING_PULSE_DURATION: 1000,
  RIPPLE_DURATION: 1500,
  RIPPLE_INTERVAL: 1000,
};

/**
 * Error handling and retry settings
 */
export const ERROR_SETTINGS = {
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,
  TIMEOUT_DURATION: 30000,
  MAX_SILENCE_DURATION: 10000,
};
