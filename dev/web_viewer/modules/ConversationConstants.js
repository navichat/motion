/**
 * Constants for Conversational WebGPU Voice Chat
 * Based on conversational-webgpu/src/constants.js
 */

/**
 * Sample rate of the input audio.
 * Coindicentally, this is the same for both models (Moonshine/Whisper and Silero VAD)
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
  (SPEECH_PAD_SAMPLES * 2) / NEW_BUFFER_SIZE
);

/**
 * Voice options for TTS
 */
export const TTS_VOICES = [
  "af_bella", "af_nicole", "af_sky", "af_heart",
  "am_adam", "am_michael", "am_sarah", "bf_emma",
  "bf_isabella", "bm_george", "bm_lewis"
];

/**
 * Default system prompts for different conversation styles
 */
export const SYSTEM_PROMPTS = {
  conversational: "You're a helpful and conversational voice assistant. Keep your responses short, clear, and casual.",
  educational: "You're an educational assistant helping users learn. Explain concepts clearly and encourage questions.",
  creative: "You're a creative assistant helping with artistic and imaginative tasks. Be inspiring and supportive.",
  technical: "You're a technical assistant helping with programming and technical questions. Be precise and detailed."
};

/**
 * Model configurations for different devices
 */
export const DEVICE_DTYPE_CONFIGS = {
  webgpu: {
    encoder_model: "fp32",
    decoder_model_merged: "fp32",
    vad_dtype: "fp32",
    llm_dtype: "q4f16",
    tts_dtype: "fp32"
  },
  wasm: {
    encoder_model: "fp32", 
    decoder_model_merged: "q8",
    vad_dtype: "fp32",
    llm_dtype: "q8",
    tts_dtype: "fp32"
  }
};

/**
 * Default model IDs
 */
export const DEFAULT_MODELS = {
  vad: "onnx-community/silero-vad",
  asr: "onnx-community/whisper-base", // or "onnx-community/moonshine-base-ONNX"
  llm: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  tts: "onnx-community/Kokoro-82M-v1.0-ONNX"
};

/**
 * Audio buffer configuration
 */
export const AUDIO_CONFIG = {
  bufferDuration: MAX_BUFFER_DURATION,
  sampleRate: INPUT_SAMPLE_RATE,
  chunkSize: NEW_BUFFER_SIZE,
  minChunkSize: 512
};
