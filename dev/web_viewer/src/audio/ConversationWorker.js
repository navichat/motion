/**
 * Conversation Worker - AI Voice Chat Worker
 * Based on conversational-webgpu patterns adapted for VRM
 */

import {
  // VAD
  AutoModel,

  // LLM
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,

  // Speech recognition
  Tensor,
  pipeline,
} from "@huggingface/transformers";

import { KokoroTTS, TextSplitterStream } from "kokoro-js";

// Configuration constants
const INPUT_SAMPLE_RATE = 16000;
const MAX_BUFFER_DURATION = 30;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const SPEECH_PAD_SAMPLES = 80 * (INPUT_SAMPLE_RATE / 1000);
const MAX_NUM_PREV_BUFFERS = 5;
const MIN_SILENCE_DURATION_SAMPLES = 400 * (INPUT_SAMPLE_RATE / 1000);
const MIN_SPEECH_DURATION_SAMPLES = 250 * (INPUT_SAMPLE_RATE / 1000);

// Model configuration
const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
let voice = "af_heart";
let device = "webgpu";

// Global variables
let tts, silero_vad, transcriber, tokenizer, llm;
let messages = [];
let past_key_values_cache;
let stopping_criteria;

// Audio processing variables
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;
let state = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
let isRecording = false;
let isPlaying = false;
let prevBuffers = [];
let postSpeechSamples = 0;

// System prompts
const SYSTEM_PROMPTS = {
  conversational: "You're a helpful and conversational voice assistant. Keep your responses short, clear, and casual.",
  educational: "You're an educational AI teacher in a virtual classroom. Explain concepts clearly and encourage learning.",
  creative: "You're a creative AI assistant. Help with artistic and imaginative projects with enthusiasm.",
  technical: "You're a technical AI assistant. Provide precise, accurate technical information and solutions.",
  friendly: "You're a friendly AI companion. Be warm, supportive, and engaging in conversation."
};

/**
 * Initialize all models
 */
async function initializeModels(config = {}) {
  try {
    device = config.device || "webgpu";
    voice = config.voice || "af_heart";
    
    self.postMessage({ type: "status", message: "Loading TTS model..." });
    
    // Load TTS
    tts = await KokoroTTS.from_pretrained(model_id, {
      dtype: "fp32",
      device: device,
    });
    
    self.postMessage({ type: "status", message: "Loading VAD model..." });
    
    // Load VAD
    silero_vad = await AutoModel.from_pretrained(
      "onnx-community/silero-vad",
      {
        config: { model_type: "custom" },
        dtype: "fp32",
      }
    );
    
    self.postMessage({ type: "status", message: "Loading ASR model..." });
    
    // Load ASR
    const DEVICE_DTYPE_CONFIGS = {
      webgpu: {
        encoder_model: "fp32",
        decoder_model_merged: "fp32",
      },
      wasm: {
        encoder_model: "fp32",
        decoder_model_merged: "q8",
      },
    };
    
    transcriber = await pipeline(
      "automatic-speech-recognition",
      "onnx-community/whisper-base",
      {
        device,
        dtype: DEVICE_DTYPE_CONFIGS[device],
      }
    );
    
    // Compile shaders
    await transcriber(new Float32Array(INPUT_SAMPLE_RATE));
    
    self.postMessage({ type: "status", message: "Loading LLM..." });
    
    // Load LLM
    const llm_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct";
    tokenizer = await AutoTokenizer.from_pretrained(llm_model_id);
    llm = await AutoModelForCausalLM.from_pretrained(llm_model_id, {
      dtype: "q4f16",
      device: "webgpu",
    });
    
    // Compile shaders
    await llm.generate({ ...tokenizer("x"), max_new_tokens: 1 });
    
    // Set initial system message
    const systemPrompt = SYSTEM_PROMPTS[config.systemPrompt] || SYSTEM_PROMPTS.conversational;
    messages = [{ role: "system", content: systemPrompt }];
    
    self.postMessage({
      type: "status",
      status: "ready",
      message: "Ready!",
      voices: tts.voices,
    });
    
  } catch (error) {
    self.postMessage({ error: { message: error.message, stack: error.stack } });
  }
}

/**
 * Voice Activity Detection
 */
async function vad(buffer) {
  const input = new Tensor("float32", buffer, [1, buffer.length]);
  const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);

  const { stateN, output } = await silero_vad({ input, sr, state });
  state = stateN;

  const isSpeech = output.data[0];

  return (
    isSpeech > SPEECH_THRESHOLD ||
    (isRecording && isSpeech >= EXIT_THRESHOLD)
  );
}

/**
 * Speech to Speech processing
 */
async function speechToSpeech(buffer) {
  if (isPlaying) return;
  
  isPlaying = true;

  try {
    // 1. Transcribe audio
    const result = await transcriber(buffer);
    const text = result.text.trim();
    
    if (["", "[BLANK_AUDIO]"].includes(text)) {
      isPlaying = false;
      return;
    }
    
    messages.push({ role: "user", content: text });
    
    self.postMessage({
      type: "transcription",
      text: text
    });

    // 2. Set up TTS streaming
    const splitter = new TextSplitterStream();
    const stream = tts.stream(splitter, { voice });
    
    (async () => {
      for await (const { text: chunkText, audio } of stream) {
        self.postMessage({ 
          type: "output", 
          text: chunkText, 
          result: { audio } 
        });
      }
    })();

    // 3. Generate LLM response
    const inputs = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
    });
    
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        splitter.push(text);
      },
    });

    stopping_criteria = new InterruptableStoppingCriteria();
    const { past_key_values, sequences } = await llm.generate({
      ...inputs,
      past_key_values: past_key_values_cache,
      do_sample: false,
      max_new_tokens: 1024,
      streamer,
      stopping_criteria,
      return_dict_in_generate: true,
    });
    
    past_key_values_cache = past_key_values;
    splitter.close();

    const decoded = tokenizer.batch_decode(
      sequences.slice(null, [inputs.input_ids.dims[1], null]),
      { skip_special_tokens: true }
    );

    const responseText = decoded[0];
    messages.push({ role: "assistant", content: responseText });
    
    self.postMessage({
      type: "response",
      text: responseText
    });
    
  } catch (error) {
    self.postMessage({ error: { message: error.message } });
  }
}

/**
 * Process text message (for typing input)
 */
async function processTextMessage(text) {
  if (isPlaying) return;
  
  isPlaying = true;
  
  try {
    messages.push({ role: "user", content: text });
    
    self.postMessage({
      type: "transcription",
      text: text
    });

    // Set up TTS streaming
    const splitter = new TextSplitterStream();
    const stream = tts.stream(splitter, { voice });
    
    (async () => {
      for await (const { text: chunkText, audio } of stream) {
        self.postMessage({ 
          type: "output", 
          text: chunkText, 
          result: { audio } 
        });
      }
    })();

    // Generate LLM response
    const inputs = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
    });
    
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        splitter.push(text);
      },
    });

    stopping_criteria = new InterruptableStoppingCriteria();
    const { past_key_values, sequences } = await llm.generate({
      ...inputs,
      past_key_values: past_key_values_cache,
      do_sample: false,
      max_new_tokens: 1024,
      streamer,
      stopping_criteria,
      return_dict_in_generate: true,
    });
    
    past_key_values_cache = past_key_values;
    splitter.close();

    const decoded = tokenizer.batch_decode(
      sequences.slice(null, [inputs.input_ids.dims[1], null]),
      { skip_special_tokens: true }
    );

    const responseText = decoded[0];
    messages.push({ role: "assistant", content: responseText });
    
    self.postMessage({
      type: "response",
      text: responseText
    });
    
  } catch (error) {
    self.postMessage({ error: { message: error.message } });
  }
}

/**
 * Reset audio buffer after recording
 */
function resetAfterRecording(offset = 0) {
  self.postMessage({
    type: "status",
    status: "recording_end",
    message: "Processing...",
  });
  
  BUFFER.fill(0, offset);
  bufferPointer = offset;
  isRecording = false;
  postSpeechSamples = 0;
}

/**
 * Dispatch for transcription and reset buffer
 */
function dispatchForTranscriptionAndResetAudioBuffer(overflow) {
  const buffer = BUFFER.slice(0, bufferPointer + SPEECH_PAD_SAMPLES);
  
  const prevLength = prevBuffers.reduce((acc, b) => acc + b.length, 0);
  const paddedBuffer = new Float32Array(prevLength + buffer.length);
  let offset = 0;
  
  for (const prev of prevBuffers) {
    paddedBuffer.set(prev, offset);
    offset += prev.length;
  }
  paddedBuffer.set(buffer, offset);
  
  speechToSpeech(paddedBuffer);
  
  if (overflow) {
    BUFFER.set(overflow, 0);
  }
  resetAfterRecording(overflow?.length || 0);
}

/**
 * Send greeting message
 */
function greet(text) {
  if (isPlaying) return;
  
  isPlaying = true;
  const splitter = new TextSplitterStream();
  const stream = tts.stream(splitter, { voice });
  
  (async () => {
    for await (const { text: chunkText, audio } of stream) {
      self.postMessage({ 
        type: "output", 
        text: chunkText, 
        result: { audio } 
      });
    }
  })();
  
  splitter.push(text);
  splitter.close();
  messages.push({ role: "assistant", content: text });
  
  self.postMessage({
    type: "response",
    text: text
  });
}

/**
 * Main message handler
 */
self.onmessage = async (event) => {
  const { type, buffer, voice: newVoice, config, text } = event.data;

  // Refuse new audio while playing back
  if (type === "audio" && isPlaying) return;

  switch (type) {
    case "initialize":
      await initializeModels(config);
      break;
      
    case "start_call": {
      const name = tts?.voices[voice]?.name || "Assistant";
      greet(`Hello! I'm ${name}, your AI assistant. How can I help you today?`);
      break;
    }
    
    case "end_call":
      messages = messages.slice(0, 1); // Keep only system message
      past_key_values_cache = null;
      break;
      
    case "interrupt":
      stopping_criteria?.interrupt();
      break;
      
    case "set_voice":
      voice = newVoice;
      break;
      
    case "set_personality":
      const systemPrompt = SYSTEM_PROMPTS[config.personality] || SYSTEM_PROMPTS.conversational;
      messages[0] = { role: "system", content: systemPrompt };
      past_key_values_cache = null; // Reset context
      break;
      
    case "playback_ended":
      isPlaying = false;
      break;
      
    case "text_message":
      await processTextMessage(text);
      break;
      
    case "audio":
      await processAudioBuffer(buffer);
      break;
      
    case "test":
      self.postMessage({ type: "test_response", message: "Worker is functioning correctly!" });
      break;
  }
};

/**
 * Process incoming audio buffer
 */
async function processAudioBuffer(buffer) {
  const wasRecording = isRecording;
  const isSpeech = await vad(buffer);

  if (!wasRecording && !isSpeech) {
    if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) {
      prevBuffers.shift();
    }
    prevBuffers.push(buffer);
    return;
  }

  const remaining = BUFFER.length - bufferPointer;
  if (buffer.length >= remaining) {
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
    bufferPointer += remaining;

    const overflow = buffer.subarray(remaining);
    dispatchForTranscriptionAndResetAudioBuffer(overflow);
    return;
  } else {
    BUFFER.set(buffer, bufferPointer);
    bufferPointer += buffer.length;
  }

  if (isSpeech) {
    if (!isRecording) {
      self.postMessage({
        type: "status",
        status: "recording_start",
        message: "Listening...",
      });
    }
    isRecording = true;
    postSpeechSamples = 0;
    return;
  }

  postSpeechSamples += buffer.length;

  if (postSpeechSamples < MIN_SILENCE_DURATION_SAMPLES) {
    return;
  }

  if (bufferPointer < MIN_SPEECH_DURATION_SAMPLES) {
    resetAfterRecording();
    return;
  }

  dispatchForTranscriptionAndResetAudioBuffer();
}
