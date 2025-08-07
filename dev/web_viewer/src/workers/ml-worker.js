/**
 * ML Worker for Voice Chat Interface
 * Based on conversational-webgpu worker.js patterns
 * Handles all ML model loading and inference in a dedicated worker
 */

// Import transformers from CDN directly since workers can't use importmaps
import {
  AutoModel,
  AutoTokenizer,
  AutoModelForCausalLM,
  pipeline,
  env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3/dist/transformers.min.js";

import { 
  DEFAULT_MODELS, 
  DEVICE_DTYPE_CONFIGS, 
  SYSTEM_PROMPTS,
  INPUT_SAMPLE_RATE 
} from "./constants.js";

// Global state
let models = {
  whisper: null,
  llama: null,
  vad: null,
  tokenizer: null
};

let device = 'wasm'; // Default to WASM for better compatibility
let isReady = false;
let conversation = [];

// Configure transformers environment
env.allowRemoteModels = false; // Force local models
env.allowLocalModels = true;
env.localModelPath = './models/';

/**
 * Initialize the worker
 */
async function initialize(config = {}) {
  try {
    device = config.device || 'wasm';
    
    self.postMessage({ 
      type: 'status', 
      message: `Initializing ML worker with device: ${device}` 
    });
    
    // Configure device-specific settings
    const deviceConfig = DEVICE_DTYPE_CONFIGS[device];
    
    // Load VAD model first (smaller, faster)
    self.postMessage({ type: 'status', message: 'Loading VAD model...' });
    models.vad = await AutoModel.from_pretrained(DEFAULT_MODELS.vad, {
      config: { model_type: "custom" },
      dtype: "fp32"
    });
    
    self.postMessage({ type: 'modelLoaded', model: 'vad' });
    
    // Load Whisper model
    self.postMessage({ type: 'status', message: 'Loading Whisper model...' });
    models.whisper = await pipeline(
      "automatic-speech-recognition",
      DEFAULT_MODELS.whisper,
      {
        device,
        dtype: deviceConfig
      }
    );
    
    // Compile shaders by running a dummy inference
    await models.whisper(new Float32Array(INPUT_SAMPLE_RATE));
    self.postMessage({ type: 'modelLoaded', model: 'whisper' });
    
    // Load LLM
    self.postMessage({ type: 'status', message: 'Loading language model...' });
    models.tokenizer = await AutoTokenizer.from_pretrained(DEFAULT_MODELS.llama);
    models.llama = await AutoModelForCausalLM.from_pretrained(DEFAULT_MODELS.llama, {
      dtype: deviceConfig.llm_dtype,
      device
    });
    
    // Compile LLM shaders
    await models.llama.generate({ 
      ...models.tokenizer("test"), 
      max_new_tokens: 1 
    });
    
    self.postMessage({ type: 'modelLoaded', model: 'llama' });
    
    // Initialize conversation
    conversation = [{
      role: "system",
      content: SYSTEM_PROMPTS.default
    }];
    
    isReady = true;
    self.postMessage({ 
      type: 'ready', 
      message: 'All models loaded successfully!',
      models: Object.keys(models).filter(k => models[k] !== null)
    });
    
  } catch (error) {
    self.postMessage({ 
      type: 'error', 
      error: error.message,
      details: error.toString()
    });
  }
}

/**
 * Process audio for speech recognition
 */
async function processAudio(audioBuffer) {
  if (!models.whisper) {
    throw new Error('Whisper model not loaded');
  }
  
  try {
    self.postMessage({ type: 'status', message: 'Transcribing audio...' });
    
    const result = await models.whisper(audioBuffer, {
      language: 'english',
      task: 'transcribe',
      return_timestamps: false
    });
    
    const transcription = result.text?.trim() || '';
    
    self.postMessage({
      type: 'transcription',
      text: transcription,
      confidence: result.confidence || 0.8
    });
    
    return transcription;
    
  } catch (error) {
    self.postMessage({ 
      type: 'error', 
      error: `Transcription failed: ${error.message}` 
    });
    throw error;
  }
}

/**
 * Generate text response
 */
async function generateResponse(userText) {
  if (!models.llama || !models.tokenizer) {
    throw new Error('Language model not loaded');
  }
  
  try {
    self.postMessage({ type: 'status', message: 'Generating response...' });
    
    // Add user message to conversation
    conversation.push({
      role: "user",
      content: userText
    });
    
    // Format conversation for the model
    const prompt = conversation.map(msg => {
      if (msg.role === 'system') return `<|system|>${msg.content}`;
      if (msg.role === 'user') return `<|user|>${msg.content}`;
      if (msg.role === 'assistant') return `<|assistant|>${msg.content}`;
      return msg.content;
    }).join('\n') + '\n<|assistant|>';
    
    // Tokenize
    const inputs = models.tokenizer(prompt);
    
    // Generate response
    const outputs = await models.llama.generate({
      ...inputs,
      max_new_tokens: 150,
      temperature: 0.8,
      top_p: 0.9,
      do_sample: true,
      pad_token_id: models.tokenizer.eos_token_id
    });
    
    // Decode response
    const responseIds = outputs.sequences[0].slice(inputs.input_ids.dims[1]);
    const responseText = models.tokenizer.decode(responseIds, { skip_special_tokens: true });
    
    // Add to conversation
    conversation.push({
      role: "assistant", 
      content: responseText
    });
    
    // Limit conversation history
    if (conversation.length > 10) {
      conversation = [conversation[0], ...conversation.slice(-8)]; // Keep system + last 8
    }
    
    self.postMessage({
      type: 'response',
      text: responseText,
      conversationLength: conversation.length
    });
    
    return responseText;
    
  } catch (error) {
    self.postMessage({ 
      type: 'error', 
      error: `Response generation failed: ${error.message}` 
    });
    throw error;
  }
}

/**
 * Process voice activity detection
 */
async function processVAD(audioBuffer) {
  if (!models.vad) {
    // Fallback to energy-based VAD
    return simpleEnergyVAD(audioBuffer);
  }
  
  try {
    // TODO: Implement Silero VAD processing
    // For now, use simple energy-based detection
    return simpleEnergyVAD(audioBuffer);
    
  } catch (error) {
    return simpleEnergyVAD(audioBuffer);
  }
}

/**
 * Simple energy-based VAD fallback
 */
function simpleEnergyVAD(audioBuffer) {
  let sum = 0;
  for (let i = 0; i < audioBuffer.length; i++) {
    sum += audioBuffer[i] * audioBuffer[i];
  }
  const rms = Math.sqrt(sum / audioBuffer.length);
  return rms > 0.01; // Simple threshold
}

/**
 * Reset conversation
 */
function resetConversation() {
  conversation = [{
    role: "system",
    content: SYSTEM_PROMPTS.default
  }];
  
  self.postMessage({ 
    type: 'conversationReset',
    message: 'Conversation history cleared'
  });
}

/**
 * Get model status
 */
function getStatus() {
  const status = {
    ready: isReady,
    device,
    models: {}
  };
  
  Object.keys(models).forEach(key => {
    status.models[key] = models[key] !== null;
  });
  
  self.postMessage({ type: 'status', status });
}

// Message handler
self.onmessage = async function(event) {
  const { type, data } = event.data;
  
  try {
    switch (type) {
      case 'initialize':
        await initialize(data);
        break;
        
      case 'processAudio':
        await processAudio(data.audioBuffer);
        break;
        
      case 'generateResponse':
        await generateResponse(data.text);
        break;
        
      case 'processVAD':
        const isVoice = await processVAD(data.audioBuffer);
        self.postMessage({ type: 'vadResult', isVoice });
        break;
        
      case 'resetConversation':
        resetConversation();
        break;
        
      case 'getStatus':
        getStatus();
        break;
        
      case 'setSystemPrompt':
        if (conversation.length > 0) {
          conversation[0].content = data.prompt;
        }
        break;
        
      default:
        self.postMessage({ 
          type: 'error', 
          error: `Unknown message type: ${type}` 
        });
    }
    
  } catch (error) {
    self.postMessage({ 
      type: 'error', 
      error: error.message,
      details: error.toString(),
      messageType: type
    });
  }
};

// Initialize on startup
self.postMessage({ 
  type: 'workerReady', 
  message: 'ML Worker initialized and ready' 
});

// Global error handler for the worker
self.addEventListener('error', (event) => {
  console.error('Worker global error:', event);
  self.postMessage({
    type: 'error',
    error: `Worker error: ${event.message || 'Unknown error'}`,
    details: {
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      error: event.error
    }
  });
});

// Unhandled promise rejection handler
self.addEventListener('unhandledrejection', (event) => {
  console.error('Worker unhandled promise rejection:', event);
  self.postMessage({
    type: 'error',
    error: `Unhandled promise rejection: ${event.reason}`,
    details: event.reason
  });
});
