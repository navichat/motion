/**
 * Incremental Audio-to-Audio Conversation Worker
 * Loads models one by one with detailed error reporting
 */

// Import with error handling
let transformers = null;
let kokorojs = null;

// Try to load transformers
try {
  const { 
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    InterruptableStoppingCriteria,
    Tensor,
    pipeline,
  } = await import("@huggingface/transformers");
  
  transformers = {
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    InterruptableStoppingCriteria,
    Tensor,
    pipeline,
  };
  
  self.postMessage({ type: "info", message: "✅ Transformers.js loaded successfully" });
} catch (error) {
  self.postMessage({ type: "info", message: `❌ Failed to load transformers: ${error.message}` });
}

// Constants
const INPUT_SAMPLE_RATE = 16000;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const MIN_SILENCE_DURATION_SAMPLES = 400 * INPUT_SAMPLE_RATE / 1000;
const SPEECH_PAD_SAMPLES = 80 * INPUT_SAMPLE_RATE / 1000;
const MIN_SPEECH_DURATION_SAMPLES = 250 * INPUT_SAMPLE_RATE / 1000;
const MAX_BUFFER_DURATION = 30;

// Global state
let voice = "af_heart";
let silero_vad = null;
let transcriber = null;
let llm = null;
let tokenizer = null;
let tts = null;
let messages = [];
let past_key_values_cache = null;
let stopping_criteria = null;

// Audio processing state
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;
let state = null;
let isRecording = false;
let isPlaying = false;
let callActive = false;
let postSpeechSamples = 0;

// Default voices
const defaultVoices = {
  "af_heart": { name: "Heart", language: "en-us", gender: "female" },
  "am_adam": { name: "Adam", language: "en-us", gender: "male" },
  "af_sarah": { name: "Sarah", language: "en-us", gender: "female" }
};

// Initialize models incrementally
async function initializeModels() {
  try {
    self.postMessage({ type: "info", message: "🚀 Starting incremental model loading..." });
    
    if (!transformers) {
      throw new Error("Transformers.js not available");
    }

    // Step 1: Load VAD
    try {
      self.postMessage({ type: "info", message: "Loading Silero VAD..." });
      silero_vad = await transformers.AutoModel.from_pretrained("onnx-community/silero-vad", {
        config: { model_type: "custom" },
        dtype: "fp32",
      });
      state = new transformers.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
      self.postMessage({ type: "info", message: "✅ VAD loaded successfully" });
    } catch (vadError) {
      self.postMessage({ type: "info", message: `❌ VAD loading failed: ${vadError.message}` });
      throw vadError;
    }

    // Step 2: Load Whisper
    try {
      self.postMessage({ type: "info", message: "Loading Whisper STT..." });
      transcriber = await transformers.pipeline("automatic-speech-recognition", "onnx-community/whisper-base", {
        device: "webgpu",
        dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" },
      });
      
      // Compile shaders
      await transcriber(new Float32Array(INPUT_SAMPLE_RATE));
      self.postMessage({ type: "info", message: "✅ Whisper loaded successfully" });
    } catch (whisperError) {
      self.postMessage({ type: "info", message: `❌ Whisper loading failed: ${whisperError.message}` });
      throw whisperError;
    }

    // Step 3: Load SmolLM2
    try {
      self.postMessage({ type: "info", message: "Loading SmolLM2..." });
      tokenizer = await transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct");
      llm = await transformers.AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", {
        dtype: "q4f16",
        device: "webgpu",
      });
      
      // Compile shaders
      await llm.generate({ ...tokenizer("test"), max_new_tokens: 1 });
      self.postMessage({ type: "info", message: "✅ SmolLM2 loaded successfully" });
    } catch (llmError) {
      self.postMessage({ type: "info", message: `❌ SmolLM2 loading failed: ${llmError.message}` });
      throw llmError;
    }

    // Step 4: Try to load TTS (optional)
    try {
      self.postMessage({ type: "info", message: "Loading Kokoro TTS..." });
      
      // Use direct CDN import
      const response = await fetch("https://cdn.jsdelivr.net/npm/kokoro-js@0.2.1/dist/index.js");
      const kokoroCode = await response.text();
      
      // Evaluate the module code (fallback method)
      const moduleFunc = new Function('exports', 'module', 'require', kokoroCode + '; return { KokoroTTS, TextSplitterStream };');
      const kokoro = moduleFunc({}, { exports: {} }, () => {});
      
      if (kokoro.KokoroTTS) {
        tts = await kokoro.KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
          dtype: "fp32",
          device: "webgpu",
        });
        kokorojs = kokoro;
        self.postMessage({ type: "info", message: "✅ Kokoro TTS loaded successfully" });
      } else {
        throw new Error("KokoroTTS not found in module");
      }
    } catch (ttsError) {
      self.postMessage({ type: "info", message: `⚠️ TTS loading failed: ${ttsError.message}` });
      self.postMessage({ type: "info", message: "Will use Web Speech API fallback" });
    }

    // Initialize conversation
    const SYSTEM_MESSAGE = {
      role: "system",
      content: "You're a helpful and conversational AI avatar assistant. Keep your responses short, clear, and casual. You are speaking through voice, so be natural and engaging.",
    };
    messages = [SYSTEM_MESSAGE];

    // Send ready status
    const voices = tts?.voices || defaultVoices;
    
    self.postMessage({
      type: "status",
      status: "ready",
      message: "AI models loaded successfully!",
      voices: voices
    });

  } catch (error) {
    self.postMessage({ error: `Model initialization failed: ${error.message}` });
    console.error('Full error:', error);
  }
}

// Voice activity detection
function vad(audioData) {
  try {
    if (!silero_vad || !state || !transformers) return false;
    
    const inputs = {
      input: new transformers.Tensor("float32", audioData, [1, audioData.length]),
      state: state,
      sample_rate: new transformers.Tensor("int64", [INPUT_SAMPLE_RATE]),
    };

    const { output: speech_prob, state: new_state } = silero_vad(inputs);
    state = new_state;

    const isSpeech = speech_prob.data[0];
    return isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD);
  } catch (error) {
    console.error('VAD error:', error);
    return false;
  }
}

// Process incoming audio buffer
function processAudioBuffer(audioData) {
  if (!callActive || isPlaying) return;

  const buffer = new Float32Array(audioData);
  
  // Add to circular buffer
  const remaining = BUFFER.length - bufferPointer;
  if (buffer.length <= remaining) {
    BUFFER.set(buffer, bufferPointer);
    bufferPointer += buffer.length;
  } else {
    BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
    bufferPointer = 0;
    BUFFER.set(buffer.subarray(remaining), bufferPointer);
    bufferPointer += buffer.length - remaining;
  }

  // Check for voice activity
  const hasSpeech = vad(buffer);
  
  if (!isRecording && hasSpeech) {
    // Start recording
    isRecording = true;
    postSpeechSamples = 0;
    self.postMessage({ type: "status", status: "recording_start", message: "Listening..." });
  } else if (isRecording) {
    if (hasSpeech) {
      postSpeechSamples = 0;
    } else {
      postSpeechSamples += buffer.length;
      
      if (postSpeechSamples >= MIN_SILENCE_DURATION_SAMPLES) {
        // End recording
        isRecording = false;
        self.postMessage({ type: "status", status: "recording_end", message: "Processing..." });
        
        // Extract recorded audio and process
        const audioToProcess = extractRecordedAudio();
        if (audioToProcess.length >= MIN_SPEECH_DURATION_SAMPLES) {
          speechToSpeech(audioToProcess);
        }
      }
    }
  }
}

// Extract recorded audio from buffer
function extractRecordedAudio() {
  const samplesToExtract = Math.min(3 * INPUT_SAMPLE_RATE, bufferPointer);
  const startIndex = Math.max(0, bufferPointer - samplesToExtract);
  
  const audio = new Float32Array(samplesToExtract);
  for (let i = 0; i < samplesToExtract; i++) {
    audio[i] = BUFFER[startIndex + i];
  }
  
  return audio;
}

// Complete speech-to-speech pipeline
async function speechToSpeech(audioData) {
  try {
    if (!transcriber || !llm || !tokenizer) {
      throw new Error("Required models not loaded");
    }

    self.postMessage({ type: "info", message: "🎤 Processing speech..." });
    
    // Speech-to-text using Whisper
    const transcription = await transcriber(audioData);
    const userText = transcription.text.trim();
    
    if (!userText) {
      self.postMessage({ type: "info", message: "No speech detected" });
      return;
    }

    self.postMessage({ type: "info", message: `User said: "${userText}"` });
    messages.push({ role: "user", content: userText });

    // Generate AI response using SmolLM2
    self.postMessage({ type: "info", message: "🧠 Generating response..." });
    const inputs = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
    });

    stopping_criteria = new transformers.InterruptableStoppingCriteria();
    const outputs = await llm.generate({
      ...inputs,
      max_new_tokens: 100,
      do_sample: false,
      temperature: 0.7,
      stopping_criteria,
      return_dict_in_generate: true,
      past_key_values: past_key_values_cache,
    });

    past_key_values_cache = outputs.past_key_values;
    const response_tokens = outputs.sequences.slice(0, [1, -inputs.input_ids.dims.slice(-1)[0]]);
    const responseText = tokenizer.decode(response_tokens[0], { skip_special_tokens: true }).trim();

    self.postMessage({ type: "info", message: `AI response: "${responseText}"` });
    messages.push({ role: "assistant", content: responseText });

    // Generate audio response
    const audioResult = await generateAudio(responseText);
    
    // Send the complete result
    self.postMessage({
      type: "output",
      text: responseText,
      result: audioResult
    });

  } catch (error) {
    console.error('Speech processing error:', error);
    self.postMessage({ error: `Speech processing failed: ${error.message}` });
  }
}

// Generate audio using TTS
async function generateAudio(text) {
  try {
    if (tts && kokorojs) {
      self.postMessage({ type: "info", message: "🔊 Generating TTS audio..." });
      
      const splitter = new kokorojs.TextSplitterStream();
      const stream = tts.stream(splitter, { voice });
      
      const audioChunks = [];
      splitter.push(text);
      splitter.close();
      
      for await (const { audio } of stream) {
        if (audio && audio.length > 0) {
          audioChunks.push(audio);
        }
      }
      
      if (audioChunks.length > 0) {
        const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const combinedAudio = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of audioChunks) {
          combinedAudio.set(chunk, offset);
          offset += chunk.length;
        }
        return { audio: combinedAudio };
      }
    }
    
    // Fallback to Web Speech API
    return { audio: { useWebSpeech: true, text: text } };
    
  } catch (error) {
    console.error('TTS generation error:', error);
    return { audio: { useWebSpeech: true, text: text } };
  }
}

// Main message handler
self.onmessage = async (event) => {
  try {
    const { type } = event.data;

    switch (type) {
      case "start_call":
        callActive = true;
        isPlaying = true; // Prevent processing during greeting
        const greeting = "Hello! I'm your AI assistant. How can I help you today?";
        const greetingAudio = await generateAudio(greeting);
        self.postMessage({ 
          type: "output", 
          text: greeting, 
          result: greetingAudio 
        });
        break;
        
      case "end_call":
        callActive = false;
        isRecording = false;
        isPlaying = false;
        bufferPointer = 0;
        past_key_values_cache = null;
        messages = messages.slice(0, 1); // Keep only system message
        self.postMessage({ type: "info", message: "Call ended" });
        break;
        
      case "set_voice":
        voice = event.data.voice || "af_heart";
        self.postMessage({ type: "info", message: `Voice changed to: ${voice}` });
        break;
        
      case "audio":
        if (event.data.data) {
          processAudioBuffer(event.data.data);
        }
        break;
        
      case "playback_started":
        isPlaying = true;
        break;
        
      case "playback_ended":
        isPlaying = false;
        self.postMessage({ type: "info", message: "Ready for next input" });
        break;
        
      default:
        self.postMessage({ type: "info", message: `Unknown message type: ${type}` });
    }
  } catch (error) {
    console.error('Message handler error:', error);
    self.postMessage({ error: `Message handler error: ${error.message}` });
  }
};

// Error handlers
self.addEventListener('error', (error) => {
  console.error('Worker error:', error);
  self.postMessage({ error: `Worker error: ${error.message || error.toString()}` });
});

self.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event.reason);
  self.postMessage({ error: `Promise rejection: ${event.reason}` });
});

// Initialize
self.postMessage({ type: "info", message: "🚀 Starting incremental conversation worker..." });
initializeModels();
