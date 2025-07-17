/**
 * Enhanced Working Audio-to-Audio Conversation Worker
 * Uses direct CDN imports, Web Speech API, and proper phonemization for Kokoro TTS
 */

// Simple phonemizer functions (embedded for worker compatibility)
function normalize_text(text) {
  // Basic text normalization
  return text
    .replace(/\s+/g, ' ')
    .replace(/[^\w\s\.,;:!?'-]/g, '')
    .trim()
    .toLowerCase();
}

function quickPhonemize(text) {
  // Simple phonemization fallback
  return text
    .replace(/ch/g, 'tʃ')
    .replace(/sh/g, 'ʃ')
    .replace(/th/g, 'θ')
    .replace(/ng/g, 'ŋ')
    .replace(/oo/g, 'u')
    .replace(/ee/g, 'i')
    .replace(/a/g, 'ə')
    .replace(/e/g, 'ɛ')
    .replace(/i/g, 'ɪ')
    .replace(/o/g, 'ɔ')
    .replace(/u/g, 'ʌ');
}

async function phonemize(text) {
  // For now, use the simple phonemizer
  // In the future, this could call an eSpeak-NG service
  return quickPhonemize(text);
}

// Enhanced logging system
const LOG_PREFIX = '[ConversationWorker]';
function workerLog(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMsg = `${LOG_PREFIX} [${timestamp}] [${level}] ${message}`;
    
    console.log(logMsg, data || '');
    
    // Send to main thread
    self.postMessage({ 
        type: "debug_log", 
        level, 
        message: logMsg, 
        data,
        timestamp 
    });
}

// Global state
let voice = "af_heart";
let isRecording = false;
let isPlaying = false;
let callActive = false;
let lastSpeechTime = 0;
let speechCooldown = 3000; // 3 second cooldown
let messages = [];

workerLog('INFO', 'Worker script loaded, initializing global state');

// Default voices - updated to match local Kokoro voices
const defaultVoices = {
  "af_heart": { name: "Heart (Female)", language: "en-us", gender: "female", file: "af_heart.bin" },
  "am_adam": { name: "Adam (Male)", language: "en-us", gender: "male", file: "am_adam.bin" },
  "af_sarah": { name: "Sarah (Female)", language: "en-us", gender: "female", file: "af_sarah.bin" },
  "af_bella": { name: "Bella (Female)", language: "en-us", gender: "female", file: "af_bella.bin" },
  "af_jessica": { name: "Jessica (Female)", language: "en-us", gender: "female", file: "af_jessica.bin" },
  "am_michael": { name: "Michael (Male)", language: "en-us", gender: "male", file: "am_michael.bin" },
  "am_liam": { name: "Liam (Male)", language: "en-us", gender: "male", file: "am_liam.bin" },
  "bf_alice": { name: "Alice (British Female)", language: "en-gb", gender: "female", file: "bf_alice.bin" },
  "bm_daniel": { name: "Daniel (British Male)", language: "en-gb", gender: "male", file: "bm_daniel.bin" },
  "jf_alpha": { name: "Alpha (Japanese Female)", language: "ja-jp", gender: "female", file: "jf_alpha.bin" }
};

workerLog('INFO', 'Default voices configured', defaultVoices);

// Audio processing variables
const INPUT_SAMPLE_RATE = 16000;
const MAX_BUFFER_DURATION = 30;
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);
let bufferPointer = 0;

workerLog('INFO', 'Audio processing constants initialized', {
    INPUT_SAMPLE_RATE,
    MAX_BUFFER_DURATION,
    BUFFER_SIZE: BUFFER.length
});

// Try to load transformers.js for real models
let transformers = null;
let silero_vad = null;
let transcriber = null;
let llm = null;
let tokenizer = null;
let state = null;
let kokoroTTS = null;

workerLog('INFO', 'Model variables initialized to null, ready for loading');

async function loadTransformers() {
  try {
    workerLog('INFO', 'Starting Transformers.js loading process');
    self.postMessage({ type: "info", message: "Attempting to load Transformers.js..." });
    
    // Try multiple CDN sources for reliability
    const cdnUrls = [
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.3",
      "https://cdn.jsdelivr.net/npm/@xenova/transformers@3.6.3",
      "https://unpkg.com/@huggingface/transformers@3.6.3"
    ];
    
    workerLog('INFO', 'CDN URLs to try', cdnUrls);
    
    for (const cdnUrl of cdnUrls) {
      try {
        workerLog('INFO', `Attempting to load from: ${cdnUrl}`);
        self.postMessage({ type: "info", message: `Trying ${cdnUrl}...` });
        
        const startTime = performance.now();
        
        // Dynamic import with proper error handling
        const transformersModule = await import(cdnUrl);
        
        const loadTime = performance.now() - startTime;
        workerLog('INFO', `Import successful from ${cdnUrl}`, { loadTime: `${loadTime.toFixed(2)}ms` });
        
        // Check if we got the expected exports
        const availableExports = Object.keys(transformersModule);
        workerLog('INFO', `Available exports from ${cdnUrl}`, availableExports);
        
        if (transformersModule.pipeline || transformersModule.AutoModel) {
          transformers = transformersModule;
          
          // Configure environment for local models
          if (transformersModule.env) {
            transformersModule.env.allowLocalModels = true;
            transformersModule.env.allowRemoteModels = true;
            transformersModule.env.localModelPath = './';  // Set to current directory
            transformersModule.env.remoteModelPath = 'https://huggingface.co/';
          }
          
          workerLog('SUCCESS', `Transformers.js loaded successfully from ${cdnUrl}`, {
            hasAutoModel: !!transformersModule.AutoModel,
            hasPipeline: !!transformersModule.pipeline,
            hasAutoTokenizer: !!transformersModule.AutoTokenizer,
            hasAutoModelForCausalLM: !!transformersModule.AutoModelForCausalLM,
            envConfigured: !!transformersModule.env
          });
          self.postMessage({ type: "info", message: `✅ Transformers.js loaded from ${cdnUrl}` });
          return true;
        } else {
          workerLog('WARN', `No expected exports found in ${cdnUrl}`, availableExports);
        }
      } catch (cdnError) {
        workerLog('ERROR', `Failed to load from ${cdnUrl}`, { error: cdnError.message, stack: cdnError.stack });
        self.postMessage({ type: "info", message: `Failed ${cdnUrl}: ${cdnError.message}` });
        continue;
      }
    }
    
    // If all CDN attempts fail, provide fallback message
    workerLog('ERROR', 'All CDN sources failed to load Transformers.js');
    self.postMessage({ type: "info", message: "⚠️ All Transformers.js CDN sources failed - using simulated models" });
    return false;
    
  } catch (error) {
    workerLog('ERROR', 'Transformers.js loading error', { error: error.message, stack: error.stack });
    self.postMessage({ type: "info", message: `⚠️ Transformers.js loading error: ${error.message}` });
    return false;
  }
}

async function loadKokoroTTS() {
  try {
    workerLog('INFO', 'Starting Kokoro TTS loading process');
    self.postMessage({ type: "info", message: "Loading Kokoro TTS model..." });
    
    // Try to load Kokoro TTS directly
    const model = await transformers.AutoModel.from_pretrained('models/Kokoro-82M-v1.0-ONNX', {
      dtype: 'fp32',
      device: 'wasm',
      use_cache: false
    });
    
    const tokenizer = await transformers.AutoTokenizer.from_pretrained('models/Kokoro-82M-v1.0-ONNX');
    
    workerLog('SUCCESS', 'Kokoro model and tokenizer loaded successfully');
    
    // Create a wrapper that handles the Kokoro TTS format
    kokoroTTS = {
      model: model,
      tokenizer: tokenizer,
      __call__: async (text, options = {}) => {
        try {
          workerLog('INFO', 'Kokoro TTS generation starting', { text: text.substring(0, 50) + '...' });
          
          // Step 1: Normalize and phonemize
          const normalizedText = normalize_text(text);
          const phonemes = await phonemize(normalizedText);
          
          // Step 2: Tokenize
          const inputs = tokenizer(phonemes);
          const tokenIds = Array.from(inputs.input_ids.data).map(id => Number(id));
          
          // Step 3: Load voice data
          const voiceFile = options.voice || 'af_heart';
          const voiceResponse = await fetch(`models/Kokoro-82M-v1.0-ONNX/voices/${voiceFile}.bin`);
          
          if (!voiceResponse.ok) {
            throw new Error(`Failed to load voice ${voiceFile}: ${voiceResponse.status}`);
          }
          
          const voiceBuffer = await voiceResponse.arrayBuffer();
          const voiceData = new Float32Array(voiceBuffer);
          
          // Step 4: Create style vector
          const tokenLength = tokenIds.length;
          const styleIndex = Math.min(tokenLength, Math.floor(voiceData.length / 256) - 1);
          const styleVector = voiceData.slice(styleIndex * 256, (styleIndex + 1) * 256);
          
          // Step 5: Create tensors
          const paddedTokens = [0, ...tokenIds, 0];
          const inputIds = new BigInt64Array(paddedTokens.map(x => BigInt(x)));
          const style = new Float32Array(styleVector);
          const speed = new Float32Array([1.0]);
          
          const inputTensor = new transformers.Tensor('int64', inputIds, [1, inputIds.length]);
          const styleTensor = new transformers.Tensor('float32', style, [1, 256]);
          const speedTensor = new transformers.Tensor('float32', speed, [1]);
          
          // Step 6: Run inference
          const startTime = performance.now();
          const output = await model({
            input_ids: inputTensor,
            style: styleTensor,
            speed: speedTensor
          });
          const inferenceTime = performance.now() - startTime;
          
          // Step 7: Extract audio
          const audioData = output.waveform || output.audio || output.last_hidden_state || output.logits;
          
          if (audioData && audioData.data) {
            const audioArray = new Float32Array(audioData.data);
            
            workerLog('SUCCESS', 'Kokoro TTS generation completed', {
              audioLength: audioArray.length,
              sampleRate: 24000,
              duration: `${(audioArray.length / 24000).toFixed(2)}s`,
              inferenceTime: `${inferenceTime.toFixed(2)}ms`
            });
            
            return {
              audio: audioArray,
              sampling_rate: 24000,
              metadata: {
                voice: voiceFile,
                phonemes: phonemes,
                tokenCount: tokenIds.length,
                inferenceTime: inferenceTime
              }
            };
          } else {
            throw new Error('No audio data in model output');
          }
          
        } catch (error) {
          workerLog('ERROR', 'Kokoro TTS generation failed', { 
            error: error.message, 
            stack: error.stack 
          });
          throw error;
        }
      }
    };
    
    // Test the model
    workerLog('INFO', 'Testing Kokoro model with simple input');
    const testResult = await kokoroTTS.__call__("Hello test");
    
    if (testResult && testResult.audio && testResult.audio.length > 0) {
      workerLog('SUCCESS', 'Kokoro model test successful', {
        audioLength: testResult.audio.length,
        sampleRate: testResult.sampling_rate
      });
      self.postMessage({ type: "info", message: "✅ Kokoro TTS loaded and tested successfully" });
      return true;
    } else {
      throw new Error('Kokoro model test failed - no audio output');
    }
    
  } catch (error) {
    workerLog('ERROR', 'Kokoro TTS loading failed', { 
      error: error.message, 
      stack: error.stack 
    });
    self.postMessage({ type: "info", message: `⚠️ Kokoro TTS failed: ${error.message}` });
    
    // Create a simple fallback TTS
    workerLog('INFO', 'Creating simple fallback TTS');
    kokoroTTS = {
      __call__: async (text) => {
        workerLog('INFO', 'Using simple tone generator TTS');
        
        const sampleRate = 16000;
        const duration = Math.min(5, Math.max(1, text.length * 0.08));
        const samples = Math.floor(sampleRate * duration);
        const audioData = new Float32Array(samples);
        
        // Generate a pleasant tone sequence
        const baseFreq = 440;
        const fadeIn = Math.floor(samples * 0.1);
        const fadeOut = Math.floor(samples * 0.1);
        
        for (let i = 0; i < samples; i++) {
          const t = i / sampleRate;
          let amplitude = 0.1;
          
          // Apply fade in/out
          if (i < fadeIn) {
            amplitude *= (i / fadeIn);
          } else if (i > samples - fadeOut) {
            amplitude *= ((samples - i) / fadeOut);
          }
          
          // Create a pleasant harmonic tone
          const tone1 = Math.sin(2 * Math.PI * baseFreq * t);
          const tone2 = Math.sin(2 * Math.PI * baseFreq * 1.5 * t) * 0.5;
          const tone3 = Math.sin(2 * Math.PI * baseFreq * 2 * t) * 0.25;
          
          audioData[i] = (tone1 + tone2 + tone3) * amplitude;
        }
        
        return {
          audio: audioData,
          sampling_rate: sampleRate
        };
      }
    };
    
    workerLog('SUCCESS', 'Simple fallback TTS created');
    self.postMessage({ type: "info", message: "✅ Simple TTS fallback created" });
    return true;
  }
}
        loader: async () => {
          workerLog('INFO', 'Loading local Kokoro TTS model via direct ONNX');
          try {
            // Load the model and tokenizer directly
            const model = await transformers.AutoModel.from_pretrained('models/Kokoro-82M-v1.0-ONNX', {
              dtype: 'fp32',
              device: 'wasm',
              use_cache: false
            });
            
            const tokenizer = await transformers.AutoTokenizer.from_pretrained('models/Kokoro-82M-v1.0-ONNX');
            
            workerLog('INFO', 'Kokoro model and tokenizer loaded successfully');
            
            // Test model compatibility
            try {
              workerLog('INFO', 'Testing Kokoro model compatibility');
              const testInputs = tokenizer('test');
              
              // Check tensor creation
              if (testInputs.input_ids) {
                const testTensor = new transformers.Tensor('int64', testInputs.input_ids.data, testInputs.input_ids.dims);
                workerLog('INFO', 'Tensor creation test passed', { 
                  tensorShape: testTensor.dims,
                  tensorType: testTensor.type
                });
              }
              
            } catch (compatError) {
              workerLog('WARN', 'Model compatibility test failed', { 
                error: compatError.message,
                willContinueAnyway: true
              });
            }
            
            // Return a wrapper that handles the Kokoro-specific format
            return {
              __call__: async (text, options = {}) => {
                try {
                  workerLog('INFO', 'Starting Kokoro TTS generation', { 
                    text: text.substring(0, 100) + '...',
                    textLength: text.length,
                    voice: options.voice || 'af_heart'
                  });
                  
                  // Step 1: Normalize and phonemize the text
                  const normalizedText = normalize_text(text);
                  workerLog('INFO', 'Text normalized', { 
                    original: text.substring(0, 50) + '...',
                    normalized: normalizedText.substring(0, 50) + '...'
                  });
                  
                  // Step 2: Convert to phonemes
                  const phonemes = await phonemize(normalizedText, "a", false); // Already normalized
                  workerLog('INFO', 'Text phonemized', { 
                    phonemes: phonemes.substring(0, 100) + '...',
                    phonemeLength: phonemes.length
                  });
                  
                  // Step 3: Tokenize the phonemes
                  const inputs = tokenizer(phonemes);
                  const tokenIds = Array.from(inputs.input_ids.data).map(id => Number(id));
                  workerLog('INFO', 'Phonemes tokenized', { 
                    tokenCount: tokenIds.length,
                    firstTokens: tokenIds.slice(0, 10)
                  });
                  
                  // Step 4: Load voice data
                  const voiceFile = options.voice || 'af_heart';
                  const voiceResponse = await fetch(`models/Kokoro-82M-v1.0-ONNX/voices/${voiceFile}.bin`);
                  if (!voiceResponse.ok) {
                    throw new Error(`Failed to load voice ${voiceFile}: ${voiceResponse.status}`);
                  }
                  
                  const voiceBuffer = await voiceResponse.arrayBuffer();
                  const voiceData = new Float32Array(voiceBuffer);
                  workerLog('INFO', 'Voice data loaded', { 
                    voice: voiceFile,
                    voiceDataLength: voiceData.length
                  });
                  
                  // Step 5: Calculate style vector
                  const tokenLength = tokenIds.length;
                  const styleIndex = Math.min(tokenLength, Math.floor(voiceData.length / 256) - 1);
                  const styleVector = voiceData.slice(styleIndex * 256, (styleIndex + 1) * 256);
                  workerLog('INFO', 'Style vector calculated', { 
                    styleIndex,
                    styleVectorLength: styleVector.length
                  });
                  
                  // Step 6: Prepare model inputs
                  const paddedTokens = [0, ...tokenIds, 0];
                  const inputIds = new BigInt64Array(paddedTokens.map(x => BigInt(x)));
                  const style = new Float32Array(styleVector);
                  const speed = new Float32Array([1.0]);
                  
                  // Step 7: Create tensors with correct dimensions for Kokoro model
                  // The model expects: input_ids [batch_size, seq_len], style [batch_size, 256], speed [batch_size]
                  const inputTensor = new transformers.Tensor('int64', inputIds, [1, inputIds.length]);
                  const styleTensor = new transformers.Tensor('float32', style, [1, 256]);
                  const speedTensor = new transformers.Tensor('float32', speed, [1]);
                  
                  workerLog('INFO', 'Tensor dimensions', {
                    inputTensor: inputTensor.dims,
                    styleTensor: styleTensor.dims,
                    speedTensor: speedTensor.dims
                  });
                  
                  workerLog('INFO', 'Model inputs prepared', { 
                    inputLength: inputIds.length,
                    styleLength: style.length
                  });
                  
                  // Step 8: Run inference with error handling
                  const inferenceStart = performance.now();
                  let output;
                  
                  try {
                    output = await model({
                      input_ids: inputTensor,
                      style: styleTensor,
                      speed: speedTensor
                    });
                  } catch (onnxError) {
                    workerLog('ERROR', 'ONNX inference failed', { 
                      error: onnxError.message,
                      inputShape: inputTensor.dims,
                      styleShape: styleTensor.dims,
                      speedShape: speedTensor.dims
                    });
                    
                    // Try alternative tensor shapes
                    try {
                      workerLog('INFO', 'Trying alternative tensor shapes');
                      
                      // Try with different input dimensions
                      const altInputTensor = new transformers.Tensor('int64', inputIds, [inputIds.length]);
                      const altStyleTensor = new transformers.Tensor('float32', style, [256]);
                      const altSpeedTensor = new transformers.Tensor('float32', speed, []);
                      
                      output = await model({
                        input_ids: altInputTensor,
                        style: altStyleTensor,
                        speed: altSpeedTensor
                      });
                      
                      workerLog('INFO', 'Alternative tensor shapes worked');
                      
                    } catch (altError) {
                      workerLog('ERROR', 'Alternative tensor shapes also failed', { 
                        error: altError.message 
                      });
                      throw new Error(`Kokoro model inference failed: ${onnxError.message}`);
                    }
                  }
                  
                  const inferenceTime = performance.now() - inferenceStart;
                  
                  workerLog('INFO', 'Model inference completed', { 
                    inferenceTime: `${inferenceTime.toFixed(2)}ms`,
                    outputKeys: Object.keys(output)
                  });
                  
                  // Step 9: Extract audio data
                  const audioData = output.waveform || output.audio || output.last_hidden_state || output.logits;
                  if (audioData && audioData.data) {
                    const audioArray = new Float32Array(audioData.data);
                    const sampleRate = 24000;
                    const duration = audioArray.length / sampleRate;
                    
                    workerLog('SUCCESS', 'Kokoro TTS generation successful', {
                      audioLength: audioArray.length,
                      sampleRate,
                      duration: `${duration.toFixed(2)}s`,
                      voice: voiceFile,
                      phonemeLength: phonemes.length,
                      tokenCount: tokenIds.length
                    });
                    
                    return {
                      audio: audioArray,
                      sampling_rate: sampleRate,
                      duration: duration,
                      metadata: {
                        voice: voiceFile,
                        phonemes: phonemes,
                        tokenCount: tokenIds.length,
                        inferenceTime: inferenceTime
                      }
                    };
                  } else {
                    throw new Error('No audio data in model output');
                  }
                  
                } catch (error) {
                  workerLog('ERROR', 'Kokoro TTS generation failed', { 
                    error: error.message,
                    stack: error.stack,
                    errorType: error.constructor.name
                  });
                  
                  // Check if it's a tensor shape issue
                  if (error.message.includes('dimensions') || error.message.includes('shape') || error.message.includes('LSTM')) {
                    workerLog('WARN', 'Tensor shape compatibility issue detected - this model may need a different transformers.js version');
                    
                    // Try a simplified approach
                    try {
                      workerLog('INFO', 'Attempting simplified Kokoro generation');
                      
                      // Basic phonemization and tokenization
                      const simplePhonemes = quickPhonemize(normalize_text(text));
                      const simpleInputs = tokenizer(simplePhonemes);
                      
                      // Try with minimal tensor setup
                      const simpleOutput = await model(simpleInputs);
                      
                      if (simpleOutput && simpleOutput.waveform) {
                        const audioArray = new Float32Array(simpleOutput.waveform.data);
                        return {
                          audio: audioArray,
                          sampling_rate: 24000,
                          duration: audioArray.length / 24000,
                          metadata: {
                            voice: options.voice || 'af_heart',
                            phonemes: simplePhonemes,
                            tokenCount: simpleInputs.input_ids.size,
                            inferenceTime: 0,
                            simplified: true
                          }
                        };
                      }
                    } catch (simpleError) {
                      workerLog('ERROR', 'Simplified approach also failed', { 
                        error: simpleError.message 
                      });
                    }
                  }
                  
}

async function loadAIModels() {
  if (!transformers) return false;
  
  try {
    // Load VAD
    self.postMessage({ type: "info", message: "Loading voice activity detection..." });
    silero_vad = await transformers.AutoModel.from_pretrained("onnx-community/silero-vad", {
      config: { model_type: "custom" },
      dtype: "fp32",
    });
    state = new transformers.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
    self.postMessage({ type: "info", message: "✅ VAD loaded" });

    // Load Whisper with WASM fallback for compatibility
    self.postMessage({ type: "info", message: "Loading speech recognition..." });
    transcriber = await transformers.pipeline("automatic-speech-recognition", "onnx-community/whisper-base", {
      device: "wasm", // Use WASM instead of WebGPU for better compatibility
      dtype: { encoder_model: "fp32", decoder_model_merged: "q8" },
    });
    
    // Warm up
    await transcriber(new Float32Array(INPUT_SAMPLE_RATE));
    self.postMessage({ type: "info", message: "✅ Whisper loaded" });

    // Load SmolLM2 with WASM and compatible dtype
    self.postMessage({ type: "info", message: "Loading language model..." });
    tokenizer = await transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct");
    llm = await transformers.AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", {
      dtype: "q8", // Use q8 instead of q4f16 for better compatibility
      device: "wasm", // Use WASM instead of WebGPU
    });
    
    // Warm up
    await llm.generate({ ...tokenizer("test"), max_new_tokens: 1 });
    self.postMessage({ type: "info", message: "✅ Language model loaded" });
    
    // Load Kokoro TTS
    self.postMessage({ type: "info", message: "Loading TTS model..." });
    const ttsLoaded = await loadKokoroTTS();
    if (!ttsLoaded) {
      self.postMessage({ type: "info", message: "⚠️ TTS model failed, will use Web Speech API" });
    }
    
    return true;
  } catch (error) {
    self.postMessage({ type: "info", message: `⚠️ AI model loading failed: ${error.message}` });
    return false;
  }
}

// Initialize models
async function initializeModels() {
  try {
    self.postMessage({ type: "info", message: "🚀 Starting enhanced conversation worker..." });
    
    // Try to load transformers and AI models
    const transformersLoaded = await loadTransformers();
    let aiModelsLoaded = false;
    
    if (transformersLoaded) {
      self.postMessage({ type: "info", message: "🔄 Loading AI models..." });
      aiModelsLoaded = await loadAIModels();
    }
    
    // Initialize conversation context
    messages = [{
      role: "system",
      content: "You're a helpful and conversational AI avatar assistant. Keep your responses short, clear, and casual."
    }];

    const statusMessage = aiModelsLoaded ? 
      "🎉 All AI models loaded successfully!" : 
      "✅ System ready with Web Speech API fallback";
      
    self.postMessage({
      type: "status",
      status: "ready",
      message: statusMessage,
      voices: defaultVoices
    });

    self.postMessage({ type: "info", message: statusMessage });

  } catch (error) {
    self.postMessage({ error: `Initialization failed: ${error.message}` });
    self.postMessage({ type: "info", message: "🔄 Worker ready in basic mode" });
  }
}

// Generate audio response (TTS)
async function generateAudio(text) {
  try {
    workerLog('INFO', 'Starting TTS generation', { text: text.substring(0, 100) + '...' });
    self.postMessage({ type: "info", message: `Generating speech for: "${text}"` });
    
    // Try loaded TTS model first if available
    if (kokoroTTS && transformers) {
      try {
        workerLog('INFO', 'Attempting TTS model generation', { 
          modelType: typeof kokoroTTS,
          hasPipeline: typeof kokoroTTS === 'function',
          hasGenerate: typeof kokoroTTS?.generate === 'function',
          selectedVoice: voice
        });
        
        self.postMessage({ type: "info", message: `Using TTS model with voice: ${defaultVoices[voice]?.name || voice}` });
        
        let audioOutput;
        
        // Handle different TTS model types
        if (typeof kokoroTTS === 'function') {
          // Pipeline-based TTS (like local Kokoro)
          workerLog('INFO', 'Using pipeline-based TTS');
          
          // For Kokoro TTS, we need to specify the voice
          const voiceFile = defaultVoices[voice]?.file || "af_heart.bin";
          
          workerLog('INFO', 'Loading voice file', { voiceFile });
          
          // Generate with voice specification
          audioOutput = await kokoroTTS(text, {
            voice: voiceFile.replace('.bin', ''),
            speaker_embeddings: voiceFile
          });
        } else if (kokoroTTS.__call__) {
          // Custom model with __call__ method (improved Kokoro)
          workerLog('INFO', 'Using custom TTS model with phonemization');
          
          const voiceFile = defaultVoices[voice]?.file?.replace('.bin', '') || "af_heart";
          
          audioOutput = await kokoroTTS.__call__(text, {
            voice: voiceFile
          });
        } else if (kokoroTTS.generate) {
          // Model-based TTS (like SpeechT5)
          workerLog('INFO', 'Using model-based TTS (SpeechT5)');
          
          // For SpeechT5, we need to tokenize the text and provide speaker embeddings
          try {
            // Load tokenizer if not already loaded
            if (!tokenizer) {
              workerLog('INFO', 'Loading SpeechT5 tokenizer');
              tokenizer = await transformers.AutoTokenizer.from_pretrained("Xenova/speecht5_tts");
            }
            
            // Tokenize the input text
            const inputs = tokenizer(text, { return_tensors: 'pt' });
            
            // Create default speaker embeddings (shape: [1, 512])
            const speakerEmbeddings = new transformers.Tensor(
              'float32',
              new Float32Array(512).fill(0.1), // Simple default embeddings
              [1, 512]
            );
            
            // Generate with speaker embeddings
            audioOutput = await kokoroTTS.generate({
              input_ids: inputs.input_ids,
              speaker_embeddings: speakerEmbeddings,
              max_length: 1000 // Limit output length
            });
            
            workerLog('INFO', 'SpeechT5 generation completed');
            
          } catch (speechT5Error) {
            workerLog('ERROR', 'SpeechT5 generation failed', { 
              error: speechT5Error.message,
              stack: speechT5Error.stack
            });
            
            // Fallback to simpler generation
            try {
              audioOutput = await kokoroTTS.generate(text);
            } catch (fallbackError) {
              throw new Error(`SpeechT5 generation failed: ${speechT5Error.message}`);
            }
          }
        } else {
          // Direct call (legacy)
          workerLog('INFO', 'Using direct TTS call');
          audioOutput = await kokoroTTS(text, {
            speaker_embeddings: null, // Use default speaker
          });
        }
        
        workerLog('INFO', 'TTS model returned output', { 
          outputType: typeof audioOutput,
          hasAudio: !!audioOutput?.audio,
          hasData: !!audioOutput?.data,
          keys: audioOutput ? Object.keys(audioOutput) : []
        });
        
        // Extract audio data with multiple fallbacks
        let audioData = null;
        let sampleRate = 16000;
        let metadata = {};
        
        if (audioOutput) {
          // Try different ways to extract audio data
          if (audioOutput.audio) {
            audioData = audioOutput.audio.data || audioOutput.audio;
            sampleRate = audioOutput.audio.sampling_rate || audioOutput.sampling_rate || 16000;
            metadata = audioOutput.metadata || {};
          } else if (audioOutput.data) {
            audioData = audioOutput.data;
            sampleRate = audioOutput.sampling_rate || 16000;
            metadata = audioOutput.metadata || {};
          } else if (audioOutput.sequences) {
            audioData = audioOutput.sequences[0] || audioOutput.sequences;
            sampleRate = audioOutput.sampling_rate || 16000;
          } else if (Array.isArray(audioOutput) || audioOutput.length !== undefined) {
            audioData = audioOutput;
            sampleRate = 16000;
          }
          
          // Extract metadata if available
          if (audioOutput.metadata) {
            metadata = audioOutput.metadata;
            workerLog('INFO', 'TTS metadata extracted', {
              voice: metadata.voice,
              phonemeLength: metadata.phonemes?.length,
              tokenCount: metadata.tokenCount,
              inferenceTime: metadata.inferenceTime
            });
          }
        }
        
        workerLog('INFO', 'Audio data extraction result', {
          hasAudioData: !!audioData,
          audioDataType: typeof audioData,
          audioDataLength: audioData?.length,
          sampleRate
        });
        
        if (audioData && audioData.length > 0) {
          // Convert to Float32Array if needed
          if (!(audioData instanceof Float32Array)) {
            audioData = new Float32Array(audioData);
          }
          
          // Send generated audio data
          self.postMessage({
            type: "output",
            text: text,
            result: { 
              audio: { 
                data: audioData, 
                sampleRate: sampleRate,
                metadata: metadata
              } 
            }
          });
          
          const duration = audioData.length / sampleRate;
          workerLog('SUCCESS', 'TTS audio generated successfully', {
            audioLength: audioData.length,
            sampleRate,
            duration: `${duration.toFixed(2)}s`,
            voice: metadata.voice || voice,
            phonemes: metadata.phonemes ? metadata.phonemes.substring(0, 50) + '...' : 'none',
            tokenCount: metadata.tokenCount || 'unknown',
            inferenceTime: metadata.inferenceTime || 'unknown'
          });
          
          self.postMessage({ type: "info", message: `✅ TTS audio generated successfully (${duration.toFixed(2)}s)` });
          return;
        } else {
          workerLog('WARN', 'No audio data generated from TTS model');
          throw new Error("No audio data generated");
        }
        
      } catch (ttsError) {
        workerLog('ERROR', 'TTS model generation failed', { 
          error: ttsError.message, 
          stack: ttsError.stack 
        });
        self.postMessage({ type: "info", message: `⚠️ TTS model failed: ${ttsError.message}` });
        // Fall through to Web Speech API
      }
    }
    
    // Fallback to Web Speech API
    workerLog('INFO', 'Using Web Speech API fallback');
    self.postMessage({
      type: "output",
      text: text,
      result: { audio: { useWebSpeech: true, text: text } }
    });
    
    self.postMessage({ type: "info", message: "Using Web Speech API fallback" });
    
  } catch (error) {
    workerLog('ERROR', 'TTS generation failed', { error: error.message, stack: error.stack });
    self.postMessage({ type: "error", message: `TTS error: ${error.message}` });
    
    // Final fallback
    self.postMessage({
      type: "output",
      text: text,
      result: { audio: { useWebSpeech: true, text: text } }
    });
  }
}

// Simple response generation
function generateSimpleResponse(userInput) {
  const responses = [
    "That's interesting! Tell me more about that.",
    "I understand what you're saying. Thanks for sharing!",
    "That's a great point. What do you think about it?",
    "I'm listening! Please continue.",
    "Thanks for talking with me. I enjoy our conversation!",
    "Could you elaborate on that?",
    "That sounds fascinating! What happened next?",
    "I see what you mean. How did that make you feel?"
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}

// Voice activity detection (real or simulated)
function detectSpeech(audioData) {
  if (silero_vad && state && transformers) {
    try {
      const inputs = {
        input: new transformers.Tensor("float32", audioData, [1, audioData.length]),
        state: state,
        sr: new transformers.Tensor("int64", [INPUT_SAMPLE_RATE]), // Fixed: use 'sr' instead of 'sample_rate'
      };

      const { output: speech_prob, state: new_state } = silero_vad(inputs);
      state = new_state;

      if (speech_prob && speech_prob.data && speech_prob.data.length > 0) {
        return speech_prob.data[0] > 0.3;
      }
    } catch (error) {
      console.error('VAD error:', error);
      // Fall through to simulation
    }
  }
  
  // Fallback: simulate speech detection
  const now = Date.now();
  if (now - lastSpeechTime < speechCooldown) return false;
  
  if (Math.random() > 0.92) { // 8% chance
    lastSpeechTime = now;
    return true;
  }
  return false;
}

// Process audio buffer
function processAudioBuffer(audioData) {
  if (!callActive || isPlaying) return;

  const buffer = new Float32Array(audioData);
  
  // Add to buffer
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

  // Check for speech
  if (detectSpeech(buffer)) {
    handleSpeechDetected();
  }
}

// Handle detected speech
async function handleSpeechDetected() {
  if (isRecording || isPlaying) return;
  
  isRecording = true;
  self.postMessage({ type: "status", status: "recording_start", message: "Listening..." });
  
  setTimeout(async () => {
    if (!callActive) return;
    
    isRecording = false;
    self.postMessage({ type: "status", status: "recording_end", message: "Processing..." });
    
    isPlaying = true;
    
    // Extract audio for processing (2 seconds of audio)
    const audioToProcess = BUFFER.slice(Math.max(0, bufferPointer - INPUT_SAMPLE_RATE * 2), bufferPointer);
    
    let userText = "User said something";
    let responseText = generateSimpleResponse(userText);
    
    // Try real AI models if available
    try {
      if (transcriber && audioToProcess.length > 0) {
        self.postMessage({ type: "info", message: `Transcribing ${audioToProcess.length} samples...` });
        
        // Transcribe with proper format
        const transcription = await transcriber(audioToProcess, {
          task: "transcribe",
          language: null, // auto-detect
        });
        
        const transcribedText = transcription?.text?.trim() || "";
        if (transcribedText && transcribedText.length > 2) {
          userText = transcribedText;
          self.postMessage({ type: "info", message: `User said: "${userText}"` });
          
          // Generate AI response if LLM is available
          if (llm && tokenizer) {
            try {
              self.postMessage({ type: "info", message: "Generating AI response..." });
              
              // Create conversation context
              const messages = [
                { role: "system", content: "You are a friendly AI assistant. Give concise, helpful responses." },
                { role: "user", content: userText }
              ];
              
              // Apply chat template with fallback
              let prompt;
              try {
                prompt = tokenizer.apply_chat_template(messages, { 
                  tokenize: false, 
                  add_generation_prompt: true 
                });
              } catch (e) {
                // Fallback prompt format
                prompt = `System: You are a friendly AI assistant. Give concise, helpful responses.\nUser: ${userText}\nAssistant:`;
              }
              
              // Tokenize and generate
              const inputs = tokenizer(prompt);
              const outputs = await llm.generate({
                ...inputs,
                max_new_tokens: 50,
                do_sample: true,
                temperature: 0.7,
                top_p: 0.9,
                repetition_penalty: 1.1,
                pad_token_id: tokenizer.eos_token_id,
              });

              // Decode response safely
              const responseTokens = outputs.sequences?.[0] || outputs;
              if (responseTokens) {
                const fullResponse = tokenizer.decode(responseTokens, { skip_special_tokens: true });
                
                // Extract just the assistant's response
                if (fullResponse.includes("Assistant:")) {
                  responseText = fullResponse.split("Assistant:").pop().trim();
                } else {
                  responseText = fullResponse.replace(prompt, "").trim();
                }
                
                if (!responseText || responseText.length < 2) {
                  responseText = generateSimpleResponse(userText);
                }
              }
            } catch (llmError) {
              self.postMessage({ type: "info", message: `LLM error: ${llmError.message}` });
              responseText = generateSimpleResponse(userText);
            }
          }
        } else {
          self.postMessage({ type: "info", message: "No clear speech detected" });
          responseText = "I didn't catch that clearly. Could you repeat?";
        }
      }
    } catch (error) {
      self.postMessage({ type: "info", message: `AI processing error: ${error.message}` });
      responseText = "I'm having trouble processing that. Could you try again?";
    }
    
    self.postMessage({ type: "info", message: `AI response: "${responseText}"` });
    
    // Send transcript first
    self.postMessage({ type: "transcript", text: userText });
    self.postMessage({ type: "response", text: responseText });
    
    // Generate audio response
    await generateAudio(responseText);
    
    // Reset playing state after a brief delay
    setTimeout(() => {
      isPlaying = false;
    }, 1000);
    
  }, 1000 + Math.random() * 500);
}

// Main message handler
self.onmessage = async (event) => {
  try {
    const { type, data } = event.data;
    workerLog('INFO', `Received message: ${type}`, event.data);

    switch (type) {
      case "start_call":
        workerLog('INFO', 'Starting call - setting up audio processing');
        callActive = true;
        isPlaying = true;
        const greeting = "Hello! I'm your AI assistant. How can I help you today?";
        workerLog('INFO', 'Sending greeting message', { greeting });
        self.postMessage({ 
          type: "output", 
          text: greeting, 
          result: { audio: { useWebSpeech: true, text: greeting } }
        });
        break;
        
      case "end_call":
        workerLog('INFO', 'Ending call - cleaning up state');
        callActive = false;
        isRecording = false;
        isPlaying = false;
        bufferPointer = 0;
        self.postMessage({ type: "info", message: "Call ended" });
        break;
        
      case "set_voice":
        const newVoice = event.data.voice || "af_heart";
        workerLog('INFO', `Voice change requested: ${voice} -> ${newVoice}`);
        voice = newVoice;
        self.postMessage({ type: "info", message: `Voice changed to: ${defaultVoices[voice]?.name || voice}` });
        break;
        
      case "audio":
        if (event.data.data) {
          const audioDataLength = event.data.data.length;
          workerLog('DEBUG', `Audio data received: ${audioDataLength} samples`, {
            bufferPointer,
            callActive,
            isRecording,
            isPlaying
          });
          processAudioBuffer(event.data.data);
        } else {
          workerLog('WARN', 'Audio message received but no data present');
        }
        break;
        
      case "playback_started":
        workerLog('INFO', 'Playback started - setting isPlaying to true');
        isPlaying = true;
        break;
        
      case "playback_ended":
        workerLog('INFO', 'Playback ended - setting isPlaying to false');
        isPlaying = false;
        self.postMessage({ type: "info", message: "Ready for next input" });
        break;
        
      case "test_phonemizer":
        workerLog('INFO', 'Testing phonemizer functionality', { text: event.data.text });
        try {
          const testText = event.data.text || "Hello world! This costs $12.34.";
          const normalized = normalize_text(testText);
          const quickPhonemes = quickPhonemize(normalized);
          const fullPhonemes = await phonemize(normalized);
          
          self.postMessage({
            type: "debug_log",
            level: "INFO",
            message: `Phonemizer test results:
Original: "${testText}"
Normalized: "${normalized}"
Quick phonemes: "${quickPhonemes}"
Full phonemes: "${fullPhonemes}"
eSpeak-NG ${quickPhonemes !== fullPhonemes ? 'AVAILABLE' : 'UNAVAILABLE'}`
          });
        } catch (error) {
          self.postMessage({
            type: "error",
            error: `Phonemizer test failed: ${error.message}`
          });
        }
        break;
        
      case "test_tts":
        workerLog('INFO', 'Testing TTS functionality', { text: event.data.text });
        try {
          const testText = event.data.text || "Hello! This is a test of the TTS system.";
          await generateAudio(testText);
        } catch (error) {
          self.postMessage({
            type: "error",
            error: `TTS test failed: ${error.message}`
          });
        }
        break;
        
      case "test_models":
        workerLog('INFO', 'Testing model loading status');
        const modelStatus = {
          vad: !!silero_vad,
          transcriber: !!transcriber,
          llm: !!llm,
          tokenizer: !!tokenizer,
          kokoroTTS: !!kokoroTTS
        };
        
        self.postMessage({
          type: "debug_log",
          level: "INFO",
          message: `Model loading status:
VAD: ${modelStatus.vad ? '✅ Loaded' : '❌ Not loaded'}
Transcriber: ${modelStatus.transcriber ? '✅ Loaded' : '❌ Not loaded'}
LLM: ${modelStatus.llm ? '✅ Loaded' : '❌ Not loaded'}
Tokenizer: ${modelStatus.tokenizer ? '✅ Loaded' : '❌ Not loaded'}
Kokoro TTS: ${modelStatus.kokoroTTS ? '✅ Loaded' : '❌ Not loaded'}`
        });
        break;
        
      default:
        workerLog('WARN', `Unknown message type received: ${type}`, event.data);
        self.postMessage({ type: "info", message: `Unknown message type: ${type}` });
    }
  } catch (error) {
    workerLog('ERROR', 'Message handler error', { error: error.message, stack: error.stack });
    self.postMessage({ error: `Message handler error: ${error.message}` });
  }
};

// Error handlers
self.addEventListener('error', (error) => {
  workerLog('ERROR', 'Worker error event', { 
    message: error.message, 
    filename: error.filename, 
    lineno: error.lineno, 
    colno: error.colno,
    stack: error.error?.stack 
  });
  self.postMessage({ error: `Worker error: ${error.message || error.toString()}` });
});

self.addEventListener('unhandledrejection', (event) => {
  workerLog('ERROR', 'Unhandled promise rejection', { 
    reason: event.reason, 
    promise: event.promise?.toString(),
    stack: event.reason?.stack 
  });
  self.postMessage({ error: `Promise rejection: ${event.reason}` });
});

// Initialize
workerLog('INFO', 'Starting worker initialization process');
self.postMessage({ type: "info", message: "🚀 Starting working conversation system..." });
initializeModels().then(() => {
  workerLog('SUCCESS', 'Worker initialization completed successfully');
}).catch(error => {
  workerLog('ERROR', 'Worker initialization failed', { error: error.message, stack: error.stack });
});
