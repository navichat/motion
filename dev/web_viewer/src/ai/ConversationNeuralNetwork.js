/**
 * Conversation Neural Network Logic
 * Handles model management and neural network operations
 * Separated from worker orchestration for modularity
 */

import {
  AutoModel,
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
  Tensor,
  pipeline,
} from "@huggingface/transformers";

import { KokoroTTS } from "kokoro-js";

import {
  INPUT_SAMPLE_RATE,
  SPEECH_THRESHOLD,
  EXIT_THRESHOLD,
  MIN_SPEECH_DURATION_SAMPLES,
  DEVICE_DTYPE_CONFIGS,
  DEFAULT_MODELS,
  SYSTEM_PROMPTS
} from "./ConversationConstants.js";

export class ConversationNeuralNetwork {
  constructor(device = 'webgpu') {
    this.device = device;
    this.models = {
      vad: null,
      asr: null,
      llm: null,
      tts: null
    };
    
    this.tokenizer = null;
    this.vadState = null;
    this.vadSr = null;
    
    this.isInitialized = false;
    this.conversations = new Map(); // Support multiple conversations
    this.currentConversationId = 'default';
  }

  /**
   * Initialize all neural network models
   */
  async initialize(config = {}) {
    try {
      this.device = config.device || this.device;
      
      const tasks = [
        this.initializeVAD(),
        this.initializeSpeechRecognition(),
        this.initializeLLM(config),
        this.initializeTTS()
      ];
      
      await Promise.all(tasks);
      
      this.isInitialized = true;
      return {
        success: true,
        voices: this.models.tts ? this.models.tts.voices : []
      };
      
    } catch (error) {
      throw new Error(`Neural network initialization failed: ${error.message}`);
    }
  }

  /**
   * Initialize Voice Activity Detection model
   */
  async initializeVAD() {
    this.models.vad = await AutoModel.from_pretrained(DEFAULT_MODELS.vad, {
      config: { model_type: "custom" },
      dtype: DEVICE_DTYPE_CONFIGS[this.device].vad_dtype,
    });
    
    // Initialize VAD state tensors
    this.vadSr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
    this.vadState = new Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);
  }

  /**
   * Initialize Automatic Speech Recognition model
   */
  async initializeSpeechRecognition() {
    this.models.asr = await pipeline(
      "automatic-speech-recognition",
      DEFAULT_MODELS.asr,
      {
        device: this.device,
        dtype: DEVICE_DTYPE_CONFIGS[this.device],
      }
    );
    
    // Compile shaders with dummy input
    await this.models.asr(new Float32Array(INPUT_SAMPLE_RATE));
  }

  /**
   * Initialize Large Language Model
   */
  async initializeLLM(config) {
    this.tokenizer = await AutoTokenizer.from_pretrained(DEFAULT_MODELS.llm);
    this.models.llm = await AutoModelForCausalLM.from_pretrained(DEFAULT_MODELS.llm, {
      dtype: DEVICE_DTYPE_CONFIGS[this.device].llm_dtype,
      device: this.device,
    });
    
    // Initialize default conversation
    const systemPromptKey = config.systemPrompt || 'conversational';
    const systemPrompt = SYSTEM_PROMPTS[systemPromptKey] || SYSTEM_PROMPTS.conversational;
    
    this.conversations.set(this.currentConversationId, {
      messages: [{ role: "system", content: systemPrompt }],
      cache: null
    });
    
    // Compile shaders
    await this.models.llm.generate({ ...this.tokenizer("x"), max_new_tokens: 1 });
  }

  /**
   * Initialize Text-to-Speech model
   */
  async initializeTTS() {
    try {
      this.models.tts = await KokoroTTS.from_pretrained(DEFAULT_MODELS.tts, {
        dtype: DEVICE_DTYPE_CONFIGS[this.device].tts_dtype,
        device: this.device,
      });
    } catch (error) {
      console.warn('TTS model failed to load:', error.message);
      this.models.tts = null;
    }
  }

  /**
   * Perform Voice Activity Detection
   */
  async performVAD(audioBuffer, isCurrentlyRecording = false) {
    if (!this.models.vad || !this.vadState) {
      return false;
    }
    
    const input = new Tensor("float32", audioBuffer, [1, audioBuffer.length]);
    const { stateN, output } = await this.models.vad({ 
      input, 
      sr: this.vadSr, 
      state: this.vadState 
    });
    
    this.vadState = stateN;
    const speechProbability = output.data[0];
    
    // Return both the decision and probability for advanced logic
    const isSpeech = (
      speechProbability > SPEECH_THRESHOLD ||
      (isCurrentlyRecording && speechProbability >= EXIT_THRESHOLD)
    );
    
    return {
      isSpeech,
      probability: speechProbability,
      threshold: isCurrentlyRecording ? EXIT_THRESHOLD : SPEECH_THRESHOLD
    };
  }

  /**
   * Transcribe audio to text
   */
  async transcribeAudio(audioBuffer) {
    if (!this.models.asr) {
      throw new Error('Speech recognition model not loaded');
    }
    
    const result = await this.models.asr(audioBuffer);
    const text = result.text.trim();
    
    return {
      text,
      confidence: result.confidence || 1.0,
      isEmpty: ["", "[BLANK_AUDIO]"].includes(text)
    };
  }

  /**
   * Generate text response using LLM
   */
  async generateResponse(userText, conversationId = null) {
    if (!this.models.llm || !this.tokenizer) {
      throw new Error('Language model not loaded');
    }
    
    const convId = conversationId || this.currentConversationId;
    const conversation = this.conversations.get(convId);
    
    if (!conversation) {
      throw new Error(`Conversation ${convId} not found`);
    }
    
    // Add user message
    conversation.messages.push({ role: "user", content: userText });
    
    // Prepare inputs
    const inputs = this.tokenizer.apply_chat_template(conversation.messages, { 
      add_generation_prompt: true, 
      return_dict: true 
    });
    
    // Generate response with streaming
    const stoppingCriteria = new InterruptableStoppingCriteria();
    let responseText = "";
    
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        responseText += text;
      }
    });
    
    await this.models.llm.generate({
      ...inputs,
      max_new_tokens: 256,
      do_sample: true,
      temperature: 0.6,
      streamer,
      stopping_criteria: stoppingCriteria
    });
    
    if (responseText.trim()) {
      conversation.messages.push({ role: "assistant", content: responseText });
    }
    
    return {
      text: responseText,
      conversationLength: conversation.messages.length,
      stoppingCriteria
    };
  }

  /**
   * Generate speech audio from text
   */
  async generateSpeech(text, voice = 'af_heart') {
    if (!this.models.tts) {
      throw new Error('TTS model not available');
    }
    
    try {
      const audioData = await this.models.tts.generate(text, { voice });
      return {
        audio: audioData,
        voice,
        text
      };
    } catch (error) {
      throw new Error(`TTS generation failed: ${error.message}`);
    }
  }

  /**
   * Complete speech-to-speech pipeline
   */
  async processSpeechtToSpeech(audioBuffer, voice = 'af_heart', conversationId = null) {
    const results = {
      transcription: null,
      response: null,
      audio: null
    };
    
    try {
      // 1. Transcribe
      results.transcription = await this.transcribeAudio(audioBuffer);
      
      if (results.transcription.isEmpty) {
        return results;
      }
      
      // 2. Generate response
      results.response = await this.generateResponse(
        results.transcription.text, 
        conversationId
      );
      
      // 3. Generate speech (if TTS available)
      if (this.models.tts && results.response.text.trim()) {
        try {
          results.audio = await this.generateSpeech(results.response.text, voice);
        } catch (ttsError) {
          console.warn('TTS failed:', ttsError.message);
          // Continue without audio
        }
      }
      
      return results;
      
    } catch (error) {
      throw new Error(`Speech-to-speech processing failed: ${error.message}`);
    }
  }

  /**
   * Process text input (bypass speech recognition)
   */
  async processTextInput(text, voice = 'af_heart', conversationId = null) {
    const results = {
      transcription: { text, confidence: 1.0, isEmpty: false },
      response: null,
      audio: null
    };
    
    try {
      // Generate response
      results.response = await this.generateResponse(text, conversationId);
      
      // Generate speech (if TTS available)
      if (this.models.tts && results.response.text.trim()) {
        try {
          results.audio = await this.generateSpeech(results.response.text, voice);
        } catch (ttsError) {
          console.warn('TTS failed:', ttsError.message);
        }
      }
      
      return results;
      
    } catch (error) {
      throw new Error(`Text processing failed: ${error.message}`);
    }
  }

  /**
   * Create a new conversation
   */
  createConversation(conversationId, systemPrompt = 'conversational') {
    const prompt = SYSTEM_PROMPTS[systemPrompt] || SYSTEM_PROMPTS.conversational;
    
    this.conversations.set(conversationId, {
      messages: [{ role: "system", content: prompt }],
      cache: null
    });
    
    return conversationId;
  }

  /**
   * Reset a conversation
   */
  resetConversation(conversationId = null) {
    const convId = conversationId || this.currentConversationId;
    const conversation = this.conversations.get(convId);
    
    if (conversation) {
      // Keep only the system message
      conversation.messages = conversation.messages.slice(0, 1);
      conversation.cache = null;
    }
  }

  /**
   * Get conversation history
   */
  getConversationHistory(conversationId = null) {
    const convId = conversationId || this.currentConversationId;
    const conversation = this.conversations.get(convId);
    
    return conversation ? [...conversation.messages] : [];
  }

  /**
   * Get available TTS voices
   */
  getAvailableVoices() {
    return this.models.tts ? this.models.tts.voices : [];
  }

  /**
   * Get model status
   */
  getModelStatus() {
    return {
      initialized: this.isInitialized,
      device: this.device,
      models: {
        vad: !!this.models.vad,
        asr: !!this.models.asr,
        llm: !!this.models.llm,
        tts: !!this.models.tts
      }
    };
  }

  /**
   * Cleanup resources
   */
  dispose() {
    // Clean up models and free memory
    this.models = {
      vad: null,
      asr: null,
      llm: null,
      tts: null
    };
    
    this.tokenizer = null;
    this.vadState = null;
    this.vadSr = null;
    this.conversations.clear();
    this.isInitialized = false;
  }
}
