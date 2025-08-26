/**
 * ConversationManager.js
 * 
 * Unified conversation orchestrator that integrates:
 * - Speech-to-Text (Whisper)
 * - Response generation
 * - Text-to-Speech (Multi-engine)
 * - 3D Avatar animation sync
 * - Classroom environment interaction
 */

class ConversationManager {
  constructor() {
    this.stt = null;
    this.tts = null;
    this.avatar = null;
    this.classroom = null;
    this.state = 'idle'; // idle, listening, processing, speaking
    this.conversationHistory = [];
    
    this.initialized = false;
    this.audioContext = null;
    this.mediaStream = null;
  }

  /**
   * Initialize all conversation components
   */
  async initialize() {
    try {
      console.log('ConversationManager: Initializing components...');
      
      // Initialize audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Initialize STT system
      await this.initializeSTT();
      
      // Initialize TTS system  
      await this.initializeTTS();
      
      // Initialize 3D Avatar in classroom
      await this.initializeAvatarScene();
      
      this.initialized = true;
      console.log('ConversationManager: All components initialized successfully');
      
    } catch (error) {
      console.error('ConversationManager: Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize Speech-to-Text with Whisper
   */
  async initializeSTT() {
    // Import Whisper STT processor (if available)
    if (typeof WhisperSTTProcessor !== 'undefined') {
      this.stt = new WhisperSTTProcessor();
      await this.stt.initialize();
    } else {
      console.warn('ConversationManager: Whisper STT not available, using Web Speech API fallback');
      this.stt = this.createWebSpeechSTT();
    }
  }

  /**
   * Initialize Multi-engine TTS system
   */
  async initializeTTS() {
    // Import multi-engine TTS manager (if available)
    if (typeof MultiEngineTTSManager !== 'undefined') {
      this.tts = new MultiEngineTTSManager();
      await this.tts.initialize();
    } else {
      console.warn('ConversationManager: Multi-engine TTS not available, using Speech Synthesis API');
      this.tts = this.createWebSpeechTTS();
    }
  }

  /**
   * Initialize 3D Avatar in classroom scene
   */
  async initializeAvatarScene() {
    // Import classroom avatar integration
    if (typeof ClassroomAvatarIntegration !== 'undefined') {
      this.classroom = new ClassroomAvatarIntegration();
      await this.classroom.initializeScene();
      this.avatar = this.classroom.avatar;
    } else {
      console.warn('ConversationManager: 3D Avatar integration not available');
      this.avatar = this.createAvatarFallback();
    }
  }

  /**
   * Start interactive conversation
   */
  async startConversation() {
    if (!this.initialized) {
      await this.initialize();
    }

    console.log('ConversationManager: Starting conversation...');
    this.state = 'listening';
    
    // Show avatar greeting
    if (this.avatar && this.avatar.speak) {
      await this.avatar.speak("Hello! I'm Ichika. How can I help you today?");
    }
    
    // Start listening for user input
    await this.startListening();
    
    return true;
  }

  /**
   * Start listening for user speech
   */
  async startListening() {
    try {
      // Get microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      if (this.stt && this.stt.startListening) {
        // Use advanced STT system
        await this.stt.startListening(this.mediaStream, (transcript) => {
          this.handleUserSpeech(transcript);
        });
      } else {
        // Use Web Speech API fallback
        this.startWebSpeechListening();
      }
      
      console.log('ConversationManager: Listening started');
      
    } catch (error) {
      console.error('ConversationManager: Failed to start listening:', error);
      this.state = 'idle';
    }
  }

  /**
   * Handle user speech input
   */
  async handleUserSpeech(transcript) {
    if (this.state !== 'listening') return;
    
    console.log('ConversationManager: User said:', transcript);
    this.state = 'processing';
    
    try {
      // Add to conversation history
      this.conversationHistory.push({ role: 'user', content: transcript });
      
      // Generate response
      const response = this.generateResponse(transcript);
      
      // Add response to history
      this.conversationHistory.push({ role: 'assistant', content: response });
      
      // Have avatar speak the response
      await this.speakResponse(response);
      
      // Return to listening state
      this.state = 'listening';
      
    } catch (error) {
      console.error('ConversationManager: Error handling speech:', error);
      this.state = 'listening';
    }
  }

  /**
   * Generate contextual response
   */
  generateResponse(userInput) {
    // Simple response generation for now
    // This could be enhanced with AI models or more sophisticated logic
    const responses = [
      "That's interesting! Tell me more about that.",
      "I understand. How does that make you feel?",
      "Thanks for sharing that with me. What would you like to know?",
      "That's a great question. Let me think about that.",
      "I see. Is there anything specific you'd like help with?",
      "Fascinating! I love learning new things from our conversations.",
    ];
    
    // Context-aware responses for classroom setting
    if (userInput.toLowerCase().includes('learn') || userInput.toLowerCase().includes('teach')) {
      return "I'd love to help you learn! What subject interests you most?";
    }
    
    if (userInput.toLowerCase().includes('homework') || userInput.toLowerCase().includes('study')) {
      return "I'm here to help with your studies. What are you working on?";
    }
    
    // Default response
    return responses[Math.floor(Math.random() * responses.length)];
  }

  /**
   * Have avatar speak response with animation sync
   */
  async speakResponse(text) {
    this.state = 'speaking';
    
    try {
      if (this.avatar && this.avatar.speak) {
        // Use advanced avatar system with animation sync
        await this.avatar.speak(text);
      } else if (this.tts && this.tts.speak) {
        // Use TTS system
        await this.tts.speak(text);
      } else {
        // Fallback to Web Speech API
        await this.speakWithWebAPI(text);
      }
      
    } catch (error) {
      console.error('ConversationManager: Error speaking response:', error);
    }
  }

  /**
   * Stop conversation and cleanup
   */
  async stopConversation() {
    console.log('ConversationManager: Stopping conversation...');
    this.state = 'idle';
    
    // Stop listening
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    if (this.stt && this.stt.stopListening) {
      this.stt.stopListening();
    }
    
    // Stop avatar
    if (this.avatar && this.avatar.stop) {
      this.avatar.stop();
    }
  }

  /**
   * Get current conversation state
   */
  getState() {
    return {
      state: this.state,
      initialized: this.initialized,
      conversationHistory: this.conversationHistory,
      hasSTT: !!this.stt,
      hasTTS: !!this.tts,
      hasAvatar: !!this.avatar
    };
  }

  // Fallback implementations for when advanced components aren't available

  createWebSpeechSTT() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      return null;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    return {
      startListening: (mediaStream, callback) => {
        recognition.onresult = (event) => {
          const transcript = event.results[event.results.length - 1][0].transcript;
          callback(transcript);
        };
        recognition.start();
      },
      stopListening: () => recognition.stop()
    };
  }

  startWebSpeechListening() {
    if (this.stt && this.stt.startListening) {
      this.stt.startListening(null, (transcript) => {
        this.handleUserSpeech(transcript);
      });
    }
  }

  createWebSpeechTTS() {
    return {
      speak: (text) => {
        return new Promise((resolve) => {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.onend = resolve;
          speechSynthesis.speak(utterance);
        });
      }
    };
  }

  async speakWithWebAPI(text) {
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onend = resolve;
      speechSynthesis.speak(utterance);
    });
  }

  createAvatarFallback() {
    return {
      speak: async (text) => {
        console.log('Avatar would say:', text);
        // Could trigger simple visual feedback
        if (this.tts && this.tts.speak) {
          await this.tts.speak(text);
        }
      },
      stop: () => console.log('Avatar stopped')
    };
  }
}

// Export for use in browser environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ConversationManager;
} else if (typeof window !== 'undefined') {
  window.ConversationManager = ConversationManager;
}