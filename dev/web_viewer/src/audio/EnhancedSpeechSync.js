/**
 * EnhancedSpeechSync.js
 * 
 * Enhanced audio-visual synchronization for speech and animation:
 * - Precise viseme timing for lip-sync
 * - Audio-driven gesture generation
 * - Real-time animation scheduling
 * - Multi-track animation coordination
 */

class EnhancedSpeechSync {
  constructor() {
    this.audioContext = null;
    this.analyzer = null;
    this.isAnalyzing = false;
    
    this.visemeTracker = null;
    this.gestureGenerator = null;
    this.animationScheduler = null;
    
    // Audio analysis parameters
    this.fftSize = 2048;
    this.frequencyBins = null;
    this.audioData = null;
    this.energyHistory = [];
    this.maxHistorySize = 60; // 1 second at 60fps
    
    // Synchronization timing
    this.syncOffset = 0; // ms offset for audio-visual sync
    this.frameDuration = 1000 / 60; // Target 60fps
    
    this.initialized = false;
  }

  /**
   * Initialize enhanced speech synchronization system
   */
  async initialize() {
    try {
      console.log('EnhancedSpeechSync: Initializing...');
      
      // Setup audio analysis - don't fail if not available
      try {
        await this.setupAudioAnalysis();
      } catch (error) {
        console.warn('EnhancedSpeechSync: Audio analysis setup failed, using basic mode:', error);
      }
      
      // Initialize viseme tracking - always available
      this.initializeVisemeTracking();
      
      // Initialize gesture generation - always available
      this.initializeGestureGeneration();
      
      // Setup animation scheduler - always available
      this.setupAnimationScheduler();
      
      this.initialized = true;
      console.log('EnhancedSpeechSync: Initialization complete');
      
    } catch (error) {
      console.warn('EnhancedSpeechSync: Initialization failed, using fallback mode:', error);
      // Still mark as initialized for fallback mode
      this.initialized = true;
      console.log('EnhancedSpeechSync: Running in fallback mode');
    }
  }

  /**
   * Setup real-time audio analysis
   */
  async setupAudioAnalysis() {
    try {
      // Create or reuse audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Create analyzer node
      this.analyzer = this.audioContext.createAnalyser();
      this.analyzer.fftSize = this.fftSize;
      this.analyzer.smoothingTimeConstant = 0.3;
      
      // Initialize frequency data arrays
      this.frequencyBins = new Uint8Array(this.analyzer.frequencyBinCount);
      this.audioData = new Float32Array(this.analyzer.frequencyBinCount);
      
      console.log('Audio analysis setup complete');
    } catch (error) {
      console.warn('Audio analysis setup failed:', error);
      // Continue without audio analysis - viseme sync will use text analysis instead
    }
  }

  /**
   * Initialize viseme tracking for lip-sync
   */
  initializeVisemeTracking() {
    this.visemeTracker = {
      // Viseme mappings based on frequency analysis
      visemeMap: {
        'A': { frequencies: [800, 1200], threshold: 0.3 },
        'E': { frequencies: [500, 2300], threshold: 0.25 },
        'I': { frequencies: [300, 2500], threshold: 0.2 },
        'O': { frequencies: [500, 800], threshold: 0.3 },
        'U': { frequencies: [300, 600], threshold: 0.25 },
        'M': { frequencies: [200, 400], threshold: 0.15 },
        'L': { frequencies: [200, 1000], threshold: 0.2 },
        'F': { frequencies: [1500, 4000], threshold: 0.2 },
        'S': { frequencies: [4000, 8000], threshold: 0.15 },
        'T': { frequencies: [2000, 6000], threshold: 0.1 }
      },
      
      currentViseme: 'neutral',
      visemeIntensity: 0,
      
      // Extract viseme from frequency data
      extractViseme: (frequencyData) => {
        let bestViseme = 'neutral';
        let maxIntensity = 0;
        
        for (const [viseme, config] of Object.entries(this.visemeMap)) {
          const intensity = this.calculateVisemeIntensity(frequencyData, config);
          if (intensity > config.threshold && intensity > maxIntensity) {
            bestViseme = viseme;
            maxIntensity = intensity;
          }
        }
        
        this.currentViseme = bestViseme;
        this.visemeIntensity = maxIntensity;
        
        return { viseme: bestViseme, intensity: maxIntensity };
      },
      
      calculateVisemeIntensity: (frequencyData, config) => {
        const nyquist = this.audioContext.sampleRate / 2;
        let totalIntensity = 0;
        let sampleCount = 0;
        
        for (let freq of config.frequencies) {
          const bin = Math.floor((freq / nyquist) * frequencyData.length);
          if (bin < frequencyData.length) {
            totalIntensity += frequencyData[bin];
            sampleCount++;
          }
        }
        
        return sampleCount > 0 ? totalIntensity / (sampleCount * 255) : 0;
      }
    };
    
    console.log('Viseme tracking initialized');
  }

  /**
   * Initialize audio-driven gesture generation
   */
  initializeGestureGeneration() {
    this.gestureGenerator = {
      // Gesture mappings based on audio characteristics
      gestureMap: {
        'emphasis': { energyThreshold: 0.7, duration: 1000 },
        'idle': { energyThreshold: 0.1, duration: 2000 },
        'speaking': { energyThreshold: 0.3, duration: 500 },
        'question': { pitchRise: true, duration: 800 },
        'explanation': { steadyEnergy: true, duration: 1500 }
      },
      
      currentGesture: 'idle',
      gestureIntensity: 0,
      gestureStartTime: 0,
      
      // Generate appropriate gesture based on audio
      generateGesture: (audioFeatures) => {
        const { energy, pitch, variance } = audioFeatures;
        
        // Determine gesture type based on audio characteristics
        let newGesture = 'idle';
        let intensity = 0;
        
        if (energy > 0.7) {
          newGesture = 'emphasis';
          intensity = Math.min(1.0, energy);
        } else if (energy > 0.3) {
          if (pitch > 200 && variance > 0.5) {
            newGesture = 'question';
            intensity = 0.8;
          } else {
            newGesture = 'speaking';
            intensity = energy;
          }
        } else if (energy > 0.1) {
          newGesture = 'explanation';
          intensity = 0.4;
        }
        
        // Update gesture state
        if (newGesture !== this.currentGesture) {
          this.currentGesture = newGesture;
          this.gestureStartTime = performance.now();
        }
        
        this.gestureIntensity = intensity;
        
        return {
          gesture: newGesture,
          intensity: intensity,
          duration: this.gestureMap[newGesture]?.duration || 1000
        };
      }
    };
    
    console.log('Gesture generation initialized');
  }

  /**
   * Setup animation scheduler for multi-track coordination
   */
  setupAnimationScheduler() {
    this.animationScheduler = {
      tracks: {
        face: { animations: [], priority: 3 },
        gesture: { animations: [], priority: 2 },
        body: { animations: [], priority: 1 }
      },
      
      activeAnimations: new Map(),
      
      // Schedule animation with proper timing
      scheduleAnimation: (track, animation, startTime = null) => {
        const actualStartTime = startTime || (performance.now() + this.syncOffset);
        
        const scheduledAnimation = {
          ...animation,
          track: track,
          startTime: actualStartTime,
          id: this.generateAnimationId()
        };
        
        this.tracks[track].animations.push(scheduledAnimation);
        
        return scheduledAnimation.id;
      },
      
      // Update active animations
      update: (currentTime) => {
        for (const [trackName, track] of Object.entries(this.tracks)) {
          // Start new animations
          const readyAnimations = track.animations.filter(anim => 
            anim.startTime <= currentTime && !this.activeAnimations.has(anim.id)
          );
          
          for (const anim of readyAnimations) {
            this.activeAnimations.set(anim.id, anim);
          }
          
          // Remove completed animations
          track.animations = track.animations.filter(anim => 
            anim.startTime + anim.duration > currentTime
          );
        }
        
        // Clean up completed active animations
        for (const [id, anim] of this.activeAnimations) {
          if (performance.now() > anim.startTime + anim.duration) {
            this.activeAnimations.delete(id);
          }
        }
      },
      
      generateAnimationId: () => {
        return 'anim_' + Math.random().toString(36).substr(2, 9);
      }
    };
    
    console.log('Animation scheduler setup complete');
  }

  /**
   * Process TTS result with enhanced synchronization
   */
  async processTTSWithSync(ttsResult, avatar) {
    if (!this.initialized) {
      await this.initialize();
    }
    
    console.log('EnhancedSpeechSync: Processing TTS with synchronization...');
    
    try {
      const { audio, text, duration, visemes } = ttsResult;
      
      // Connect audio for real-time analysis
      const audioSource = await this.connectAudioForAnalysis(audio);
      
      // Schedule viseme animations if available
      if (visemes && visemes.length > 0) {
        await this.scheduleVisemeAnimations(visemes, duration, avatar);
      } else {
        // Use real-time viseme extraction
        this.startRealtimeVisemeExtraction(duration, avatar);
      }
      
      // Generate and schedule gesture animations
      await this.scheduleGestureAnimations(audio, duration, avatar);
      
      // Start coordinated playback
      await this.startCoordinatedPlayback(audioSource, duration, avatar);
      
      console.log('TTS synchronization complete');
      
    } catch (error) {
      console.error('EnhancedSpeechSync: TTS processing failed:', error);
      throw error;
    }
  }

  /**
   * Connect audio to analyzer for real-time processing
   */
  async connectAudioForAnalysis(audioBuffer) {
    // Create audio source
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    
    // Connect to analyzer
    source.connect(this.analyzer);
    this.analyzer.connect(this.audioContext.destination);
    
    return source;
  }

  /**
   * Schedule precise viseme animations from TTS data
   */
  async scheduleVisemeAnimations(visemes, duration, avatar) {
    console.log('Scheduling viseme animations:', visemes.length, 'visemes');
    
    for (const viseme of visemes) {
      const animation = {
        type: 'viseme',
        viseme: viseme.phoneme,
        intensity: viseme.intensity || 1.0,
        duration: viseme.duration,
        target: avatar.face || avatar.head
      };
      
      this.animationScheduler.scheduleAnimation('face', animation, viseme.time);
    }
  }

  /**
   * Start real-time viseme extraction and animation
   */
  startRealtimeVisemeExtraction(duration, avatar) {
    console.log('Starting real-time viseme extraction for', duration, 'ms');
    
    this.isAnalyzing = true;
    
    const extractVisemes = () => {
      if (!this.isAnalyzing) return;
      
      // Get frequency data
      this.analyzer.getByteFrequencyData(this.frequencyBins);
      this.analyzer.getFloatFrequencyData(this.audioData);
      
      // Extract viseme
      const visemeData = this.visemeTracker.extractViseme(this.frequencyBins);
      
      // Apply to avatar
      if (avatar && avatar.applyViseme) {
        avatar.applyViseme(visemeData.viseme, visemeData.intensity);
      }
      
      // Schedule next frame
      setTimeout(extractVisemes, this.frameDuration);
    };
    
    extractVisemes();
    
    // Stop after duration
    setTimeout(() => {
      this.isAnalyzing = false;
    }, duration);
  }

  /**
   * Schedule gesture animations based on audio analysis
   */
  async scheduleGestureAnimations(audioBuffer, duration, avatar) {
    console.log('Generating gesture animations for', duration, 'ms audio');
    
    // Analyze audio for gesture cues
    const audioFeatures = await this.analyzeAudioFeatures(audioBuffer);
    
    // Generate gesture timeline
    const gestureTimeline = this.generateGestureTimeline(audioFeatures, duration);
    
    // Schedule gesture animations
    for (const gesture of gestureTimeline) {
      const animation = {
        type: 'gesture',
        gesture: gesture.type,
        intensity: gesture.intensity,
        duration: gesture.duration,
        target: avatar.body || avatar
      };
      
      this.animationScheduler.scheduleAnimation('gesture', animation, gesture.startTime);
    }
  }

  /**
   * Analyze audio for gesture generation features
   */
  async analyzeAudioFeatures(audioBuffer) {
    // Simple audio feature extraction
    const channelData = audioBuffer.getChannelData(0);
    const windowSize = 1024;
    const features = [];
    
    for (let i = 0; i < channelData.length; i += windowSize) {
      const window = channelData.slice(i, i + windowSize);
      
      // Calculate energy
      const energy = window.reduce((sum, val) => sum + Math.abs(val), 0) / window.length;
      
      // Estimate pitch (simplified)
      const pitch = this.estimatePitch(window);
      
      // Calculate variance
      const mean = window.reduce((sum, val) => sum + val, 0) / window.length;
      const variance = window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window.length;
      
      features.push({ energy, pitch, variance, timestamp: i / audioBuffer.sampleRate * 1000 });
    }
    
    return features;
  }

  /**
   * Simple pitch estimation using autocorrelation
   */
  estimatePitch(audioWindow) {
    const sampleRate = 44100; // Assume standard sample rate
    const minFreq = 80;
    const maxFreq = 400;
    
    let bestCorrelation = 0;
    let bestPeriod = 0;
    
    const minPeriod = Math.floor(sampleRate / maxFreq);
    const maxPeriod = Math.floor(sampleRate / minFreq);
    
    for (let period = minPeriod; period <= maxPeriod && period < audioWindow.length / 2; period++) {
      let correlation = 0;
      
      for (let i = 0; i < audioWindow.length - period; i++) {
        correlation += audioWindow[i] * audioWindow[i + period];
      }
      
      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestPeriod = period;
      }
    }
    
    return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
  }

  /**
   * Generate gesture timeline based on audio features
   */
  generateGestureTimeline(audioFeatures, duration) {
    const timeline = [];
    const gestureTypes = ['idle', 'speaking', 'emphasis', 'question', 'explanation'];
    
    let currentTime = 0;
    const segmentDuration = 500; // 500ms segments
    
    for (let i = 0; i < audioFeatures.length; i += Math.floor(segmentDuration * 44.1)) {
      const segment = audioFeatures.slice(i, i + Math.floor(segmentDuration * 44.1));
      
      if (segment.length === 0) continue;
      
      const avgEnergy = segment.reduce((sum, f) => sum + f.energy, 0) / segment.length;
      const avgPitch = segment.reduce((sum, f) => sum + f.pitch, 0) / segment.length;
      const avgVariance = segment.reduce((sum, f) => sum + f.variance, 0) / segment.length;
      
      const gestureData = this.gestureGenerator.generateGesture({
        energy: avgEnergy,
        pitch: avgPitch,
        variance: avgVariance
      });
      
      timeline.push({
        type: gestureData.gesture,
        intensity: gestureData.intensity,
        duration: Math.min(gestureData.duration, duration - currentTime),
        startTime: currentTime
      });
      
      currentTime += segmentDuration;
      if (currentTime >= duration) break;
    }
    
    return timeline;
  }

  /**
   * Start coordinated playback of audio and animations
   */
  async startCoordinatedPlayback(audioSource, duration, avatar) {
    console.log('Starting coordinated playback...');
    
    const startTime = performance.now();
    
    // Start audio playback
    audioSource.start();
    
    // Start animation update loop
    const updateAnimations = () => {
      const currentTime = performance.now();
      const elapsed = currentTime - startTime;
      
      if (elapsed >= duration) {
        console.log('Playback complete');
        return;
      }
      
      // Update animation scheduler
      this.animationScheduler.update(currentTime);
      
      // Apply active animations to avatar
      this.applyActiveAnimations(avatar);
      
      // Schedule next update
      requestAnimationFrame(updateAnimations);
    };
    
    updateAnimations();
  }

  /**
   * Apply active animations to avatar
   */
  applyActiveAnimations(avatar) {
    if (!avatar) return;
    
    for (const [id, animation] of this.animationScheduler.activeAnimations) {
      const progress = this.calculateAnimationProgress(animation);
      
      switch (animation.type) {
        case 'viseme':
          if (avatar.applyViseme) {
            avatar.applyViseme(animation.viseme, animation.intensity * progress);
          }
          break;
          
        case 'gesture':
          if (avatar.applyGesture) {
            avatar.applyGesture(animation.gesture, animation.intensity * progress);
          }
          break;
      }
    }
  }

  /**
   * Calculate animation progress (0-1)
   */
  calculateAnimationProgress(animation) {
    const elapsed = performance.now() - animation.startTime;
    const progress = Math.min(1, elapsed / animation.duration);
    
    // Apply easing for smoother animations
    return this.easeInOutQuad(progress);
  }

  /**
   * Quadratic ease-in-out function
   */
  easeInOutQuad(t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  }

  /**
   * Stop all synchronization and cleanup
   */
  stop() {
    console.log('EnhancedSpeechSync: Stopping...');
    
    this.isAnalyzing = false;
    
    // Clear animation scheduler
    if (this.animationScheduler) {
      for (const track of Object.values(this.animationScheduler.tracks)) {
        track.animations = [];
      }
      this.animationScheduler.activeAnimations.clear();
    }
    
    // Reset viseme tracker
    if (this.visemeTracker) {
      this.visemeTracker.currentViseme = 'neutral';
      this.visemeTracker.visemeIntensity = 0;
    }
    
    // Reset gesture generator
    if (this.gestureGenerator) {
      this.gestureGenerator.currentGesture = 'idle';
      this.gestureGenerator.gestureIntensity = 0;
    }
  }

  /**
   * Get current synchronization status
   */
  getStatus() {
    return {
      initialized: this.initialized,
      analyzing: this.isAnalyzing,
      currentViseme: this.visemeTracker?.currentViseme,
      visemeIntensity: this.visemeTracker?.visemeIntensity,
      currentGesture: this.gestureGenerator?.currentGesture,
      gestureIntensity: this.gestureGenerator?.gestureIntensity,
      activeAnimations: this.animationScheduler?.activeAnimations.size || 0
    };
  }
}

// Export for use in browser environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EnhancedSpeechSync;
} else if (typeof window !== 'undefined') {
  window.EnhancedSpeechSync = EnhancedSpeechSync;
}