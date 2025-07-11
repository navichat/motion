/**
 * ResourceManager - Centralized GPU context and resource management
 * Provides dependency injection for AI models and audio processing
 */

import { BrowserCompatibility } from './BrowserCompatibility.js';
import { 
  DEFAULT_MODELS, 
  DEVICE_DTYPE_CONFIGS, 
  MEMORY_THRESHOLDS,
  TIMEOUTS 
} from './constants.js';
import { ModernVoiceActivityDetector } from './ModernVoiceActivityDetector.js';
import { VoiceActivityDetector } from './VoiceActivityDetector.js';
import { ModernAudioQueue } from './ModernAudioQueue.js';
import { AudioQueue } from './AudioQueue.js';

export class ResourceManager extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            preferredDevice: options.device || 'wasm',
            memoryThresholdMB: options.memoryThresholdMB || 512,
            maxConcurrentModels: options.maxConcurrentModels || 2,
            modelCacheTimeout: options.modelCacheTimeout || 30000,
            ...options
        };

        // GPU/Compute contexts
        this.gpuContext = null;
        this.audioContext = null;
        this.transformersEnv = null;
        
        // Model registry and lifecycle management
        this.modelRegistry = new Map();
        this.loadedModels = new Set();
        this.modelLoadPromises = new Map();
        
        // Resource monitoring
        this.memoryUsage = 0;
        this.resourceMonitor = null;
        
        // Shared pipeline cache
        this.pipelineCache = new Map();
        
        this.initialize();
    }

    /**
     * Initialize the resource manager
     */
    async initialize() {
        try {
            // Initialize audio context
            await this.initializeAudioContext();
            
            // Initialize GPU/ML context
            await this.initializeMLContext();
            
            // Start resource monitoring
            this.startResourceMonitoring();
            
            this.emit('initialized', { 
                audioContext: !!this.audioContext,
                mlContext: !!this.transformersEnv,
                preferredDevice: this.options.preferredDevice
            });
            
        } catch (error) {
            this.emit('error', { type: 'initialization', error });
            throw error;
        }
    }

    /**
     * Initialize audio context for TTS and audio processing
     */
    async initializeAudioContext() {
        try {
            // Use BrowserCompatibility for cross-browser audio context creation
            this.audioContext = BrowserCompatibility.createAudioContext();
            
            // Update our options to match the actual sample rate
            this.options.audioSampleRate = this.audioContext.sampleRate;
            
            console.log(`🎵 Audio context initialized: ${this.audioContext.sampleRate}Hz`);
            
            // Get browser compatibility info
            const features = BrowserCompatibility.detectFeatures();
            if (features.isMobile) {
                console.log('📱 Mobile device detected - audio optimizations applied');
            }
            
        } catch (error) {
            console.warn('Audio context initialization failed:', error);
            // Non-critical for ML models
        }
    }

    /**
     * Initialize ML/GPU context
     */
    async initializeMLContext() {
        try {
            // Check for global transformers.js
            if (window.transformers && window.transformers.env) {
                this.transformersEnv = window.transformers.env;
                
                // Configure transformers environment
                this.transformersEnv.allowRemoteModels = true;
                this.transformersEnv.allowLocalModels = true;
                this.transformersEnv.localModelPath = './models/';
                
                // Set remote URL template to use local models first
                this.transformersEnv.remoteHost = './models/';
                this.transformersEnv.remotePathTemplate = '{model}/';
                
                // Disable automatic model downloading to force local usage
                this.transformersEnv.allowRemoteModels = false;
                
                // Prefer WASM backend to avoid WebGPU experimental warnings
                if (this.transformersEnv.backends) {
                    this.transformersEnv.backends.onnx = {
                        wasm: {
                            executionProviders: ['wasm']
                        }
                    };
                }
                
                console.log(`🧠 ML context initialized with device: ${this.options.preferredDevice}`);
                console.log(`📁 Local model path: ${this.transformersEnv.localModelPath}`);
            }
            
        } catch (error) {
            console.warn('ML context initialization failed:', error);
            // Will fall back to browser APIs
        }
    }

    /**
     * Register a model with the resource manager
     */
    registerModel(modelId, modelClass, options = {}) {
        const modelConfig = {
            id: modelId,
            class: modelClass,
            options: {
                ...options,
                device: this.options.preferredDevice,
                resourceManager: this
            },
            instance: null,
            lastUsed: 0,
            loadPromise: null
        };
        
        this.modelRegistry.set(modelId, modelConfig);
        
        this.emit('modelRegistered', { modelId, options });
        
        return modelConfig;
    }

    /**
     * Get or create a model instance
     */
    async getModel(modelId) {
        const modelConfig = this.modelRegistry.get(modelId);
        
        if (!modelConfig) {
            throw new Error(`Model '${modelId}' not registered`);
        }

        // Return existing instance if loaded
        if (modelConfig.instance && modelConfig.instance.isLoaded()) {
            modelConfig.lastUsed = Date.now();
            return modelConfig.instance;
        }

        // Check if already loading
        if (modelConfig.loadPromise) {
            return modelConfig.loadPromise;
        }

        // Check memory and evict if necessary
        await this.checkMemoryAndEvict(modelId);

        // Create and load model instance
        modelConfig.loadPromise = this.loadModel(modelConfig);
        
        try {
            const instance = await modelConfig.loadPromise;
            modelConfig.instance = instance;
            modelConfig.lastUsed = Date.now();
            modelConfig.loadPromise = null;
            
            this.loadedModels.add(modelId);
            
            this.emit('modelLoaded', { modelId, memoryUsage: this.memoryUsage });
            
            return instance;
            
        } catch (error) {
            modelConfig.loadPromise = null;
            this.emit('modelLoadFailed', { modelId, error });
            throw error;
        }
    }

    /**
     * Load a model instance
     */
    async loadModel(modelConfig) {
        const { class: ModelClass, options } = modelConfig;
        
        // Inject dependencies
        const modelOptions = {
            ...options,
            audioContext: this.audioContext,
            transformersEnv: this.transformersEnv,
            resourceManager: this
        };
        
        const instance = new ModelClass(modelOptions);
        
        // Load the model
        await instance.load();
        
        return instance;
    }

    /**
     * Unload a specific model
     */
    async unloadModel(modelId) {
        const modelConfig = this.modelRegistry.get(modelId);
        
        if (!modelConfig || !modelConfig.instance) {
            return;
        }

        try {
            await modelConfig.instance.unload();
            modelConfig.instance = null;
            modelConfig.lastUsed = 0;
            
            this.loadedModels.delete(modelId);
            
            this.emit('modelUnloaded', { modelId });
            
        } catch (error) {
            this.emit('error', { type: 'unload', modelId, error });
        }
    }

    /**
     * Get shared pipeline (cached)
     */
    async getPipeline(task, modelName, options = {}) {
        const cacheKey = `${task}-${modelName}-${JSON.stringify(options)}`;
        
        if (this.pipelineCache.has(cacheKey)) {
            return this.pipelineCache.get(cacheKey);
        }

        if (!window.transformers || !window.transformers.pipeline) {
            throw new Error('Transformers.js not available');
        }

        const pipelineOptions = {
            device: this.options.preferredDevice,
            ...options
        };

        const pipeline = await window.transformers.pipeline(task, modelName, pipelineOptions);
        
        this.pipelineCache.set(cacheKey, pipeline);
        
        return pipeline;
    }

    /**
     * Check memory usage and evict models if necessary
     */
    async checkMemoryAndEvict(modelToLoad) {
        // Estimate memory usage (simplified)
        this.memoryUsage = this.loadedModels.size * 100; // Rough estimate
        
        if (this.memoryUsage > this.options.memoryThresholdMB) {
            this.emit('memoryPressure', { 
                usedMB: this.memoryUsage, 
                threshold: this.options.memoryThresholdMB 
            });
            
            // Find least recently used model
            let oldestModel = null;
            let oldestTime = Date.now();
            
            for (const [modelId, config] of this.modelRegistry) {
                if (config.instance && config.lastUsed < oldestTime && modelId !== modelToLoad) {
                    oldestModel = modelId;
                    oldestTime = config.lastUsed;
                }
            }
            
            if (oldestModel) {
                await this.unloadModel(oldestModel);
                this.emit('modelEvicted', { modelId: oldestModel });
            }
        }
    }

    /**
     * Start resource monitoring
     */
    startResourceMonitoring() {
        if (this.resourceMonitor) {
            return;
        }

        this.resourceMonitor = setInterval(() => {
            this.checkIdleModels();
        }, this.options.modelCacheTimeout / 2);
    }

    /**
     * Check for idle models and unload them
     */
    async checkIdleModels() {
        const now = Date.now();
        
        for (const [modelId, config] of this.modelRegistry) {
            if (config.instance && 
                config.lastUsed > 0 && 
                now - config.lastUsed > this.options.modelCacheTimeout) {
                
                await this.unloadModel(modelId);
                this.emit('modelEvicted', { modelId, reason: 'timeout' });
            }
        }
    }

    /**
     * Get all loaded models
     */
    getLoadedModels() {
        return Array.from(this.loadedModels);
    }

    /**
     * Get resource status
     */
    getStatus() {
        return {
            audioContext: !!this.audioContext,
            mlContext: !!this.transformersEnv,
            loadedModels: this.getLoadedModels(),
            memoryUsage: this.memoryUsage,
            memoryThreshold: this.options.memoryThresholdMB,
            preferredDevice: this.options.preferredDevice
        };
    }

    /**
     * Clean up all resources
     */
    async cleanup() {
        // Stop monitoring
        if (this.resourceMonitor) {
            clearInterval(this.resourceMonitor);
            this.resourceMonitor = null;
        }

        // Unload all models
        for (const modelId of this.loadedModels) {
            await this.unloadModel(modelId);
        }

        // Clear caches
        this.pipelineCache.clear();
        this.modelRegistry.clear();
        this.loadedModels.clear();

        // Close audio context
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        this.emit('cleanup');
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }

    /**
     * Create appropriate VAD instance based on browser capabilities
     */
    createVAD(options = {}) {
        const features = BrowserCompatibility.detectFeatures();
        
        // Merge with audio context
        const vadOptions = {
            audioContext: this.audioContext,
            ...options
        };
        
        if (features.audioWorklet) {
            console.log('🎤 Creating Modern VAD (AudioWorklet)');
            return new ModernVoiceActivityDetector(vadOptions);
        } else {
            console.log('🎤 Creating Legacy VAD (ScriptProcessorNode)');
            return new VoiceActivityDetector(vadOptions);
        }
    }

    /**
     * Create appropriate AudioQueue instance based on browser capabilities
     */
    createAudioQueue(options = {}) {
        const features = BrowserCompatibility.detectFeatures();
        
        // Merge with audio context
        const queueOptions = {
            audioContext: this.audioContext,
            ...options
        };
        
        if (features.audioWorklet) {
            console.log('🔊 Creating Modern AudioQueue (AudioWorklet)');
            return new ModernAudioQueue(queueOptions);
        } else {
            console.log('🔊 Creating Legacy AudioQueue');
            return new AudioQueue(queueOptions);
        }
    }
}

/**
 * Base class for models that use dependency injection
 */
export class BaseModel extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = options;
        this.resourceManager = options.resourceManager;
        this.audioContext = options.audioContext;
        this.transformersEnv = options.transformersEnv;
        
        this.isModelLoaded = false;
        this.loadingPromise = null;
    }

    /**
     * Get shared pipeline from resource manager
     */
    async getPipeline(task, modelName, options = {}) {
        if (!this.resourceManager) {
            throw new Error('Resource manager not available');
        }
        
        return this.resourceManager.getPipeline(task, modelName, options);
    }

    /**
     * Base load method - override in subclasses
     */
    async load() {
        throw new Error('load() method must be implemented by subclass');
    }

    /**
     * Base unload method - override in subclasses
     */
    async unload() {
        this.isModelLoaded = false;
        this.loadingPromise = null;
        
        this.emit('unloaded', { module: this.constructor.name });
    }

    /**
     * Check if model is loaded
     */
    isLoaded() {
        return this.isModelLoaded;
    }

    /**
     * Emit custom events
     */
    emit(eventType, detail = {}) {
        this.dispatchEvent(new CustomEvent(eventType, { detail }));
    }
}
