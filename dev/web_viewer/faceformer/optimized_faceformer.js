// Integrated FaceFormer with Optimized Multi-Head Attention
// Combines existing FaceFormer with performance-optimized attention for increased FPS

class OptimizedFaceFormer {
    constructor(config = {}) {
        this.config = {
            numLayers: config.numLayers || 6,
            numHeads: config.numHeads || 8,
            headDim: config.headDim || 64,
            dropout: config.dropout || 0.1,
            chunkSize: config.chunkSize || 100,
            maxSequenceLength: config.maxSequenceLength || 2000,
            preferredBackend: config.preferredBackend || 'auto',
            enableCaching: config.enableCaching !== false,
            enableProfiling: config.enableProfiling !== false
        };

        this.modelDim = this.config.numHeads * this.config.headDim;
        this.layers = [];
        this.isInitialized = false;
        this.backend = null;
        this.cache = new Map();
        this.profiler = new PerformanceProfiler();
    }

    async initialize() {
        console.log('🚀 Initializing Optimized FaceFormer...');

        try {
            // Initialize attention layers
            for (let i = 0; i < this.config.numLayers; i++) {
                const attention = new OptimizedMultiHeadAttention({
                    numHeads: this.config.numHeads,
                    headDim: this.config.headDim,
                    dropout: this.config.dropout
                });
                
                await attention.initializeBackend(this.config.preferredBackend);
                this.layers.push({
                    attention: attention,
                    layerNorm1: new LayerNorm(this.modelDim),
                    layerNorm2: new LayerNorm(this.modelDim),
                    feedForward: new FeedForward(this.modelDim)
                });
            }

            this.backend = this.layers[0].attention.currentBackend;
            this.isInitialized = true;
            
            console.log(`✅ Optimized FaceFormer initialized with ${this.config.numLayers} layers using ${this.backend} backend`);
            
        } catch (error) {
            console.error('❌ Failed to initialize Optimized FaceFormer:', error);
            throw error;
        }
    }

    async processAudioToGesture(audioFeatures, options = {}) {
        if (!this.isInitialized) {
            throw new Error('FaceFormer not initialized. Call initialize() first.');
        }

        const startTime = performance.now();
        this.profiler.startSession('audio_to_gesture');

        try {
            // Validate input
            const sequenceLength = audioFeatures.length;
            if (sequenceLength === 0) {
                throw new Error('Empty audio features provided');
            }

            if (sequenceLength > this.config.maxSequenceLength) {
                console.warn(`Sequence length ${sequenceLength} exceeds maximum ${this.config.maxSequenceLength}, using chunked processing`);
                return await this.processLongSequence(audioFeatures, options);
            }

            // Process through transformer layers
            let hidden = audioFeatures;
            
            for (let layerIdx = 0; layerIdx < this.layers.length; layerIdx++) {
                const layer = this.layers[layerIdx];
                hidden = await this.processLayer(hidden, layer, layerIdx);
                
                this.profiler.recordEvent(`layer_${layerIdx}_complete`, performance.now() - startTime);
            }

            // Convert to gesture parameters
            const gestureParams = this.hiddenToGesture(hidden);
            
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            
            this.profiler.endSession('audio_to_gesture', {
                processingTime,
                sequenceLength,
                fps: 1000 / processingTime,
                backend: this.backend
            });

            if (this.config.enableProfiling) {
                console.log(`⚡ Processed ${sequenceLength} frames in ${processingTime.toFixed(2)}ms (${(1000/processingTime).toFixed(1)} FPS)`);
            }

            return {
                gestureParams,
                metadata: {
                    processingTime,
                    fps: 1000 / processingTime,
                    backend: this.backend,
                    sequenceLength
                }
            };

        } catch (error) {
            this.profiler.recordError('audio_to_gesture', error);
            throw error;
        }
    }

    async processLayer(hidden, layer, layerIdx) {
        const layerStartTime = performance.now();

        try {
            // Self-attention with residual connection and layer norm
            const attentionStart = performance.now();
            const attended = await layer.attention.computeAttention(hidden, hidden, hidden);
            const attentionTime = performance.now() - attentionStart;
            
            const residual1 = this.addResidual(hidden, attended);
            const normed1 = layer.layerNorm1.forward(residual1);
            
            // Feed-forward with residual connection and layer norm
            const ffStart = performance.now();
            const feedForwardOutput = layer.feedForward.forward(normed1);
            const ffTime = performance.now() - ffStart;
            
            const residual2 = this.addResidual(normed1, feedForwardOutput);
            const output = layer.layerNorm2.forward(residual2);

            const layerTime = performance.now() - layerStartTime;
            
            this.profiler.recordEvent(`layer_${layerIdx}_attention`, attentionTime);
            this.profiler.recordEvent(`layer_${layerIdx}_feedforward`, ffTime);
            this.profiler.recordEvent(`layer_${layerIdx}_total`, layerTime);

            return output;

        } catch (error) {
            console.error(`Error in layer ${layerIdx}:`, error);
            throw error;
        }
    }

    async processLongSequence(audioFeatures, options = {}) {
        console.log(`📦 Processing long sequence of ${audioFeatures.length} frames in chunks`);
        
        const chunkSize = options.chunkSize || this.config.chunkSize;
        const chunks = this.createChunks(audioFeatures, chunkSize);
        const results = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const chunkResult = await this.processAudioToGesture(chunk, { ...options, isChunk: true });
            results.push(chunkResult.gestureParams);
            
            if (this.config.enableProfiling && i % 10 === 0) {
                console.log(`📊 Processed chunk ${i + 1}/${chunks.length}`);
            }
        }

        // Combine chunk results
        const combinedGestureParams = this.combineChunkResults(results);
        
        return {
            gestureParams: combinedGestureParams,
            metadata: {
                totalChunks: chunks.length,
                chunkSize: chunkSize,
                backend: this.backend,
                sequenceLength: audioFeatures.length
            }
        };
    }

    createChunks(sequence, chunkSize) {
        const chunks = [];
        for (let i = 0; i < sequence.length; i += chunkSize) {
            chunks.push(sequence.slice(i, i + chunkSize));
        }
        return chunks;
    }

    combineChunkResults(results) {
        // Concatenate all chunk results
        const combined = [];
        for (const result of results) {
            combined.push(...result);
        }
        return combined;
    }

    addResidual(input, output) {
        const result = [];
        for (let i = 0; i < input.length; i++) {
            const row = [];
            for (let j = 0; j < input[i].length; j++) {
                row.push(input[i][j] + output[i][j]);
            }
            result.push(row);
        }
        return result;
    }

    hiddenToGesture(hidden) {
        // Convert transformer hidden states to facial gesture parameters
        // This includes jaw, lip, eyebrow, and eye movements
        
        const gestureParams = [];
        
        for (let i = 0; i < hidden.length; i++) {
            const frame = hidden[i];
            
            // Extract different gesture components from hidden state
            const jawParams = this.extractJawParams(frame);
            const lipParams = this.extractLipParams(frame);
            const eyebrowParams = this.extractEyebrowParams(frame);
            const eyeParams = this.extractEyeParams(frame);
            
            gestureParams.push({
                jaw: jawParams,
                lips: lipParams,
                eyebrows: eyebrowParams,
                eyes: eyeParams,
                timestamp: i
            });
        }
        
        return gestureParams;
    }

    extractJawParams(hiddenFrame) {
        // Extract jaw movement parameters from hidden state
        const startIdx = 0;
        const endIdx = 6;
        const rawParams = hiddenFrame.slice(startIdx, endIdx);
        
        return {
            open: this.sigmoid(rawParams[0]) * 0.8,      // Jaw opening (0-0.8)
            shift_x: this.tanh(rawParams[1]) * 0.3,      // Horizontal shift
            shift_y: this.tanh(rawParams[2]) * 0.2,      // Vertical shift
            rotation: this.tanh(rawParams[3]) * 0.1,     // Rotation
            protrusion: this.sigmoid(rawParams[4]) * 0.5, // Forward protrusion
            asymmetry: this.tanh(rawParams[5]) * 0.2     // Left-right asymmetry
        };
    }

    extractLipParams(hiddenFrame) {
        // Extract lip movement parameters
        const startIdx = 6;
        const endIdx = 18;
        const rawParams = hiddenFrame.slice(startIdx, endIdx);
        
        return {
            pucker: this.sigmoid(rawParams[0]) * 0.7,        // Lip pucker
            spread: this.sigmoid(rawParams[1]) * 0.6,        // Lip spread
            upper_raise: this.sigmoid(rawParams[2]) * 0.5,   // Upper lip raise
            lower_depress: this.sigmoid(rawParams[3]) * 0.5, // Lower lip depress
            corner_pull: this.tanh(rawParams[4]) * 0.4,      // Corner pull
            tighten: this.sigmoid(rawParams[5]) * 0.3,       // Lip tightening
            roll_upper: this.sigmoid(rawParams[6]) * 0.3,    // Upper lip roll
            roll_lower: this.sigmoid(rawParams[7]) * 0.3,    // Lower lip roll
            press: this.sigmoid(rawParams[8]) * 0.4,         // Lip press
            part: this.sigmoid(rawParams[9]) * 0.5,          // Lip parting
            funnel: this.sigmoid(rawParams[10]) * 0.4,       // Lip funnel
            asymmetry: this.tanh(rawParams[11]) * 0.2        // Asymmetry
        };
    }

    extractEyebrowParams(hiddenFrame) {
        // Extract eyebrow movement parameters
        const startIdx = 18;
        const endIdx = 24;
        const rawParams = hiddenFrame.slice(startIdx, endIdx);
        
        return {
            inner_up: this.sigmoid(rawParams[0]) * 0.6,      // Inner brow up
            outer_up: this.sigmoid(rawParams[1]) * 0.5,      // Outer brow up
            down: this.sigmoid(rawParams[2]) * 0.4,          // Brow down
            squeeze: this.sigmoid(rawParams[3]) * 0.3,       // Brow squeeze
            asymmetry_lr: this.tanh(rawParams[4]) * 0.3,     // Left-right asymmetry
            asymmetry_ud: this.tanh(rawParams[5]) * 0.2      // Up-down asymmetry
        };
    }

    extractEyeParams(hiddenFrame) {
        // Extract eye movement parameters
        const startIdx = 24;
        const endIdx = 32;
        const rawParams = hiddenFrame.slice(startIdx, endIdx);
        
        return {
            blink: this.sigmoid(rawParams[0]) * 0.9,         // Blink amount
            squint: this.sigmoid(rawParams[1]) * 0.4,        // Squint
            wide: this.sigmoid(rawParams[2]) * 0.3,          // Eye wide
            gaze_x: this.tanh(rawParams[3]) * 0.5,           // Horizontal gaze
            gaze_y: this.tanh(rawParams[4]) * 0.3,           // Vertical gaze
            upper_lid: this.sigmoid(rawParams[5]) * 0.4,     // Upper lid position
            lower_lid: this.sigmoid(rawParams[6]) * 0.3,     // Lower lid position
            asymmetry: this.tanh(rawParams[7]) * 0.2         // Eye asymmetry
        };
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    tanh(x) {
        return Math.tanh(x);
    }

    async switchBackend(backend) {
        console.log(`🔄 Switching to ${backend} backend...`);
        
        try {
            for (const layer of this.layers) {
                await layer.attention.initializeBackend(backend);
            }
            this.backend = backend;
            console.log(`✅ Successfully switched to ${backend} backend`);
        } catch (error) {
            console.error(`❌ Failed to switch to ${backend} backend:`, error);
            throw error;
        }
    }

    getPerformanceReport() {
        return this.profiler.generateReport();
    }

    clearCache() {
        this.cache.clear();
        console.log('🧹 Cache cleared');
    }

    getSystemInfo() {
        return {
            config: this.config,
            backend: this.backend,
            isInitialized: this.isInitialized,
            modelDim: this.modelDim,
            numParameters: this.estimateParameterCount(),
            memoryUsage: this.estimateMemoryUsage()
        };
    }

    estimateParameterCount() {
        // Rough estimation of model parameters
        const attentionParams = this.config.numLayers * (
            this.modelDim * this.modelDim * 4 + // Q, K, V, O projections
            this.modelDim * 4 // biases
        );
        
        const ffParams = this.config.numLayers * (
            this.modelDim * this.modelDim * 4 + // Two linear layers
            this.modelDim * 2 // biases
        );
        
        const normParams = this.config.numLayers * this.modelDim * 4; // LayerNorm parameters
        
        return attentionParams + ffParams + normParams;
    }

    estimateMemoryUsage() {
        // Rough estimation of memory usage in MB
        const paramCount = this.estimateParameterCount();
        const paramMemory = paramCount * 4 / (1024 * 1024); // 4 bytes per float32
        
        const activationMemory = this.config.maxSequenceLength * this.modelDim * 4 / (1024 * 1024);
        
        return {
            parameters: paramMemory,
            activations: activationMemory,
            total: paramMemory + activationMemory
        };
    }
}

// Simple implementations of missing components
class LayerNorm {
    constructor(dim, eps = 1e-5) {
        this.dim = dim;
        this.eps = eps;
        this.weight = new Array(dim).fill(1.0);
        this.bias = new Array(dim).fill(0.0);
    }

    forward(input) {
        const result = [];
        
        for (let i = 0; i < input.length; i++) {
            const row = input[i];
            
            // Compute mean and variance
            const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
            const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
            const std = Math.sqrt(variance + this.eps);
            
            // Normalize
            const normalizedRow = row.map((val, j) => 
                ((val - mean) / std) * this.weight[j] + this.bias[j]
            );
            
            result.push(normalizedRow);
        }
        
        return result;
    }
}

class FeedForward {
    constructor(dim, hiddenDim = null) {
        this.dim = dim;
        this.hiddenDim = hiddenDim || dim * 4;
        
        // Initialize weights randomly
        this.weights1 = this.initializeWeights(this.dim, this.hiddenDim);
        this.bias1 = new Array(this.hiddenDim).fill(0);
        this.weights2 = this.initializeWeights(this.hiddenDim, this.dim);
        this.bias2 = new Array(this.dim).fill(0);
    }

    initializeWeights(inputDim, outputDim) {
        const weights = [];
        const scale = Math.sqrt(2 / inputDim); // He initialization
        
        for (let i = 0; i < inputDim; i++) {
            const row = [];
            for (let j = 0; j < outputDim; j++) {
                row.push((Math.random() - 0.5) * 2 * scale);
            }
            weights.push(row);
        }
        
        return weights;
    }

    forward(input) {
        // First linear layer + ReLU
        const hidden = this.linearLayer(input, this.weights1, this.bias1);
        const activated = this.relu(hidden);
        
        // Second linear layer
        const output = this.linearLayer(activated, this.weights2, this.bias2);
        
        return output;
    }

    linearLayer(input, weights, bias) {
        const result = [];
        
        for (let i = 0; i < input.length; i++) {
            const row = [];
            
            for (let j = 0; j < weights[0].length; j++) {
                let sum = bias[j];
                for (let k = 0; k < input[i].length; k++) {
                    sum += input[i][k] * weights[k][j];
                }
                row.push(sum);
            }
            
            result.push(row);
        }
        
        return result;
    }

    relu(input) {
        return input.map(row => row.map(val => Math.max(0, val)));
    }
}

class PerformanceProfiler {
    constructor() {
        this.sessions = {};
        this.events = [];
    }

    startSession(name) {
        this.sessions[name] = {
            startTime: performance.now(),
            events: []
        };
    }

    recordEvent(name, timestamp) {
        this.events.push({
            name,
            timestamp,
            time: timestamp
        });
    }

    recordError(sessionName, error) {
        if (this.sessions[sessionName]) {
            this.sessions[sessionName].error = error.message;
        }
    }

    endSession(name, metadata = {}) {
        if (this.sessions[name]) {
            this.sessions[name].endTime = performance.now();
            this.sessions[name].duration = this.sessions[name].endTime - this.sessions[name].startTime;
            this.sessions[name].metadata = metadata;
        }
    }

    generateReport() {
        return {
            sessions: this.sessions,
            events: this.events,
            summary: this.generateSummary()
        };
    }

    generateSummary() {
        const sessionNames = Object.keys(this.sessions);
        const summary = {};
        
        for (const name of sessionNames) {
            const session = this.sessions[name];
            if (session.duration) {
                summary[name] = {
                    avgDuration: session.duration,
                    count: 1,
                    metadata: session.metadata
                };
            }
        }
        
        return summary;
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.OptimizedFaceFormer = OptimizedFaceFormer;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = OptimizedFaceFormer;
}
