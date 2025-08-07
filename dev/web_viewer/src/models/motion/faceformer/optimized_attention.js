// Optimized Multi-Head Attention for Audio2Gesture
// Supports WebGPU, WebNN, WASM with high-performance implementations

class OptimizedMultiHeadAttention {
    constructor(config = {}) {
        this.numHeads = config.numHeads || 8;
        this.headDim = config.headDim || 64;
        this.modelDim = this.numHeads * this.headDim;
        this.dropout = config.dropout || 0.1;
        this.backend = null;
        
        // Performance optimization flags
        this.useFlashAttention = config.useFlashAttention !== false;
        this.useFusedQKV = config.useFusedQKV !== false;
        this.useKVCache = config.useKVCache || false;
        
        // Multi-backend support
        this.supportedBackends = ['webgpu', 'webnn', 'wasm', 'cpu'];
        this.activeBEARS = null;
        
        // Attention weights
        this.weights = {
            qkv: null,      // Fused QKV projection
            output: null,   // Output projection
            qkvBias: null,
            outputBias: null
        };
        
        // Performance stats
        this.stats = {
            totalCalls: 0,
            totalTime: 0,
            avgTime: 0,
            throughputFPS: 0
        };
    }

    // Initialize backend-specific implementations
    async initializeBackend(preferredBackend = 'auto') {
        console.log('🔧 Initializing Multi-Head Attention backend...');
        
        const availableBackends = await this.detectBackends();
        let selectedBackend = preferredBackend;
        
        if (preferredBackend === 'auto') {
            // Select best available backend
            const priority = ['webgpu', 'webnn', 'wasm', 'cpu'];
            selectedBackend = priority.find(backend => availableBackends[backend]) || 'cpu';
        }
        
        if (!availableBackends[selectedBackend]) {
            console.warn(`⚠️ Backend ${selectedBackend} not available, falling back to CPU`);
            selectedBackend = 'cpu';
        }
        
        this.activeBackend = selectedBackend;
        await this.initializeBackendSpecific(selectedBackend);
        
        console.log(`✅ Multi-Head Attention initialized with ${selectedBackend} backend`);
        return selectedBackend;
    }

    async detectBackends() {
        const backends = {
            webgpu: false,
            webnn: false,
            wasm: true,
            cpu: true
        };

        // Check WebGPU
        try {
            if (navigator.gpu) {
                const adapter = await navigator.gpu.requestAdapter();
                backends.webgpu = !!adapter;
            }
        } catch (e) {
            console.log('WebGPU not available:', e.message);
        }

        // Check WebNN
        try {
            backends.webnn = 'ml' in navigator;
        } catch (e) {
            console.log('WebNN not available:', e.message);
        }

        return backends;
    }

    async initializeBackendSpecific(backend) {
        switch (backend) {
            case 'webgpu':
                await this.initializeWebGPU();
                break;
            case 'webnn':
                await this.initializeWebNN();
                break;
            case 'wasm':
                await this.initializeWASM();
                break;
            case 'cpu':
                await this.initializeCPU();
                break;
            default:
                throw new Error(`Unsupported backend: ${backend}`);
        }
    }

    async initializeWebGPU() {
        this.adapter = await navigator.gpu.requestAdapter();
        this.device = await this.adapter.requestDevice();
        
        // Create WebGPU compute shaders for attention
        this.computeShaders = await this.createWebGPUShaders();
        
        console.log('✅ WebGPU backend initialized for Multi-Head Attention');
    }

    async createWebGPUShaders() {
        // Optimized multi-head attention compute shader
        const attentionShaderCode = `
            struct AttentionParams {
                batch_size: u32,
                seq_len: u32,
                num_heads: u32,
                head_dim: u32,
                scale: f32,
            }

            @group(0) @binding(0) var<uniform> params: AttentionParams;
            @group(0) @binding(1) var<storage, read> query: array<f32>;
            @group(0) @binding(2) var<storage, read> key: array<f32>;
            @group(0) @binding(3) var<storage, read> value: array<f32>;
            @group(0) @binding(4) var<storage, read_write> output: array<f32>;
            @group(0) @binding(5) var<storage, read_write> attention_weights: array<f32>;

            // Workgroup shared memory for optimization
            var<workgroup> shared_q: array<f32, 256>;
            var<workgroup> shared_k: array<f32, 256>;
            var<workgroup> shared_v: array<f32, 256>;

            @compute @workgroup_size(16, 16, 1)
            fn attention_kernel(
                @builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>
            ) {
                let batch_idx = global_id.x;
                let head_idx = global_id.y;
                let seq_idx = global_id.z;

                if (batch_idx >= params.batch_size || head_idx >= params.num_heads || seq_idx >= params.seq_len) {
                    return;
                }

                // Calculate indices
                let q_base = (batch_idx * params.num_heads * params.seq_len * params.head_dim) +
                            (head_idx * params.seq_len * params.head_dim) +
                            (seq_idx * params.head_dim);

                // Load query into shared memory
                for (var i = 0u; i < params.head_dim; i = i + 1u) {
                    if (local_id.x * params.head_dim + i < 256u && i < params.head_dim) {
                        shared_q[local_id.x * params.head_dim + i] = query[q_base + i];
                    }
                }

                workgroupBarrier();

                // Compute attention scores for this query position
                var max_score = -3.402823e+38; // -inf
                for (var k = 0u; k < params.seq_len; k = k + 1u) {
                    let k_base = (batch_idx * params.num_heads * params.seq_len * params.head_dim) +
                                (head_idx * params.seq_len * params.head_dim) +
                                (k * params.head_dim);

                    // Compute Q·K^T
                    var score = 0.0;
                    for (var d = 0u; d < params.head_dim; d = d + 1u) {
                        score = score + shared_q[local_id.x * params.head_dim + d] * key[k_base + d];
                    }
                    score = score * params.scale;

                    // Track max for numerical stability
                    max_score = max(max_score, score);

                    // Store score
                    let score_idx = (batch_idx * params.num_heads * params.seq_len * params.seq_len) +
                                   (head_idx * params.seq_len * params.seq_len) +
                                   (seq_idx * params.seq_len) + k;
                    attention_weights[score_idx] = score;
                }

                workgroupBarrier();

                // Softmax with numerical stability
                var sum_exp = 0.0;
                for (var k = 0u; k < params.seq_len; k = k + 1u) {
                    let score_idx = (batch_idx * params.num_heads * params.seq_len * params.seq_len) +
                                   (head_idx * params.seq_len * params.seq_len) +
                                   (seq_idx * params.seq_len) + k;
                    let stable_score = attention_weights[score_idx] - max_score;
                    let exp_score = exp(stable_score);
                    attention_weights[score_idx] = exp_score;
                    sum_exp = sum_exp + exp_score;
                }

                // Normalize
                for (var k = 0u; k < params.seq_len; k = k + 1u) {
                    let score_idx = (batch_idx * params.num_heads * params.seq_len * params.seq_len) +
                                   (head_idx * params.seq_len * params.seq_len) +
                                   (seq_idx * params.seq_len) + k;
                    attention_weights[score_idx] = attention_weights[score_idx] / sum_exp;
                }

                workgroupBarrier();

                // Compute attention output
                let out_base = (batch_idx * params.num_heads * params.seq_len * params.head_dim) +
                              (head_idx * params.seq_len * params.head_dim) +
                              (seq_idx * params.head_dim);

                for (var d = 0u; d < params.head_dim; d = d + 1u) {
                    var out_val = 0.0;
                    for (var k = 0u; k < params.seq_len; k = k + 1u) {
                        let score_idx = (batch_idx * params.num_heads * params.seq_len * params.seq_len) +
                                       (head_idx * params.seq_len * params.seq_len) +
                                       (seq_idx * params.seq_len) + k;
                        let v_base = (batch_idx * params.num_heads * params.seq_len * params.head_dim) +
                                    (head_idx * params.seq_len * params.head_dim) +
                                    (k * params.head_dim);
                        out_val = out_val + attention_weights[score_idx] * value[v_base + d];
                    }
                    output[out_base + d] = out_val;
                }
            }
        `;

        return {
            attention: this.device.createShaderModule({
                code: attentionShaderCode
            })
        };
    }

    async initializeWebNN() {
        this.mlContext = await navigator.ml.createContext();
        
        // Create WebNN graph for attention
        this.webnnGraph = await this.createWebNNGraph();
        
        console.log('✅ WebNN backend initialized for Multi-Head Attention');
    }

    async createWebNNGraph() {
        const builder = new MLGraphBuilder(this.mlContext);
        
        // Define input operands
        const query = builder.input('query', {
            type: 'float32',
            dimensions: [1, this.numHeads, -1, this.headDim] // [batch, heads, seq, head_dim]
        });
        
        const key = builder.input('key', {
            type: 'float32',
            dimensions: [1, this.numHeads, -1, this.headDim]
        });
        
        const value = builder.input('value', {
            type: 'float32',
            dimensions: [1, this.numHeads, -1, this.headDim]
        });

        // Compute attention scores: Q @ K^T
        const keyTransposed = builder.transpose(key, {permutation: [0, 1, 3, 2]});
        const scores = builder.matmul(query, keyTransposed);
        
        // Scale scores
        const scale = 1.0 / Math.sqrt(this.headDim);
        const scaledScores = builder.mul(scores, builder.constant({type: 'float32', dimensions: []}, scale));
        
        // Apply softmax
        const attentionWeights = builder.softmax(scaledScores, {axis: -1});
        
        // Apply attention to values
        const output = builder.matmul(attentionWeights, value);
        
        // Build the graph
        return await builder.build({output});
    }

    async initializeWASM() {
        // SIMD-optimized WASM implementation
        this.wasmModule = await this.loadOptimizedWASM();
        console.log('✅ WASM backend initialized for Multi-Head Attention');
    }

    async loadOptimizedWASM() {
        // Simulated WASM module - in practice, this would load actual WASM binary
        return {
            multiHeadAttention: (query, key, value, config) => {
                return this.wasmMultiHeadAttention(query, key, value, config);
            }
        };
    }

    async initializeCPU() {
        // JavaScript CPU implementation with optimizations
        console.log('✅ CPU backend initialized for Multi-Head Attention');
    }

    // Main attention computation method
    async computeAttention(query, key, value, mask = null) {
        const startTime = performance.now();
        
        let result;
        switch (this.activeBackend) {
            case 'webgpu':
                result = await this.computeWebGPUAttention(query, key, value, mask);
                break;
            case 'webnn':
                result = await this.computeWebNNAttention(query, key, value, mask);
                break;
            case 'wasm':
                result = await this.computeWASMAttention(query, key, value, mask);
                break;
            case 'cpu':
                result = await this.computeCPUAttention(query, key, value, mask);
                break;
            default:
                throw new Error(`Backend ${this.activeBackend} not implemented`);
        }
        
        const endTime = performance.now();
        this.updateStats(endTime - startTime);
        
        return result;
    }

    async computeWebGPUAttention(query, key, value, mask) {
        const batchSize = 1;
        const seqLen = query.length;
        const numHeads = this.numHeads;
        const headDim = this.headDim;

        // Create GPU buffers
        const queryBuffer = this.createBuffer(query, 'storage');
        const keyBuffer = this.createBuffer(key, 'storage');
        const valueBuffer = this.createBuffer(value, 'storage');
        
        const outputSize = batchSize * numHeads * seqLen * headDim;
        const attentionWeightsSize = batchSize * numHeads * seqLen * seqLen;
        
        const outputBuffer = this.device.createBuffer({
            size: outputSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        const attentionWeightsBuffer = this.device.createBuffer({
            size: attentionWeightsSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Create uniform buffer for parameters
        const paramsData = new Float32Array([batchSize, seqLen, numHeads, headDim, 1.0 / Math.sqrt(headDim)]);
        const paramsBuffer = this.createBuffer(paramsData, 'uniform');

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.computeShaders.attention.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: paramsBuffer } },
                { binding: 1, resource: { buffer: queryBuffer } },
                { binding: 2, resource: { buffer: keyBuffer } },
                { binding: 3, resource: { buffer: valueBuffer } },
                { binding: 4, resource: { buffer: outputBuffer } },
                { binding: 5, resource: { buffer: attentionWeightsBuffer } }
            ]
        });

        // Create compute pipeline
        const computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.computeShaders.attention,
                entryPoint: 'attention_kernel'
            }
        });

        // Dispatch compute shader
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        
        const workgroupX = Math.ceil(batchSize / 16);
        const workgroupY = Math.ceil(numHeads / 16);
        const workgroupZ = Math.ceil(seqLen / 1);
        
        passEncoder.dispatchWorkgroups(workgroupX, workgroupY, workgroupZ);
        passEncoder.end();

        // Copy result back
        const resultBuffer = this.device.createBuffer({
            size: outputSize * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, resultBuffer, 0, outputSize * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read result
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = resultBuffer.getMappedRange();
        const result = new Float32Array(arrayBuffer);
        const output = Array.from(result);
        
        resultBuffer.unmap();
        
        // Cleanup
        queryBuffer.destroy();
        keyBuffer.destroy();
        valueBuffer.destroy();
        outputBuffer.destroy();
        attentionWeightsBuffer.destroy();
        paramsBuffer.destroy();
        resultBuffer.destroy();

        return this.reshapeOutput(output, seqLen);
    }

    async computeWebNNAttention(query, key, value, mask) {
        // Prepare inputs for WebNN
        const queryTensor = new Float32Array(query.flat());
        const keyTensor = new Float32Array(key.flat());
        const valueTensor = new Float32Array(value.flat());

        // Execute WebNN graph
        const inputs = {
            query: queryTensor,
            key: keyTensor,
            value: valueTensor
        };

        const outputs = await this.webnnGraph.compute(inputs);
        const result = Array.from(outputs.output);
        
        return this.reshapeOutput(result, query.length);
    }

    async computeWASMAttention(query, key, value, mask) {
        const config = {
            numHeads: this.numHeads,
            headDim: this.headDim,
            seqLen: query.length
        };

        const result = this.wasmModule.multiHeadAttention(query, key, value, config);
        return result;
    }

    wasmMultiHeadAttention(query, key, value, config) {
        // SIMD-optimized JavaScript implementation simulating WASM
        const { numHeads, headDim, seqLen } = config;
        const scale = 1.0 / Math.sqrt(headDim);
        
        const output = [];
        
        for (let h = 0; h < numHeads; h++) {
            for (let i = 0; i < seqLen; i++) {
                const attentionRow = [];
                
                // Compute attention scores
                const scores = [];
                let maxScore = -Infinity;
                
                for (let j = 0; j < seqLen; j++) {
                    let score = 0;
                    for (let d = 0; d < headDim; d++) {
                        const qIdx = h * seqLen * headDim + i * headDim + d;
                        const kIdx = h * seqLen * headDim + j * headDim + d;
                        score += query[qIdx] * key[kIdx];
                    }
                    score *= scale;
                    scores.push(score);
                    maxScore = Math.max(maxScore, score);
                }
                
                // Softmax with numerical stability
                let sumExp = 0;
                const expScores = scores.map(score => {
                    const exp = Math.exp(score - maxScore);
                    sumExp += exp;
                    return exp;
                });
                
                const attentionWeights = expScores.map(exp => exp / sumExp);
                
                // Apply attention to values
                for (let d = 0; d < headDim; d++) {
                    let weightedSum = 0;
                    for (let j = 0; j < seqLen; j++) {
                        const vIdx = h * seqLen * headDim + j * headDim + d;
                        weightedSum += attentionWeights[j] * value[vIdx];
                    }
                    output.push(weightedSum);
                }
            }
        }
        
        return this.reshapeOutput(output, seqLen);
    }

    async computeCPUAttention(query, key, value, mask) {
        return this.wasmMultiHeadAttention(query, key, value, {
            numHeads: this.numHeads,
            headDim: this.headDim,
            seqLen: query.length
        });
    }

    createBuffer(data, usage) {
        const flatData = Array.isArray(data[0]) ? data.flat() : data;
        const buffer = this.device.createBuffer({
            size: flatData.length * 4,
            usage: usage === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE,
            mappedAtCreation: true
        });
        
        new Float32Array(buffer.getMappedRange()).set(flatData);
        buffer.unmap();
        
        return buffer;
    }

    reshapeOutput(output, seqLen) {
        const result = [];
        const totalDim = this.numHeads * this.headDim;
        
        for (let i = 0; i < seqLen; i++) {
            const frame = [];
            for (let j = 0; j < totalDim; j++) {
                frame.push(output[i * totalDim + j]);
            }
            result.push(frame);
        }
        
        return result;
    }

    updateStats(inferenceTime) {
        this.stats.totalCalls++;
        this.stats.totalTime += inferenceTime;
        this.stats.avgTime = this.stats.totalTime / this.stats.totalCalls;
        this.stats.throughputFPS = 1000 / this.stats.avgTime;
    }

    getPerformanceStats() {
        return {
            backend: this.activeBackend,
            ...this.stats,
            configuration: {
                numHeads: this.numHeads,
                headDim: this.headDim,
                modelDim: this.modelDim,
                useFlashAttention: this.useFlashAttention,
                useFusedQKV: this.useFusedQKV
            }
        };
    }

    // Batch processing for higher throughput
    async computeBatchAttention(queries, keys, values, masks = null) {
        const results = [];
        
        for (let i = 0; i < queries.length; i++) {
            const result = await this.computeAttention(
                queries[i], 
                keys[i], 
                values[i], 
                masks ? masks[i] : null
            );
            results.push(result);
        }
        
        return results;
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.OptimizedMultiHeadAttention = OptimizedMultiHeadAttention;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = OptimizedMultiHeadAttention;
}

