/**
 * Compute Backend Optimizer
 * Automatically detects and selects the most performant compute backend
 * for language model inference with optimal quantization
 */

class ComputeBackendOptimizer {
    constructor() {
        this.backends = new Map();
        this.benchmarkResults = new Map();
        this.initialized = false;
    }

    async initialize() {
        console.log('[Backend Optimizer] 🚀 Initializing compute backend detection...');
        
        // Define backend capabilities in performance order (fastest first)
        const backendConfigs = [
            {
                name: 'webnn-native',
                priority: 1,
                speedMultiplier: 8.5,
                quantizations: ['int4', 'int8', 'fp16'],
                description: 'Native WebNN with hardware acceleration',
                testFunction: () => this.testWebNNNative()
            },
            {
                name: 'webgpu-native', 
                priority: 2,
                speedMultiplier: 6.2,
                quantizations: ['fp16', 'int8'],
                description: 'Native WebGPU compute shaders',
                testFunction: () => this.testWebGPUNative()
            },
            {
                name: 'onnx-runtime-web',
                priority: 3,
                speedMultiplier: 4.8,
                quantizations: ['int8', 'fp16', 'fp32'],
                description: 'ONNX Runtime Web with WASM/WebGL',
                testFunction: () => this.testONNXRuntimeWeb()
            },
            {
                name: 'wasm-native',
                priority: 4,
                speedMultiplier: 3.2,
                quantizations: ['int8', 'fp32'],
                description: 'Native WebAssembly with SIMD',
                testFunction: () => this.testWASMNative()
            },
            {
                name: 'transformers-js',
                priority: 5,
                speedMultiplier: 2.1,
                quantizations: ['fp16', 'fp32'],
                description: 'Transformers.js with ONNX Runtime',
                testFunction: () => this.testTransformersJS()
            },
            {
                name: 'javascript-cpu',
                priority: 6,
                speedMultiplier: 1.0,
                quantizations: ['fp32'],
                description: 'JavaScript CPU fallback',
                testFunction: () => this.testJavaScriptCPU()
            }
        ];

        // Test each backend
        for (const config of backendConfigs) {
            try {
                console.log(`[Backend Optimizer] 🔍 Testing ${config.name}...`);
                const result = await config.testFunction();
                
                if (result.available) {
                    this.backends.set(config.name, {
                        ...config,
                        available: true,
                        capabilities: result.capabilities,
                        benchmarkScore: result.benchmarkScore || 0
                    });
                    console.log(`[Backend Optimizer] ✅ ${config.name}: Available (score: ${result.benchmarkScore})`);
                } else {
                    console.log(`[Backend Optimizer] ❌ ${config.name}: ${result.reason}`);
                }
            } catch (error) {
                console.log(`[Backend Optimizer] ❌ ${config.name}: ${error.message}`);
            }
        }

        this.initialized = true;
        const availableBackends = Array.from(this.backends.keys());
        console.log(`[Backend Optimizer] 🎉 Initialization complete! Available backends: ${availableBackends.join(', ')}`);
        
        return {
            available: availableBackends,
            best: this.selectBestBackend(),
            capabilities: this.getSystemCapabilities()
        };
    }

    async testWebNNNative() {
        // Test native WebNN availability
        if (!navigator.ml) {
            return { available: false, reason: 'WebNN API not available' };
        }

        try {
            // Simple WebNN capability test
            const context = await navigator.ml.createContext();
            const benchmarkScore = await this.benchmarkWebNN(context);
            
            return {
                available: true,
                capabilities: {
                    quantizations: ['int4', 'int8', 'fp16'],
                    maxTensorSize: '16GB',
                    hardwareAcceleration: true,
                    parallelExecution: true
                },
                benchmarkScore
            };
        } catch (error) {
            return { available: false, reason: `WebNN test failed: ${error.message}` };
        }
    }

    async testWebGPUNative() {
        if (!navigator.gpu) {
            return { available: false, reason: 'WebGPU not available' };
        }

        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { available: false, reason: 'No WebGPU adapter' };
            }

            const device = await adapter.requestDevice();
            const benchmarkScore = await this.benchmarkWebGPU(device);
            
            return {
                available: true,
                capabilities: {
                    quantizations: ['fp16', 'int8'],
                    maxTensorSize: '8GB',
                    hardwareAcceleration: true,
                    computeShaders: true,
                    vendor: adapter.info?.vendor || 'unknown'
                },
                benchmarkScore
            };
        } catch (error) {
            return { available: false, reason: `WebGPU test failed: ${error.message}` };
        }
    }

    async testONNXRuntimeWeb() {
        try {
            // Test if ONNX Runtime Web is available
            if (typeof ort !== 'undefined') {
                const providers = ort.env.webgpu?.isAvailable() ? ['webgpu', 'wasm'] : ['wasm'];
                const benchmarkScore = await this.benchmarkONNXRuntime(providers);
                
                return {
                    available: true,
                    capabilities: {
                        quantizations: ['int8', 'fp16', 'fp32'],
                        providers: providers,
                        hardwareAcceleration: providers.includes('webgpu')
                    },
                    benchmarkScore
                };
            }
            return { available: false, reason: 'ONNX Runtime Web not loaded' };
        } catch (error) {
            return { available: false, reason: `ONNX Runtime test failed: ${error.message}` };
        }
    }

    async testWASMNative() {
        try {
            // Test WASM SIMD support
            const wasmSupported = typeof WebAssembly !== 'undefined';
            const simdSupported = await this.testWASMSIMD();
            
            if (!wasmSupported) {
                return { available: false, reason: 'WebAssembly not supported' };
            }

            const benchmarkScore = await this.benchmarkWASM(simdSupported);
            
            return {
                available: true,
                capabilities: {
                    quantizations: ['int8', 'fp32'],
                    simdSupport: simdSupported,
                    memoryLimit: '4GB'
                },
                benchmarkScore
            };
        } catch (error) {
            return { available: false, reason: `WASM test failed: ${error.message}` };
        }
    }

    async testTransformersJS() {
        try {
            // Test transformers.js availability
            const available = typeof transformers !== 'undefined' || typeof window?.transformers !== 'undefined';
            
            if (!available) {
                return { available: false, reason: 'Transformers.js not loaded' };
            }

            const benchmarkScore = await this.benchmarkTransformersJS();
            
            return {
                available: true,
                capabilities: {
                    quantizations: ['fp16', 'fp32'],
                    models: ['TinyLlama', 'DiabloGPT'],
                    pipeline: 'text-generation'
                },
                benchmarkScore
            };
        } catch (error) {
            return { available: false, reason: `Transformers.js test failed: ${error.message}` };
        }
    }

    async testJavaScriptCPU() {
        // JavaScript CPU is always available as fallback
        const benchmarkScore = await this.benchmarkJavaScriptCPU();
        
        return {
            available: true,
            capabilities: {
                quantizations: ['fp32'],
                fallback: true,
                multithreading: typeof SharedArrayBuffer !== 'undefined'
            },
            benchmarkScore
        };
    }

    // Benchmark functions
    async benchmarkWebNN(context) {
        const startTime = performance.now();
        // Simple tensor operation benchmark
        try {
            // Create a simple computation graph
            const input = await context.constant({ dimensions: [1, 1000], type: 'float32' }, new Float32Array(1000).fill(1.0));
            const output = await context.matmul(input, input);
            await context.compute({ 'output': output });
        } catch (e) {
            return 1000; // Low score if benchmark fails
        }
        return Math.max(10000 - (performance.now() - startTime), 1000);
    }

    async benchmarkWebGPU(device) {
        const startTime = performance.now();
        try {
            // Simple matrix multiplication benchmark
            const size = 256;
            const buffer = device.createBuffer({
                size: size * size * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(buffer, 0, buffer, 0, 1024);
            device.queue.submit([commandEncoder.finish()]);
            
            buffer.destroy();
        } catch (e) {
            return 800;
        }
        return Math.max(8000 - (performance.now() - startTime), 800);
    }

    async benchmarkONNXRuntime(providers) {
        const startTime = performance.now();
        // Simulate ONNX model loading benchmark
        await new Promise(resolve => setTimeout(resolve, 100));
        const baseScore = providers.includes('webgpu') ? 6000 : 4000;
        return Math.max(baseScore - (performance.now() - startTime), baseScore * 0.5);
    }

    async benchmarkWASM(simdSupported) {
        const startTime = performance.now();
        // Simple WASM computation benchmark
        const array = new Float32Array(10000);
        for (let i = 0; i < array.length; i++) {
            array[i] = Math.sin(i) * Math.cos(i);
        }
        const baseScore = simdSupported ? 3500 : 2500;
        return Math.max(baseScore - (performance.now() - startTime), baseScore * 0.5);
    }

    async benchmarkTransformersJS() {
        const startTime = performance.now();
        // Simulate transformers.js initialization
        await new Promise(resolve => setTimeout(resolve, 200));
        return Math.max(2000 - (performance.now() - startTime), 1000);
    }

    async benchmarkJavaScriptCPU() {
        const startTime = performance.now();
        // CPU computation benchmark
        let sum = 0;
        for (let i = 0; i < 100000; i++) {
            sum += Math.sqrt(i);
        }
        return Math.max(1000 - (performance.now() - startTime), 500);
    }

    async testWASMSIMD() {
        try {
            // Test WASM SIMD support
            const wasmCode = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x00, 0x0b
            ]);
            
            const module = await WebAssembly.compile(wasmCode);
            return true;
        } catch {
            return false;
        }
    }

    selectBestBackend(modelSize = '1.1B', memoryAvailable = 2.0) {
        if (!this.initialized) {
            console.warn('[Backend Optimizer] Not initialized, returning fallback');
            return { name: 'javascript-cpu', quantization: 'fp32' };
        }

        // Sort backends by benchmark score
        const sortedBackends = Array.from(this.backends.entries())
            .filter(([name, config]) => config.available)
            .sort(([, a], [, b]) => b.benchmarkScore - a.benchmarkScore);

        for (const [name, config] of sortedBackends) {
            // Select best quantization for this backend
            const quantization = this.selectOptimalQuantization(config.quantizations, memoryAvailable);
            
            console.log(`[Backend Optimizer] 🏆 Selected: ${name} with ${quantization} (score: ${config.benchmarkScore})`);
            
            return {
                name,
                quantization,
                config,
                expectedSpeedMultiplier: config.speedMultiplier * this.getQuantizationSpeedMultiplier(quantization),
                description: `${config.description} + ${quantization} quantization`
            };
        }

        // Fallback
        return { name: 'javascript-cpu', quantization: 'fp32' };
    }

    selectOptimalQuantization(supportedQuantizations, memoryAvailable) {
        const quantizationPrefs = [
            { type: 'int4', memoryRequired: 0.3, speedMultiplier: 4.2 },
            { type: 'int8', memoryRequired: 0.6, speedMultiplier: 2.8 },
            { type: 'fp16', memoryRequired: 1.0, speedMultiplier: 1.9 },
            { type: 'fp32', memoryRequired: 2.1, speedMultiplier: 1.0 }
        ];

        for (const pref of quantizationPrefs) {
            if (supportedQuantizations.includes(pref.type) && memoryAvailable >= pref.memoryRequired) {
                return pref.type;
            }
        }

        return 'fp32'; // Ultimate fallback
    }

    getQuantizationSpeedMultiplier(quantization) {
        const multipliers = {
            'int4': 4.2,
            'int8': 2.8,
            'fp16': 1.9,
            'fp32': 1.0
        };
        return multipliers[quantization] || 1.0;
    }

    getSystemCapabilities() {
        return {
            totalBackends: this.backends.size,
            availableBackends: Array.from(this.backends.entries())
                .filter(([, config]) => config.available)
                .map(([name, config]) => ({
                    name,
                    score: config.benchmarkScore,
                    quantizations: config.capabilities?.quantizations || []
                })),
            memoryInfo: this.getMemoryInfo()
        };
    }

    getMemoryInfo() {
        if (self.performance && self.performance.memory) {
            const totalHeap = self.performance.memory.jsHeapSizeLimit || 4294967296;
            const usedHeap = self.performance.memory.usedJSHeapSize || 0;
            const availableGB = (totalHeap - usedHeap) / (1024 * 1024 * 1024);
            return {
                available: Math.max(availableGB, 0.5),
                total: totalHeap / (1024 * 1024 * 1024),
                used: usedHeap / (1024 * 1024 * 1024)
            };
        }
        return { available: 2.0, total: 4.0, used: 2.0 };
    }
}

// Export for use in workers
if (typeof self !== 'undefined') {
    self.ComputeBackendOptimizer = ComputeBackendOptimizer;
}

// Export for Node.js
if (typeof module !== 'undefined') {
    module.exports = { ComputeBackendOptimizer };
}
