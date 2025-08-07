/**
 * System Performance Analyzer
 * Comprehensive system information gathering and performance analysis
 */

class SystemPerformanceAnalyzer {
    constructor() {
        this.systemInfo = {};
        this.performanceMetrics = {};
        this.capabilities = {};
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        await this.gatherSystemInfo();
        await this.detectCapabilities();
        await this.benchmarkBaseline();
        
        this.initialized = true;
        console.log('System Performance Analyzer initialized');
    }

    async gatherSystemInfo() {
        this.systemInfo = {
            // Browser and platform info
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            cookieEnabled: navigator.cookieEnabled,
            onLine: navigator.onLine,
            
            // Hardware info
            hardwareConcurrency: navigator.hardwareConcurrency || 'unknown',
            maxTouchPoints: navigator.maxTouchPoints || 0,
            
            // Screen info
            screen: {
                width: screen.width,
                height: screen.height,
                colorDepth: screen.colorDepth,
                pixelDepth: screen.pixelDepth,
                availWidth: screen.availWidth,
                availHeight: screen.availHeight
            },
            
            // Device memory (if available)
            deviceMemory: navigator.deviceMemory || 'unknown',
            
            // Connection info
            connection: this.getConnectionInfo(),
            
            // Performance memory info
            memory: this.getMemoryInfo(),
            
            // Timing info
            timing: this.getTimingInfo(),
            
            // WebGL info
            webgl: await this.getWebGLInfo(),
            
            // WebGPU info
            webgpu: await this.getWebGPUInfo(),
            
            // WebNN info
            webnn: await this.getWebNNInfo()
        };
    }

    getConnectionInfo() {
        if ('connection' in navigator) {
            const conn = navigator.connection;
            return {
                effectiveType: conn.effectiveType,
                downlink: conn.downlink,
                rtt: conn.rtt,
                saveData: conn.saveData,
                type: conn.type
            };
        }
        return { available: false };
    }

    getMemoryInfo() {
        if (performance.memory) {
            return {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit,
                available: true
            };
        }
        return { available: false };
    }

    getTimingInfo() {
        if (performance.timing) {
            const timing = performance.timing;
            return {
                navigationStart: timing.navigationStart,
                domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                loadComplete: timing.loadEventEnd - timing.navigationStart,
                available: true
            };
        }
        return { available: false };
    }

    async getWebGLInfo() {
        try {
            const canvas = new OffscreenCanvas(1, 1);
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            
            if (!gl) {
                return { available: false };
            }

            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            
            return {
                available: true,
                version: gl.getParameter(gl.VERSION),
                vendor: gl.getParameter(gl.VENDOR),
                renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown',
                shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
                maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
                maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
                maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS),
                extensions: gl.getSupportedExtensions()
            };
        } catch (error) {
            return { available: false, error: error.message };
        }
    }

    async getWebGPUInfo() {
        try {
            if (!navigator.gpu) {
                return { available: false, reason: 'WebGPU not supported' };
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { available: false, reason: 'No GPU adapter found' };
            }

            const adapterInfo = await adapter.requestAdapterInfo();
            const features = Array.from(adapter.features);
            const limits = adapter.limits;

            // Try to get device for more detailed info
            let deviceInfo = null;
            try {
                const device = await adapter.requestDevice();
                deviceInfo = {
                    label: device.label,
                    features: Array.from(device.features),
                    limits: device.limits
                };
                device.destroy();
            } catch (deviceError) {
                console.warn('Could not create WebGPU device:', deviceError);
            }

            return {
                available: true,
                adapter: {
                    vendor: adapterInfo.vendor,
                    architecture: adapterInfo.architecture,
                    device: adapterInfo.device,
                    description: adapterInfo.description,
                    features: features,
                    limits: Object.fromEntries(
                        Object.entries(limits).map(([key, value]) => [key, value])
                    )
                },
                device: deviceInfo
            };
        } catch (error) {
            return { available: false, error: error.message };
        }
    }

    async getWebNNInfo() {
        try {
            if (!navigator.ml) {
                return { available: false, reason: 'WebNN not supported' };
            }

            const context = await navigator.ml.createContext();
            
            return {
                available: true,
                context: {
                    type: 'WebNN',
                    created: true
                }
            };
        } catch (error) {
            return { available: false, error: error.message };
        }
    }

    async detectCapabilities() {
        this.capabilities = {
            // Compute capabilities
            multiThreading: this.systemInfo.hardwareConcurrency > 1,
            webWorkers: typeof Worker !== 'undefined',
            sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
            
            // Graphics capabilities
            webgl: this.systemInfo.webgl.available,
            webgl2: this.systemInfo.webgl.available && this.systemInfo.webgl.version.includes('2'),
            webgpu: this.systemInfo.webgpu.available,
            
            // AI/ML capabilities
            webnn: this.systemInfo.webnn.available,
            
            // Memory capabilities
            performanceMemory: this.systemInfo.memory.available,
            deviceMemory: this.systemInfo.deviceMemory !== 'unknown',
            
            // Storage capabilities
            localStorage: typeof localStorage !== 'undefined',
            indexedDB: typeof indexedDB !== 'undefined',
            
            // Network capabilities
            fetch: typeof fetch !== 'undefined',
            webrtc: typeof RTCPeerConnection !== 'undefined',
            
            // Advanced features
            simd: typeof WebAssembly !== 'undefined' && WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x60, 0x01, 0x7b, 0x00
            ])),
            
            atomics: typeof Atomics !== 'undefined'
        };
    }

    async benchmarkBaseline() {
        const results = {};
        
        // CPU baseline
        results.cpu = await this.benchmarkCPU();
        
        // Memory baseline
        results.memory = await this.benchmarkMemory();
        
        // JavaScript engine performance
        results.jsEngine = await this.benchmarkJSEngine();
        
        // DOM/Canvas performance
        results.rendering = await this.benchmarkRendering();
        
        this.performanceMetrics.baseline = results;
    }

    async benchmarkCPU() {
        const startTime = performance.now();
        const iterations = 1000000;
        let result = 0;
        
        // Mathematical operations
        for (let i = 0; i < iterations; i++) {
            result += Math.sin(i) * Math.cos(i) + Math.sqrt(i);
        }
        
        const mathTime = performance.now() - startTime;
        
        // Array operations
        const arrayStart = performance.now();
        const arr = new Array(100000);
        for (let i = 0; i < arr.length; i++) {
            arr[i] = Math.random();
        }
        arr.sort();
        const arrayTime = performance.now() - arrayStart;
        
        return {
            mathOpsPerSecond: iterations / (mathTime / 1000),
            arrayTime: arrayTime,
            baselineScore: 1000 / mathTime // Higher is better
        };
    }

    async benchmarkMemory() {
        const sizes = [1024, 64 * 1024, 1024 * 1024]; // 1KB, 64KB, 1MB
        const results = [];
        
        for (const size of sizes) {
            const iterations = Math.max(10, Math.floor(10000000 / size));
            
            const src = new Uint8Array(size);
            const dst = new Uint8Array(size);
            src.fill(42);
            
            const startTime = performance.now();
            
            for (let i = 0; i < iterations; i++) {
                dst.set(src);
            }
            
            const endTime = performance.now();
            const duration = (endTime - startTime) / 1000;
            const bandwidth = (size * iterations * 2) / duration / 1024 / 1024; // MB/s
            
            results.push({
                size: size,
                bandwidth: bandwidth,
                latency: duration / iterations * 1000 // ms
            });
        }
        
        return {
            measurements: results,
            peakBandwidth: Math.max(...results.map(r => r.bandwidth))
        };
    }

    async benchmarkJSEngine() {
        // Object creation and property access
        const objStart = performance.now();
        const objects = [];
        for (let i = 0; i < 100000; i++) {
            objects.push({ id: i, value: Math.random(), active: true });
        }
        const objTime = performance.now() - objStart;
        
        // Function calls
        const funcStart = performance.now();
        const testFunc = (a, b, c) => a * b + c;
        let funcResult = 0;
        for (let i = 0; i < 1000000; i++) {
            funcResult += testFunc(i, i + 1, i + 2);
        }
        const funcTime = performance.now() - funcStart;
        
        // JSON operations
        const jsonStart = performance.now();
        const testData = { numbers: Array.from({length: 10000}, (_, i) => i) };
        const serialized = JSON.stringify(testData);
        const parsed = JSON.parse(serialized);
        const jsonTime = performance.now() - jsonStart;
        
        return {
            objectCreation: objTime,
            functionCalls: funcTime,
            jsonSerialization: jsonTime,
            overallScore: 1000 / (objTime + funcTime + jsonTime)
        };
    }

    async benchmarkRendering() {
        try {
            const canvas = new OffscreenCanvas(256, 256);
            const ctx = canvas.getContext('2d');
            
            const startTime = performance.now();
            
            // Drawing operations
            for (let i = 0; i < 1000; i++) {
                ctx.fillStyle = `hsl(${i % 360}, 50%, 50%)`;
                ctx.fillRect(Math.random() * 256, Math.random() * 256, 10, 10);
            }
            
            // Image data operations
            const imageData = ctx.getImageData(0, 0, 256, 256);
            for (let i = 0; i < imageData.data.length; i += 4) {
                imageData.data[i] = (imageData.data[i] + 10) % 255; // R
            }
            ctx.putImageData(imageData, 0, 0);
            
            const endTime = performance.now();
            
            return {
                renderingTime: endTime - startTime,
                pixelsProcessed: 256 * 256,
                renderingScore: 1000 / (endTime - startTime)
            };
        } catch (error) {
            return { available: false, error: error.message };
        }
    }

    calculateRealtimeCapabilities() {
        const analysis = {
            animation60fps: { capable: false, confidence: 0 },
            audio48khz: { capable: false, confidence: 0 },
            videoProcessing: { capable: false, confidence: 0 },
            mlInference: { capable: false, confidence: 0 },
            recommendations: []
        };

        if (!this.performanceMetrics.baseline) {
            return analysis;
        }

        const baseline = this.performanceMetrics.baseline;
        
        // 60 FPS animation capability (16.67ms per frame)
        const animationScore = baseline.cpu?.baselineScore || 0;
        const memoryScore = baseline.memory?.peakBandwidth || 0;
        
        if (animationScore > 50 && memoryScore > 500) {
            analysis.animation60fps.capable = true;
            analysis.animation60fps.confidence = Math.min(1, (animationScore / 100 + memoryScore / 1000) / 2);
        }
        
        // Audio processing capability
        if (baseline.cpu?.mathOpsPerSecond > 100000) {
            analysis.audio48khz.capable = true;
            analysis.audio48khz.confidence = Math.min(1, baseline.cpu.mathOpsPerSecond / 500000);
        }
        
        // Video processing capability
        if (this.capabilities.webgl && memoryScore > 1000 && animationScore > 30) {
            analysis.videoProcessing.capable = true;
            analysis.videoProcessing.confidence = Math.min(1, memoryScore / 2000);
        }
        
        // ML inference capability
        if (this.capabilities.webnn || this.capabilities.webgpu) {
            analysis.mlInference.capable = true;
            analysis.mlInference.confidence = this.capabilities.webnn ? 0.9 : 0.7;
        }
        
        // Generate recommendations
        if (analysis.animation60fps.capable) {
            analysis.recommendations.push('System suitable for 60 FPS animations');
        }
        if (analysis.videoProcessing.capable) {
            analysis.recommendations.push('System suitable for real-time video processing');
        }
        if (analysis.mlInference.capable) {
            analysis.recommendations.push('System suitable for ML inference tasks');
        }
        if (!analysis.animation60fps.capable) {
            analysis.recommendations.push('Consider reducing animation complexity for smooth performance');
        }
        
        return analysis;
    }

    getOptimalWorkerConfiguration() {
        const config = {
            cpuWorkers: 2,
            gpuWorkers: 0,
            webnnWorkers: 0,
            wasmWorkers: 1
        };
        
        const cores = this.systemInfo.hardwareConcurrency;
        if (cores && cores > 2) {
            config.cpuWorkers = Math.min(cores - 1, 4); // Leave one core free
        }
        
        if (this.capabilities.webgpu) {
            config.gpuWorkers = 1;
        }
        
        if (this.capabilities.webnn) {
            config.webnnWorkers = 1;
        }
        
        // WASM workers based on CPU cores
        if (cores && cores > 4) {
            config.wasmWorkers = 2;
        }
        
        return config;
    }

    generatePerformanceReport() {
        return {
            timestamp: Date.now(),
            systemInfo: this.systemInfo,
            capabilities: this.capabilities,
            performanceMetrics: this.performanceMetrics,
            realtimeCapabilities: this.calculateRealtimeCapabilities(),
            optimalConfiguration: this.getOptimalWorkerConfiguration(),
            recommendations: this.generateRecommendations()
        };
    }

    generateRecommendations() {
        const recommendations = [];
        
        // Memory recommendations
        if (this.systemInfo.deviceMemory && this.systemInfo.deviceMemory < 4) {
            recommendations.push({
                type: 'memory',
                level: 'warning',
                message: 'Low device memory detected. Consider reducing concurrent tasks.',
                impact: 'medium'
            });
        }
        
        // GPU recommendations
        if (!this.capabilities.webgpu && this.capabilities.webgl) {
            recommendations.push({
                type: 'graphics',
                level: 'info',
                message: 'WebGPU not available, but WebGL is supported. Consider WebGL fallback for GPU tasks.',
                impact: 'low'
            });
        }
        
        // CPU recommendations
        const cores = this.systemInfo.hardwareConcurrency;
        if (cores && cores <= 2) {
            recommendations.push({
                type: 'cpu',
                level: 'warning',
                message: 'Limited CPU cores detected. Avoid heavy parallel processing.',
                impact: 'high'
            });
        }
        
        // WebNN recommendations
        if (this.capabilities.webnn) {
            recommendations.push({
                type: 'ml',
                level: 'success',
                message: 'WebNN available for optimized ML inference. Prefer WebNN for AI tasks.',
                impact: 'high'
            });
        }
        
        return recommendations;
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SystemPerformanceAnalyzer };
} else if (typeof window !== 'undefined') {
    window.SystemPerformanceAnalyzer = SystemPerformanceAnalyzer;
}
