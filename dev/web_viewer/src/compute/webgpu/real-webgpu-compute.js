/**
 * Real WebGPU Compute Module
 * This file implements actual GPU computation using WebGPU API
 */

class RealWebGPUCompute {
    constructor() {
        this.device = null;
        this.adapter = null;
        this.initialized = false;
        this.capabilities = {};
    }

    async initialize() {
        try {
            console.log('[Real WebGPU] Initializing WebGPU...');

            // Check WebGPU availability
            if (!navigator.gpu) {
                throw new Error('WebGPU is not supported in this browser');
            }

            // Request adapter with explicit options
            this.adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance',
                forceFallbackAdapter: false
            });

            if (!this.adapter) {
                throw new Error('Failed to obtain WebGPU adapter - GPU may not be available');
            }

            console.log('[Real WebGPU] Adapter obtained, requesting device...');

            // Get adapter info (with fallback for older WebGPU implementations)
            let adapterInfo = {};
            try {
                if (this.adapter.requestAdapterInfo) {
                    adapterInfo = await this.adapter.requestAdapterInfo();
                    console.log('[Real WebGPU] Adapter info:', adapterInfo);
                } else if (this.adapter.info) {
                    // Fallback for older WebGPU spec
                    adapterInfo = this.adapter.info;
                    console.log('[Real WebGPU] Adapter info (legacy):', adapterInfo);
                } else {
                    console.log('[Real WebGPU] Adapter info not available, using defaults');
                }
            } catch (error) {
                console.warn('[Real WebGPU] Failed to get adapter info:', error.message);
                adapterInfo = { vendor: 'unknown', architecture: 'unknown' };
            }

            // Request device with required features
            const requiredFeatures = [];
            if (this.adapter.features.has('shader-f16')) {
                requiredFeatures.push('shader-f16');
            }
            if (this.adapter.features.has('timestamp-query')) {
                requiredFeatures.push('timestamp-query');
            }

            this.device = await this.adapter.requestDevice({
                requiredFeatures,
                requiredLimits: {
                    maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
                    maxComputeWorkgroupStorageSize: this.adapter.limits.maxComputeWorkgroupStorageSize,
                    maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup,
                }
            });

            console.log('[Real WebGPU] Device obtained successfully');

            // Set up error handling
            this.device.addEventListener('uncapturederror', (event) => {
                console.error('[Real WebGPU] Uncaptured error:', event.error);
            });

            // Store capabilities
            this.capabilities = {
                maxBufferSize: this.device.limits.maxStorageBufferBindingSize,
                maxWorkgroupSize: this.device.limits.maxComputeWorkgroupSizeX,
                maxInvocations: this.device.limits.maxComputeInvocationsPerWorkgroup,
                features: Array.from(this.device.features),
                vendor: adapterInfo.vendor || 'unknown',
                architecture: adapterInfo.architecture || 'unknown'
            };

            this.initialized = true;
            console.log('[Real WebGPU] Initialization complete:', this.capabilities);

            return {
                success: true,
                capabilities: this.capabilities,
                webgpuSupport: true,
                vendor: this.capabilities.vendor,
                architecture: this.capabilities.architecture
            };

        } catch (error) {
            console.error('[Real WebGPU] Initialization failed:', error);
            this.initialized = false;
            return {
                success: false,
                error: error.message,
                webgpuSupport: false,
                fallbackToCPU: true
            };
        }
    }

    async performMatrixMultiplication(size) {
        if (!this.initialized) {
            throw new Error('WebGPU not initialized');
        }

        console.log(`[Real WebGPU] Performing ${size}x${size} matrix multiplication on GPU`);

        const startTime = performance.now();

        // Create compute shader for matrix multiplication
        const shaderModule = this.device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
                @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;
                @group(0) @binding(3) var<uniform> size: u32;

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let row = global_id.x;
                    let col = global_id.y;
                    
                    if (row >= size || col >= size) {
                        return;
                    }
                    
                    var sum: f32 = 0.0;
                    for (var k: u32 = 0u; k < size; k = k + 1u) {
                        sum = sum + matrixA[row * size + k] * matrixB[k * size + col];
                    }
                    
                    result[row * size + col] = sum;
                }
            `
        });

        // Create buffers
        const matrixSize = size * size * 4; // 4 bytes per f32
        
        const matrixABuffer = this.device.createBuffer({
            size: matrixSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const matrixBBuffer = this.device.createBuffer({
            size: matrixSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const resultBuffer = this.device.createBuffer({
            size: matrixSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const sizeBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const readBuffer = this.device.createBuffer({
            size: matrixSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Generate random matrices
        const matrixA = new Float32Array(size * size);
        const matrixB = new Float32Array(size * size);
        
        for (let i = 0; i < size * size; i++) {
            matrixA[i] = (Math.random() - 0.5) * 2.0;
            matrixB[i] = (Math.random() - 0.5) * 2.0;
        }

        // Upload data to GPU
        this.device.queue.writeBuffer(matrixABuffer, 0, matrixA);
        this.device.queue.writeBuffer(matrixBBuffer, 0, matrixB);
        this.device.queue.writeBuffer(sizeBuffer, 0, new Uint32Array([size]));

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                ]
            }),
            entries: [
                { binding: 0, resource: { buffer: matrixABuffer } },
                { binding: 1, resource: { buffer: matrixBBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: sizeBuffer } },
            ]
        });

        // Create compute pipeline
        const computePipeline = this.device.createComputePipeline({
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroup.layout]
            })
        });

        // Execute computation
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        
        const workgroupsX = Math.ceil(size / 8);
        const workgroupsY = Math.ceil(size / 8);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
        
        passEncoder.end();

        // Copy result back
        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, matrixSize);

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);

        // Read result
        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultArray = new Float32Array(readBuffer.getMappedRange());
        
        // Verify a few elements (for debugging)
        const sampleResults = Array.from(resultArray.slice(0, 9));
        
        readBuffer.unmap();

        const executionTime = performance.now() - startTime;
        
        // Calculate performance metrics
        const operations = 2 * size * size * size; // FLOPS for matrix multiply
        const flops = operations / (executionTime / 1000);
        const bandwidth = (matrixSize * 3) / (executionTime / 1000) / (1024 * 1024 * 1024); // GB/s

        // Cleanup
        matrixABuffer.destroy();
        matrixBBuffer.destroy();
        resultBuffer.destroy();
        sizeBuffer.destroy();
        readBuffer.destroy();

        return {
            type: 'gpu_matrix_computation',
            matrix_size: size,
            operations_count: operations,
            flops_achieved: flops,
            memory_bandwidth_gb_s: bandwidth,
            execution_time_ms: executionTime,
            sample_results: sampleResults,
            gpu_vendor: this.capabilities.vendor,
            gpu_architecture: this.capabilities.architecture,
            actual_webgpu_execution: true,
            shader_used: 'matrix_multiply_compute',
            workgroups_dispatched: workgroupsX * workgroupsY
        };
    }

    async performVectorAddition(size) {
        if (!this.initialized) {
            throw new Error('WebGPU not initialized');
        }

        console.log(`[Real WebGPU] Performing vector addition of ${size} elements on GPU`);

        const startTime = performance.now();

        // Simple vector addition shader
        const shaderModule = this.device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> vectorA: array<f32>;
                @group(0) @binding(1) var<storage, read> vectorB: array<f32>;
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&vectorA)) {
                        return;
                    }
                    result[index] = vectorA[index] + vectorB[index];
                }
            `
        });

        const bufferSize = size * 4; // 4 bytes per f32
        
        // Create buffers
        const vectorABuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const vectorBBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const resultBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const readBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Generate test data
        const vectorA = new Float32Array(size);
        const vectorB = new Float32Array(size);
        
        for (let i = 0; i < size; i++) {
            vectorA[i] = Math.random() * 100;
            vectorB[i] = Math.random() * 100;
        }

        // Upload data
        this.device.queue.writeBuffer(vectorABuffer, 0, vectorA);
        this.device.queue.writeBuffer(vectorBBuffer, 0, vectorB);

        // Create compute pipeline
        const computePipeline = this.device.createComputePipeline({
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
            layout: 'auto'
        });

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: vectorABuffer } },
                { binding: 1, resource: { buffer: vectorBBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
            ]
        });

        // Execute
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(size / 64));
        passEncoder.end();

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, bufferSize);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read result
        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange());
        
        // Verify correctness
        let correctCount = 0;
        for (let i = 0; i < Math.min(100, size); i++) {
            const expected = vectorA[i] + vectorB[i];
            const actual = result[i];
            if (Math.abs(expected - actual) < 0.001) {
                correctCount++;
            }
        }

        readBuffer.unmap();

        const executionTime = performance.now() - startTime;

        // Cleanup
        vectorABuffer.destroy();
        vectorBBuffer.destroy();
        resultBuffer.destroy();
        readBuffer.destroy();

        return {
            type: 'gpu_vector_computation',
            vector_size: size,
            execution_time_ms: executionTime,
            correctness_rate: correctCount / Math.min(100, size),
            operations_per_second: size / (executionTime / 1000),
            actual_webgpu_execution: true,
            shader_used: 'vector_addition_compute'
        };
    }

    destroy() {
        if (this.device) {
            this.device.destroy();
            this.device = null;
            this.adapter = null;
            this.initialized = false;
            console.log('[Real WebGPU] Device destroyed');
        }
    }
}

// Export for use in workers
if (typeof self !== 'undefined') {
    self.RealWebGPUCompute = RealWebGPUCompute;
}

// Export for Node.js testing
if (typeof module !== 'undefined') {
    module.exports = { RealWebGPUCompute };
}
