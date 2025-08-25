// Test Advanced FaceFormer with Multiple Backends
// Tests WebGPU, WebNN, ONNX Runtime Web, and WASM fallback

const AdvancedFaceFormerWeb = require('./advanced_faceformer_web.js');

class BackendTester {
    constructor() {
        this.testResults = {};
    }

    async testAllBackends() {
        console.log('🧪 Testing Advanced FaceFormer with All Backends');
        console.log('=' * 60);

        const backends = ['webgpu', 'webnn', 'onnxruntime-web', 'wasm'];
        const datasets = ['vocaset', 'biwi'];

        for (const dataset of datasets) {
            console.log(`\n📊 Testing ${dataset.toUpperCase()} dataset:`);
            console.log('-' * 40);

            for (const backend of backends) {
                try {
                    await this.testBackend(backend, dataset);
                } catch (error) {
                    console.error(`❌ ${backend} test failed:`, error.message);
                    this.testResults[`${dataset}_${backend}`] = {
                        success: false,
                        error: error.message
                    };
                }
            }
        }

        this.printSummary();
    }

    async testBackend(backend, dataset) {
        console.log(`\n🔧 Testing ${backend} with ${dataset}...`);

        const faceformer = new AdvancedFaceFormerWeb();

        // Initialize backend
        const initStart = performance.now();
        const success = await faceformer.initialize(dataset, backend);
        const initTime = performance.now() - initStart;

        if (!success) {
            throw new Error(`Failed to initialize ${backend} backend`);
        }

        console.log(`  ✅ Initialized in ${initTime.toFixed(2)}ms`);
        console.log(`  🔧 Active backend: ${faceformer.activeBackend}`);

        // Test generation
        const config = faceformer.config;
        const sequenceLength = 5; // Short test sequence
        
        // Generate mock data
        const audioFeatures = this.generateMockAudio(sequenceLength, config.audio_input_dim);
        const template = this.generateMockTemplate(config.vertice_dim);
        const subjectId = 0;

        // Measure generation performance
        const genStart = performance.now();
        const result = await faceformer.generateVertices(audioFeatures, template, subjectId);
        const genTime = performance.now() - genStart;

        console.log(`  🎭 Generated ${result.vertices.length} frames in ${genTime.toFixed(2)}ms`);
        console.log(`  ⚡ Throughput: ${(sequenceLength / (genTime / 1000)).toFixed(1)} fps`);
        console.log(`  🎯 Backend used: ${result.backend}`);

        // Validate results
        this.validateResults(result, sequenceLength, config.vertice_dim);

        // Store results
        this.testResults[`${dataset}_${backend}`] = {
            success: true,
            initTime: initTime,
            genTime: genTime,
            throughput: sequenceLength / (genTime / 1000),
            backend: result.backend,
            frames: result.vertices.length,
            vertexDim: config.vertice_dim
        };

        console.log(`  ✅ ${backend} test completed successfully`);
    }

    generateMockAudio(sequenceLength, audioDim) {
        const features = [];
        for (let t = 0; t < sequenceLength; t++) {
            const frame = [];
            for (let i = 0; i < audioDim; i++) {
                frame.push(Math.random() * 0.1 - 0.05);
            }
            features.push(frame);
        }
        return features;
    }

    generateMockTemplate(verticeDim) {
        const template = [];
        for (let i = 0; i < verticeDim; i++) {
            template.push(Math.random() * 0.01);
        }
        return template;
    }

    validateResults(result, expectedFrames, expectedVertexDim) {
        if (result.vertices.length !== expectedFrames) {
            throw new Error(`Expected ${expectedFrames} frames, got ${result.vertices.length}`);
        }

        if (result.vertices[0].length !== expectedVertexDim) {
            throw new Error(`Expected ${expectedVertexDim} vertices, got ${result.vertices[0].length}`);
        }

        // Check for reasonable values
        const firstFrame = result.vertices[0];
        const avgValue = firstFrame.reduce((a, b) => a + b, 0) / firstFrame.length;
        
        if (Math.abs(avgValue) > 10) {
            console.warn(`⚠️ Unusual average vertex value: ${avgValue.toFixed(4)}`);
        }

        console.log(`  ✓ Results validated: ${result.vertices.length} frames × ${expectedVertexDim} vertices`);
    }

    printSummary() {
        console.log('\n📈 Test Summary');
        console.log('=' * 60);

        const datasets = ['vocaset', 'biwi'];
        const backends = ['webgpu', 'webnn', 'onnxruntime-web', 'wasm'];

        for (const dataset of datasets) {
            console.log(`\n${dataset.toUpperCase()} Results:`);
            console.log('-' * 30);

            for (const backend of backends) {
                const key = `${dataset}_${backend}`;
                const result = this.testResults[key];

                if (result && result.success) {
                    console.log(`  ✅ ${backend.padEnd(15)}: ${result.genTime.toFixed(2)}ms (${result.throughput.toFixed(1)} fps)`);
                } else {
                    const error = result ? result.error : 'Not tested';
                    console.log(`  ❌ ${backend.padEnd(15)}: ${error}`);
                }
            }
        }

        // Performance comparison
        console.log('\n🏆 Performance Ranking');
        console.log('-' * 30);

        const successfulTests = Object.entries(this.testResults)
            .filter(([_, result]) => result.success)
            .sort(([_, a], [__, b]) => a.genTime - b.genTime);

        successfulTests.forEach(([key, result], index) => {
            const [dataset, backend] = key.split('_');
            console.log(`  ${index + 1}. ${backend} (${dataset}): ${result.genTime.toFixed(2)}ms`);
        });

        // Backend availability
        console.log('\n🔧 Backend Availability');
        console.log('-' * 30);

        const backendSuccessCount = {};
        backends.forEach(backend => {
            backendSuccessCount[backend] = 0;
            datasets.forEach(dataset => {
                const key = `${dataset}_${backend}`;
                if (this.testResults[key] && this.testResults[key].success) {
                    backendSuccessCount[backend]++;
                }
            });
        });

        backends.forEach(backend => {
            const successCount = backendSuccessCount[backend];
            const total = datasets.length;
            const percentage = (successCount / total * 100).toFixed(0);
            console.log(`  ${backend.padEnd(15)}: ${successCount}/${total} (${percentage}%)`);
        });
    }

    async demonstrateMemoryUsage() {
        console.log('\n💾 Memory Usage Demonstration');
        console.log('=' * 40);

        const datasets = [
            { name: 'vocaset', vertices: 15069 },
            { name: 'biwi', vertices: 70110 }
        ];

        const sequenceLengths = [1, 10, 50, 100];

        for (const dataset of datasets) {
            console.log(`\n${dataset.name.toUpperCase()} Memory Usage:`);
            
            for (const seqLen of sequenceLengths) {
                const vertexBytes = dataset.vertices * 4; // 4 bytes per float
                const frameBytes = vertexBytes;
                const totalBytes = frameBytes * seqLen;
                const totalMB = totalBytes / (1024 * 1024);

                console.log(`  ${seqLen.toString().padStart(3)} frames: ${totalMB.toFixed(1)}MB`);
            }
        }

        console.log('\n💡 Modern web runtimes can easily handle these sizes:');
        console.log('  - WebGPU: Up to several GB of GPU memory');
        console.log('  - WebNN: Hardware-optimized memory management');
        console.log('  - ONNX Runtime Web: Efficient memory pooling');
        console.log('  - WASM: Up to 4GB memory space with 64-bit pointers');
    }
}

async function main() {
    console.log('🎮 Advanced FaceFormer Backend Testing Suite');
    console.log('Testing WebGPU, WebNN, ONNX Runtime Web, and WASM');
    console.log('');

    const tester = new BackendTester();

    try {
        await tester.testAllBackends();
        await tester.demonstrateMemoryUsage();
    } catch (error) {
        console.error('💥 Test suite failed:', error);
        process.exit(1);
    }

    console.log('\n🎉 All tests completed!');
}

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { BackendTester };
