// FaceFormer Weight Loader for JavaScript/Node.js
// Loads exported Python weights and applies them to ONNX models or custom implementations

class FaceFormerWeightLoader {
    constructor() {
        this.weights = null;
        this.modelInfo = null;
        this.isLoaded = false;
    }

    async loadPythonWeights(weightsPath = './faceformer_python_weights.json') {
        console.log('📥 Loading Python weights from:', weightsPath);
        
        try {
            let weightsData;
            
            // Handle both Node.js and browser environments
            if (typeof window !== 'undefined') {
                // Browser environment
                const response = await fetch(weightsPath);
                if (!response.ok) {
                    throw new Error(`Failed to fetch weights: ${response.status}`);
                }
                weightsData = await response.json();
            } else {
                // Node.js environment
                const fs = require('fs');
                const data = fs.readFileSync(weightsPath, 'utf8');
                weightsData = JSON.parse(data);
            }
            
            this.weights = weightsData.weights;
            this.modelInfo = weightsData.model_info;
            this.layerInfo = weightsData.layer_info;
            this.isLoaded = true;
            
            console.log('✅ Python weights loaded successfully');
            console.log(`📊 Model info: ${this.modelInfo.architecture}`);
            console.log(`🔧 Layers loaded: ${Object.keys(this.weights).length / 2}`); // Divide by 2 for weight+bias pairs
            
            return true;
            
        } catch (error) {
            console.error('❌ Failed to load Python weights:', error);
            return false;
        }
    }

    getLayerWeights(layerName) {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        const weightKey = `${layerName}_weight`;
        const biasKey = `${layerName}_bias`;
        
        if (!(weightKey in this.weights) || !(biasKey in this.weights)) {
            throw new Error(`Layer '${layerName}' not found in weights`);
        }
        
        return {
            weight: this.weights[weightKey],
            bias: this.weights[biasKey],
            info: this.layerInfo[layerName]
        };
    }

    getAllWeights() {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        return this.weights;
    }

    // Create ONNX-compatible tensors with the loaded weights
    createONNXTensors() {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        const tensors = {};
        
        for (const [name, data] of Object.entries(this.weights)) {
            // Convert nested arrays to flat Float32Array
            const flatData = this.flattenArray(data);
            const shape = this.getArrayShape(data);
            
            tensors[name] = {
                data: new Float32Array(flatData),
                shape: shape,
                type: 'float32'
            };
        }
        
        console.log(`✅ Created ${Object.keys(tensors).length} ONNX tensors`);
        return tensors;
    }

    flattenArray(arr) {
        const result = [];
        
        function flatten(item) {
            if (Array.isArray(item)) {
                item.forEach(flatten);
            } else {
                result.push(item);
            }
        }
        
        flatten(arr);
        return result;
    }

    getArrayShape(arr) {
        const shape = [];
        let current = arr;
        
        while (Array.isArray(current)) {
            shape.push(current.length);
            current = current[0];
        }
        
        return shape;
    }

    // Apply weights to a custom JavaScript neural network implementation
    applyToCustomModel(model) {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        console.log('🔧 Applying Python weights to custom model...');
        
        const layers = ['audio_proj', 'emb_proj', 'subject_embedding', 'fusion', 'vertex_proj', 'emb_update'];
        
        for (const layerName of layers) {
            try {
                const layerWeights = this.getLayerWeights(layerName);
                
                if (model[layerName] && typeof model[layerName].setWeights === 'function') {
                    model[layerName].setWeights(layerWeights.weight, layerWeights.bias);
                    console.log(`✅ Applied weights to ${layerName}`);
                } else {
                    console.warn(`⚠️ Layer ${layerName} not found in model or doesn't support setWeights`);
                }
            } catch (error) {
                console.error(`❌ Failed to apply weights to ${layerName}:`, error);
            }
        }
        
        console.log('✅ Weight application completed');
    }

    // Generate a test inference with the loaded weights
    async testWithLoadedWeights() {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        console.log('🧪 Testing inference with loaded Python weights...');
        
        // Create a simple test implementation
        const testModel = new SimpleFaceFormerJS();
        this.applyToCustomModel(testModel);
        
        // Create test inputs matching the Python test
        const testInputs = {
            audio_features: new Array(768).fill(0.01),
            vertice_emb: new Array(64).fill(0.01),
            one_hot: [1, 0, 0],
            template: new Array(15069).fill(0.01)
        };
        
        const outputs = testModel.forward(testInputs);
        
        console.log('✅ Test inference completed');
        console.log(`📊 Output shapes: vertices=${outputs.vertices.length}, embedding=${outputs.embedding.length}`);
        console.log(`📊 Sample vertices: [${outputs.vertices.slice(0, 3).map(v => v.toFixed(6)).join(', ')}...]`);
        console.log(`📊 Sample embedding: [${outputs.embedding.slice(0, 3).map(v => v.toFixed(6)).join(', ')}...]`);
        
        return outputs;
    }

    // Verify weight consistency by comparing with Python outputs
    async verifyConsistency(pythonOutputsPath = './python_model_outputs.json') {
        console.log('🔍 Verifying weight consistency with Python outputs...');
        
        try {
            // Load Python reference outputs
            let pythonData;
            if (typeof window !== 'undefined') {
                const response = await fetch(pythonOutputsPath);
                pythonData = await response.json();
            } else {
                const fs = require('fs');
                const data = fs.readFileSync(pythonOutputsPath, 'utf8');
                pythonData = JSON.parse(data);
            }
            
            const pythonOutputs = pythonData.simplified_results.simplified_python_outputs;
            
            // Run test with loaded weights
            const jsOutputs = await this.testWithLoadedWeights();
            
            // Compare outputs
            const vertexDiffs = [];
            const embDiffs = [];
            
            for (let i = 0; i < Math.min(pythonOutputs.new_vertice_out.length, jsOutputs.vertices.length); i++) {
                vertexDiffs.push(Math.abs(pythonOutputs.new_vertice_out[i] - jsOutputs.vertices[i]));
            }
            
            for (let i = 0; i < Math.min(pythonOutputs.updated_vertice_emb.length, jsOutputs.embedding.length); i++) {
                embDiffs.push(Math.abs(pythonOutputs.updated_vertice_emb[i] - jsOutputs.embedding[i]));
            }
            
            const maxVertexDiff = Math.max(...vertexDiffs);
            const maxEmbDiff = Math.max(...embDiffs);
            const meanVertexDiff = vertexDiffs.reduce((a, b) => a + b, 0) / vertexDiffs.length;
            const meanEmbDiff = embDiffs.reduce((a, b) => a + b, 0) / embDiffs.length;
            
            console.log('📊 Consistency verification results:');
            console.log(`  Max vertex difference: ${maxVertexDiff.toFixed(8)}`);
            console.log(`  Max embedding difference: ${maxEmbDiff.toFixed(8)}`);
            console.log(`  Mean vertex difference: ${meanVertexDiff.toFixed(8)}`);
            console.log(`  Mean embedding difference: ${meanEmbDiff.toFixed(8)}`);
            
            const tolerance = 1e-6;
            const isConsistent = maxVertexDiff < tolerance && maxEmbDiff < tolerance;
            
            if (isConsistent) {
                console.log('✅ PERFECT CONSISTENCY - Weights loaded correctly!');
            } else {
                console.log('⚠️ Some differences detected - Check implementation');
            }
            
            return {
                consistent: isConsistent,
                maxVertexDiff,
                maxEmbDiff,
                meanVertexDiff,
                meanEmbDiff
            };
            
        } catch (error) {
            console.error('❌ Consistency verification failed:', error);
            return null;
        }
    }

    // Export weights in different formats
    exportWeights(format = 'json') {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        switch (format) {
            case 'json':
                return JSON.stringify(this.weights, null, 2);
            
            case 'tensors':
                return this.createONNXTensors();
            
            case 'arrays':
                return this.weights;
                
            default:
                throw new Error(`Unknown export format: ${format}`);
        }
    }
}

// Simple JavaScript implementation of FaceFormer for testing
class SimpleFaceFormerJS {
    constructor() {
        this.layers = {
            audio_proj: new LinearLayer(768, 64),
            emb_proj: new LinearLayer(64, 64),
            subject_embedding: new LinearLayer(3, 64),
            fusion: new LinearLayer(192, 64), // 64*3
            vertex_proj: new LinearLayer(64, 15069),
            emb_update: new LinearLayer(64, 64)
        };
    }

    forward(inputs) {
        // Process inputs
        const audioProj = this.layers.audio_proj.forward(inputs.audio_features);
        const embProj = this.layers.emb_proj.forward(inputs.vertice_emb);
        const subjectEmb = this.layers.subject_embedding.forward(inputs.one_hot);
        
        // Fuse features
        const fused = [...audioProj, ...embProj, ...subjectEmb];
        const fusedOutput = this.layers.fusion.forward(fused);
        
        // Generate outputs
        const vertexDelta = this.layers.vertex_proj.forward(fusedOutput);
        const vertices = inputs.template.map((t, i) => t + vertexDelta[i]);
        const embedding = this.layers.emb_update.forward(fusedOutput);
        
        return { vertices, embedding };
    }
}

// Simple linear layer implementation
class LinearLayer {
    constructor(inputDim, outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.weight = null;
        this.bias = null;
    }

    setWeights(weight, bias) {
        this.weight = weight;
        this.bias = bias;
    }

    forward(input) {
        if (!this.weight || !this.bias) {
            throw new Error('Weights not set. Call setWeights() first.');
        }
        
        const output = new Array(this.outputDim).fill(0);
        
        for (let i = 0; i < this.outputDim; i++) {
            let sum = this.bias[i];
            for (let j = 0; j < this.inputDim; j++) {
                sum += input[j] * this.weight[i][j];
            }
            output[i] = sum;
        }
        
        return output;
    }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        FaceFormerWeightLoader, 
        SimpleFaceFormerJS, 
        LinearLayer 
    };
}

if (typeof window !== 'undefined') {
    window.FaceFormerWeightLoader = FaceFormerWeightLoader;
    window.SimpleFaceFormerJS = SimpleFaceFormerJS;
    window.LinearLayer = LinearLayer;
}
