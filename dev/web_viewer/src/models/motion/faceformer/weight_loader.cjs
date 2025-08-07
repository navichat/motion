// FaceFormer Weight Loader for JavaScript/Node.js (CommonJS version)
const fs = require('fs');

class FaceFormerWeightLoader {
    constructor() {
        this.weights = null;
        this.modelInfo = null;
        this.isLoaded = false;
    }

    async loadPythonWeights(weightsPath = './faceformer_python_weights.json') {
        console.log('📥 Loading Python weights from:', weightsPath);
        
        try {
            const data = fs.readFileSync(weightsPath, 'utf8');
            const weightsData = JSON.parse(data);
            
            this.weights = weightsData.weights;
            this.modelInfo = weightsData.model_info;
            this.layerInfo = weightsData.layer_info;
            this.isLoaded = true;
            
            console.log('✅ Python weights loaded successfully');
            console.log(`📊 Model info: ${this.modelInfo.architecture}`);
            console.log(`🔧 Layers loaded: ${Object.keys(this.weights).length / 2}`);
            
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
        
        const weight = this.weights[`${layerName}.weight`];
        const bias = this.weights[`${layerName}.bias`];
        
        if (!weight) {
            throw new Error(`Layer ${layerName} not found in weights`);
        }
        
        return { weight, bias };
    }

    getAllLayerNames() {
        if (!this.isLoaded) return [];
        
        const layerNames = new Set();
        for (const key of Object.keys(this.weights)) {
            if (key.endsWith('.weight')) {
                layerNames.add(key.replace('.weight', ''));
            }
        }
        return Array.from(layerNames);
    }

    applyToCustomModel(model) {
        if (!this.isLoaded) {
            throw new Error('Weights not loaded. Call loadPythonWeights() first.');
        }
        
        console.log('🔄 Applying Python weights to custom model...');
        
        let appliedCount = 0;
        
        try {
            // Apply audio_proj weights to audioProj
            if (this.weights.audio_proj_weight && this.weights.audio_proj_bias) {
                model.audioProj.weight = this.weights.audio_proj_weight;
                model.audioProj.bias = this.weights.audio_proj_bias;
                appliedCount++;
                console.log('  ✅ Applied audio_proj weights');
            }
            
            // Apply emb_proj weights to embProj
            if (this.weights.emb_proj_weight && this.weights.emb_proj_bias) {
                model.embProj.weight = this.weights.emb_proj_weight;
                model.embProj.bias = this.weights.emb_proj_bias;
                appliedCount++;
                console.log('  ✅ Applied emb_proj weights');
            }
            
            // Apply subject_embedding weights to subjectEmbedding
            if (this.weights.subject_embedding_weight && this.weights.subject_embedding_bias) {
                model.subjectEmbedding.weight = this.weights.subject_embedding_weight;
                model.subjectEmbedding.bias = this.weights.subject_embedding_bias;
                appliedCount++;
                console.log('  ✅ Applied subject_embedding weights');
            }
            
            // Apply fusion weights to fusion
            if (this.weights.fusion_weight && this.weights.fusion_bias) {
                model.fusion.weight = this.weights.fusion_weight;
                model.fusion.bias = this.weights.fusion_bias;
                appliedCount++;
                console.log('  ✅ Applied fusion weights');
            }
            
            // Apply vertex_proj weights to vertexProj
            if (this.weights.vertex_proj_weight && this.weights.vertex_proj_bias) {
                model.vertexProj.weight = this.weights.vertex_proj_weight;
                model.vertexProj.bias = this.weights.vertex_proj_bias;
                appliedCount++;
                console.log('  ✅ Applied vertex_proj weights');
            }
            
            // Apply emb_update weights to embUpdate
            if (this.weights.emb_update_weight && this.weights.emb_update_bias) {
                model.embUpdate.weight = this.weights.emb_update_weight;
                model.embUpdate.bias = this.weights.emb_update_bias;
                appliedCount++;
                console.log('  ✅ Applied emb_update weights');
            }
            
        } catch (error) {
            console.warn(`⚠️ Failed to apply weights:`, error.message);
        }
        
        console.log(`✅ Applied ${appliedCount}/6 layer weights to custom model`);
        return appliedCount;
    }
}

// Simple linear layer for custom implementation
class LinearLayer {
    constructor(inFeatures, outFeatures, useBias = true) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = useBias;
        
        // Initialize with random weights (will be overwritten by loader)
        this.weight = new Array(outFeatures).fill(0).map(() => 
            new Array(inFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.1)
        );
        this.bias = useBias ? new Array(outFeatures).fill(0) : null;
    }
    
    forward(input) {
        const output = new Array(this.outFeatures).fill(0);
        
        for (let i = 0; i < this.outFeatures; i++) {
            let sum = 0;
            for (let j = 0; j < this.inFeatures; j++) {
                sum += input[j] * this.weight[i][j];
            }
            if (this.useBias && this.bias) {
                sum += this.bias[i];
            }
            output[i] = sum;
        }
        
        return output;
    }
}

// Simplified FaceFormer implementation for exact computation
class SimpleFaceFormerJS {
    constructor() {
        // Initialize layers with the same structure as Python
        this.audioProj = new LinearLayer(768, 64, true);      // audio_proj
        this.embProj = new LinearLayer(64, 64, true);         // emb_proj  
        this.subjectEmbedding = new LinearLayer(3, 64, true); // subject_embedding
        this.fusion = new LinearLayer(192, 64, true);         // fusion (64*3 -> 64)
        this.vertexProj = new LinearLayer(64, 15069, true);   // vertex_proj
        this.embUpdate = new LinearLayer(64, 64, true);       // emb_update
    }
    
    forward(inputs) {
        const { audio_features, vertice_emb, one_hot, template } = inputs;
        
        // 1. Process audio features
        const audioProj = this.audioProj.forward(audio_features);
        
        // 2. Process embedding
        const embProj = this.embProj.forward(vertice_emb);
        
        // 3. Process one-hot subject encoding  
        const subjectEmb = this.subjectEmbedding.forward(one_hot);
        
        // 4. Concatenate all features (like torch.cat in Python)
        const fused_input = [...audioProj, ...embProj, ...subjectEmb];
        
        // 5. Apply fusion layer
        const fused = this.fusion.forward(fused_input);
        
        // 6. Generate vertex delta
        const vertexDelta = this.vertexProj.forward(fused);
        
        // 7. Apply to template
        const newVertices = new Array(15069);
        for (let i = 0; i < 15069; i++) {
            newVertices[i] = template[i] + vertexDelta[i];
        }
        
        // 8. Update embedding
        const updatedEmb = this.embUpdate.forward(fused);
        
        return {
            vertices: newVertices,
            embedding: updatedEmb
        };
    }
}

module.exports = {
    FaceFormerWeightLoader,
    SimpleFaceFormerJS,
    LinearLayer
};
