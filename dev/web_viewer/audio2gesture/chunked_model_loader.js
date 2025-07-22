const fs = require('fs').promises;
const path = require('path');

/**
 * Enhanced ONNX model loader that handles both regular and chunked models
 */
class ChunkedModelLoader {
    constructor() {
        this.cache = new Map();
    }
    
    /**
     * Load ONNX model, automatically handling chunked files
     * @param {string} modelPath - Path to model file or chunks metadata
     * @returns {Promise<ArrayBuffer>} Model data as ArrayBuffer
     */
    async loadModel(modelPath) {
        const cacheKey = path.resolve(modelPath);
        
        if (this.cache.has(cacheKey)) {
            console.log(`📦 Loading ${path.basename(modelPath)} from cache...`);
            return this.cache.get(cacheKey);
        }
        
        let modelData;
        
        // Check if it's a chunks metadata file
        if (modelPath.endsWith('.chunks.json')) {
            modelData = await this.loadFromChunks(modelPath);
        }
        // Check if the main file exists
        else if (await this.fileExists(modelPath)) {
            modelData = await this.loadRegularFile(modelPath);
        }
        // Try to find chunks metadata file
        else {
            const baseName = path.basename(modelPath, '.onnx');
            const dirName = path.dirname(modelPath);
            const chunksMetadataPath = path.join(dirName, `${baseName}.chunks.json`);
            
            if (await this.fileExists(chunksMetadataPath)) {
                console.log(`🔍 Model file not found, but chunks detected. Loading from chunks...`);
                modelData = await this.loadFromChunks(chunksMetadataPath);
            } else {
                throw new Error(`Model file not found: ${modelPath}`);
            }
        }
        
        // Cache the loaded model
        this.cache.set(cacheKey, modelData);
        return modelData;
    }
    
    /**
     * Load model from chunks
     */
    async loadFromChunks(metadataPath) {
        const dirName = path.dirname(metadataPath);
        const metadataContent = await fs.readFile(metadataPath, 'utf8');
        const metadata = JSON.parse(metadataContent);
        
        console.log(`🔧 Loading ${metadata.originalFile} from ${metadata.totalChunks} chunks...`);
        
        const chunks = [];
        
        for (const chunkInfo of metadata.chunks) {
            const chunkPath = path.join(dirName, chunkInfo.filename);
            
            if (!(await this.fileExists(chunkPath))) {
                throw new Error(`Chunk file missing: ${chunkInfo.filename}`);
            }
            
            const chunkData = await fs.readFile(chunkPath);
            chunks.push(chunkData);
            console.log(`  ✅ Loaded ${chunkInfo.filename} (${chunkInfo.sizeMB} MB)`);
        }
        
        const rebuiltModel = Buffer.concat(chunks);
        
        // Verify integrity
        if (rebuiltModel.length !== metadata.originalSize) {
            throw new Error(`Integrity check failed: Expected ${metadata.originalSize} bytes, got ${rebuiltModel.length} bytes`);
        }
        
        console.log(`🎉 Successfully assembled ${metadata.originalFile} (${metadata.originalSizeMB} MB)`);
        return rebuiltModel.buffer;
    }
    
    /**
     * Load regular ONNX file
     */
    async loadRegularFile(modelPath) {
        console.log(`📦 Loading ${path.basename(modelPath)}...`);
        const data = await fs.readFile(modelPath);
        const sizeMB = (data.length / 1024 / 1024).toFixed(2);
        console.log(`✅ Loaded ${path.basename(modelPath)} (${sizeMB} MB)`);
        return data.buffer;
    }
    
    /**
     * Check if file exists
     */
    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }
    
    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }
    
    /**
     * Get cache info
     */
    getCacheInfo() {
        return {
            size: this.cache.size,
            keys: Array.from(this.cache.keys()).map(k => path.basename(k))
        };
    }
}

// For browser use with ONNX Runtime Web
class BrowserChunkedModelLoader {
    constructor() {
        this.cache = new Map();
    }
    
    /**
     * Load model in browser environment
     * @param {string} basePath - Base path for model files
     * @param {string} modelName - Model name (without .onnx extension)
     * @returns {Promise<ArrayBuffer>} Model data
     */
    async loadModel(basePath, modelName) {
        const cacheKey = `${basePath}/${modelName}`;
        
        if (this.cache.has(cacheKey)) {
            console.log(`📦 Loading ${modelName} from cache...`);
            return this.cache.get(cacheKey);
        }
        
        let modelData;
        
        try {
            // Try to load regular file first
            const regularPath = `${basePath}/${modelName}.onnx`;
            modelData = await this.fetchFile(regularPath);
        } catch (error) {
            // Try to load from chunks
            console.log(`🔍 Regular file not found, trying chunks...`);
            modelData = await this.loadFromChunks(basePath, modelName);
        }
        
        this.cache.set(cacheKey, modelData);
        return modelData;
    }
    
    /**
     * Load model from chunks in browser
     */
    async loadFromChunks(basePath, modelName) {
        const metadataPath = `${basePath}/${modelName}.chunks.json`;
        
        const metadataResponse = await fetch(metadataPath);
        if (!metadataResponse.ok) {
            throw new Error(`Chunks metadata not found: ${metadataPath}`);
        }
        
        const metadata = await metadataResponse.json();
        console.log(`🔧 Loading ${metadata.originalFile} from ${metadata.totalChunks} chunks...`);
        
        const chunkPromises = metadata.chunks.map(async (chunkInfo) => {
            const chunkPath = `${basePath}/${chunkInfo.filename}`;
            const response = await fetch(chunkPath);
            
            if (!response.ok) {
                throw new Error(`Failed to fetch chunk: ${chunkInfo.filename}`);
            }
            
            const chunkData = await response.arrayBuffer();
            console.log(`  ✅ Loaded ${chunkInfo.filename} (${chunkInfo.sizeMB} MB)`);
            return chunkData;
        });
        
        const chunks = await Promise.all(chunkPromises);
        
        // Combine chunks
        const totalSize = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
        const rebuiltModel = new Uint8Array(totalSize);
        
        let offset = 0;
        for (const chunk of chunks) {
            rebuiltModel.set(new Uint8Array(chunk), offset);
            offset += chunk.byteLength;
        }
        
        // Verify integrity
        if (rebuiltModel.length !== metadata.originalSize) {
            throw new Error(`Integrity check failed: Expected ${metadata.originalSize} bytes, got ${rebuiltModel.length} bytes`);
        }
        
        console.log(`🎉 Successfully assembled ${metadata.originalFile} (${metadata.originalSizeMB} MB)`);
        return rebuiltModel.buffer;
    }
    
    /**
     * Fetch file with error handling
     */
    async fetchFile(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch: ${url}`);
        }
        return await response.arrayBuffer();
    }
}

// Node.js exports
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChunkedModelLoader, BrowserChunkedModelLoader };
}

// Browser global
if (typeof window !== 'undefined') {
    window.ChunkedModelLoader = BrowserChunkedModelLoader;
}
