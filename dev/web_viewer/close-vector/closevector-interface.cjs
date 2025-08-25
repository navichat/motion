/**
 * CloseVector Interface Module
 */

class CloseVectorInterface {
  constructor() {
    this.vectorStore = [];
    this.initialized = false;
    this.dimensions = null;
    this.distanceMetric = 'euclidean';
    this.maxElements = 10000;
  }

  async initialize(dimensions = 512, maxElements = 4096, metric = 'euclidean', params = {}) {
    try {
      this.dimensions = dimensions;
      this.maxElements = maxElements;
      this.distanceMetric = metric;
      this.vectorStore = [];
      this.initialized = true;
      
      console.log(`   CloseVector initialized: ${dimensions}D, max=${maxElements}, metric=${metric}`);
      
      return { 
        success: true, 
        dimensions, 
        maxElements,
        metric,
        implementation: 'CloseVector-Mock'
      };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async addVectors(vectors) {
    if (!this.initialized) {
      throw new Error('CloseVector not initialized. Call initialize() first.');
    }

    let added = 0;
    const startTime = performance.now();
    
    for (const vectorObj of vectors) {
      if (this.vectorStore.length >= this.maxElements) {
        console.warn(`Reached maximum elements (${this.maxElements}), skipping remaining vectors`);
        break;
      }
      
      if (!vectorObj.vector || vectorObj.vector.length !== this.dimensions) {
        console.warn(`Skipping vector ${vectorObj.id}: invalid dimensions`);
        continue;
      }
      
      this.vectorStore.push({
        id: vectorObj.id,
        vector: [...vectorObj.vector],
        metadata: vectorObj.metadata || {}
      });
      
      added++;
    }
    
    const addTime = performance.now() - startTime;
    console.log(`   Added ${added} vectors in ${addTime.toFixed(2)}ms`);
    
    return { 
      added, 
      total: this.vectorStore.length,
      add_time_ms: addTime,
      success: true 
    };
  }

  async knnSearch(queryVector, k = 8, options = {}) {
    if (!this.initialized) {
      throw new Error('CloseVector not initialized');
    }

    if (queryVector.length !== this.dimensions) {
      throw new Error(`Query vector dimensions (${queryVector.length}) don't match initialized dimensions (${this.dimensions})`);
    }

    const startTime = performance.now();
    
    const distances = this.vectorStore.map(item => {
      const distance = this.calculateDistance(queryVector, item.vector, this.distanceMetric);
      return {
        id: item.id,
        distance: distance,
        similarity: this.distanceMetric === 'cosine' ? 1 - distance : 1 / (1 + distance),
        metadata: item.metadata,
        vector: item.vector
      };
    });

    distances.sort((a, b) => a.distance - b.distance);
    const results = distances.slice(0, Math.min(k, distances.length));
    const searchTime = performance.now() - startTime;
    
    return {
      results: results,
      k_requested: k,
      k_returned: results.length,
      search_time_ms: searchTime,
      query_dimensions: queryVector.length,
      total_vectors_searched: this.vectorStore.length,
      distance_metric: this.distanceMetric,
      success: true
    };
  }

  calculateDistance(vec1, vec2, metric = 'euclidean') {
    if (vec1.length !== vec2.length) {
      throw new Error('Vector dimensions must match');
    }

    switch (metric) {
      case 'euclidean':
      case 'l2':
        return Math.sqrt(vec1.reduce((sum, v, i) => sum + Math.pow(v - vec2[i], 2), 0));
      
      case 'cosine':
        const dot = vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
        const norm1 = Math.sqrt(vec1.reduce((sum, v) => sum + v * v, 0));
        const norm2 = Math.sqrt(vec2.reduce((sum, v) => sum + v * v, 0));
        if (norm1 === 0 || norm2 === 0) return 1;
        return 1 - (dot / (norm1 * norm2));
      
      case 'manhattan':
      case 'l1':
        return vec1.reduce((sum, v, i) => sum + Math.abs(v - vec2[i]), 0);
      
      default:
        throw new Error(`Unknown distance metric: ${metric}`);
    }
  }

  getStats() {
    return {
      total_vectors: this.vectorStore.length,
      dimensions: this.dimensions,
      distance_metric: this.distanceMetric,
      max_elements: this.maxElements,
      initialized: this.initialized,
      memory_usage_estimate_mb: (this.vectorStore.length * this.dimensions * 8) / (1024 * 1024)
    };
  }

  clear() {
    this.vectorStore = [];
    return { success: true, message: 'Vector store cleared' };
  }

  getVector(id) {
    const found = this.vectorStore.find(item => item.id === id);
    return found ? { success: true, vector: found } : { success: false, error: 'Vector not found' };
  }

  removeVector(id) {
    const index = this.vectorStore.findIndex(item => item.id === id);
    if (index !== -1) {
      this.vectorStore.splice(index, 1);
      return { success: true, message: `Vector ${id} removed` };
    }
    return { success: false, error: 'Vector not found' };
  }

  getAllIds() {
    return this.vectorStore.map(item => item.id);
  }
}

module.exports = CloseVectorInterface;
