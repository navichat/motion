/**
 * Simple KNN Interface for Testing
 */

class SimpleKNNInterface {
    constructor() {
        this.vectorStore = [];
        this.initialized = false;
        this.dimensions = null;
    }

    async initialize(dimensions, maxElements, metric) {
        this.dimensions = dimensions;
        this.initialized = true;
        return { success: true, dimensions, maxElements, metric };
    }

    async addVectors(vectors) {
        this.vectorStore = [...vectors];
        return { added: vectors.length };
    }

    async knnSearch(queryVector, k) {
        const startTime = performance.now();
        
        // Simple brute force search for testing
        const distances = this.vectorStore.map(item => {
            const distance = this.euclideanDistance(queryVector, item.vector);
            return { ...item, distance };
        });

        distances.sort((a, b) => a.distance - b.distance);
        const results = distances.slice(0, k);

        return {
            results,
            k_requested: k,
            k_returned: results.length,
            search_time_ms: performance.now() - startTime
        };
    }

    euclideanDistance(a, b) {
        if (a.length !== b.length) return Infinity;
        return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
    }
}

module.exports = SimpleKNNInterface;
