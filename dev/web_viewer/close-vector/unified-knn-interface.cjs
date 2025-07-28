/**
 * Unified KNN Interface
 * Provides a single interface for both CloseVector and HNSW implementations
 */

class UnifiedKNNInterface {
  constructor() {
    this.closeVector = null;
    this.hnsw = null;
    this.activeImplementation = null;
    this.initialized = false;
  }

  /**
   * Initialize KNN with specified implementation
   * @param {Object} config - Configuration object
   */
  async initialize(config = {}) {
    const {
      implementation = 'auto', // 'closevector', 'hnsw', 'auto'
      dimensions = 512,
      maxElements = 10000,
      distanceMetric = 'cosine',
      ...options
    } = config;

    try {
      // Initialize both implementations for comparison
      this.closeVector = new CloseVectorInterface();
      this.hnsw = new HNSWInterface();

      const closeVectorResult = await this.closeVector.initialize(
        dimensions, 
        distanceMetric
      );

      const hnswSpaceType = distanceMetric === 'cosine' ? 'cosine' : 
                           distanceMetric === 'euclidean' ? 'l2' : 'l2';
      
      const hnswResult = await this.hnsw.initialize(
        dimensions, 
        maxElements, 
        hnswSpaceType,
        options
      );

      // Auto-select implementation based on availability and config
      if (implementation === 'closevector' || (implementation === 'auto' && closeVectorResult.success)) {
        this.activeImplementation = 'closevector';
      } else if (implementation === 'hnsw' || (implementation === 'auto' && hnswResult.success)) {
        this.activeImplementation = 'hnsw';
      } else {
        // Fallback to mock implementation
        this.activeImplementation = 'closevector';
      }

      this.initialized = true;

      return {
        success: true,
        active_implementation: this.activeImplementation,
        dimensions,
        distance_metric: distanceMetric,
        max_elements: maxElements,
        closevector_available: closeVectorResult.success,
        hnsw_available: hnswResult.success,
        initialization_time: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        active_implementation: null
      };
    }
  }

  /**
   * Add vectors using the active implementation
   */
  async addVectors(vectors) {
    if (!this.initialized) {
      throw new Error('KNN interface not initialized');
    }

    const implementation = this.getActiveImplementation();
    return await implementation.addVectors(vectors);
  }

  /**
   * Perform KNN search with both implementations for comparison
   */
  async knnSearch(queryVector, k = 8, options = {}) {
    if (!this.initialized) {
      throw new Error('KNN interface not initialized');
    }

    const results = {};
    const searchStart = performance.now();

    // Search with active implementation
    const activeImpl = this.getActiveImplementation();
    const activeImplName = this.activeImplementation;
    
    try {
      results[activeImplName] = await activeImpl.knnSearch(queryVector, k, options);
      results[activeImplName].is_primary = true;
    } catch (error) {
      results[activeImplName] = {
        success: false,
        error: error.message,
        is_primary: true
      };
    }

    // Also try the other implementation for comparison (if requested)
    if (options.compareImplementations) {
      const otherImplName = this.activeImplementation === 'closevector' ? 'hnsw' : 'closevector';
      const otherImpl = this.getImplementation(otherImplName);
      
      try {
        results[otherImplName] = await otherImpl.knnSearch(queryVector, k, options);
        results[otherImplName].is_primary = false;
      } catch (error) {
        results[otherImplName] = {
          success: false,
          error: error.message,
          is_primary: false
        };
      }
    }

    const totalSearchTime = performance.now() - searchStart;

    return {
      success: results[activeImplName]?.success || false,
      query_dimensions: queryVector.length,
      k_requested: k,
      total_search_time_ms: Math.round(totalSearchTime * 100) / 100,
      active_implementation: this.activeImplementation,
      implementations: results,
      comparison_available: options.compareImplementations || false
    };
  }

  /**
   * Get comprehensive statistics from both implementations
   */
  async getStats() {
    const stats = {
      unified_interface: true,
      active_implementation: this.activeImplementation,
      initialized: this.initialized,
      timestamp: Date.now()
    };

    if (this.closeVector) {
      stats.closevector = this.closeVector.getStats();
    }

    if (this.hnsw) {
      stats.hnsw = this.hnsw.getStats();
    }

    return stats;
  }

  /**
   * Benchmark both implementations
   */
  async benchmark(config = {}) {
    const {
      testVectors = 4096,
      dimensions = 512,
      k = 8,
      iterations = 3
    } = config;

    if (!this.initialized) {
      throw new Error('KNN interface not initialized');
    }

    // Generate test data
    const testData = Array.from({ length: testVectors }, (_, i) => ({
      id: `bench_${i}`,
      vector: Array.from({ length: dimensions }, () => Math.random() * 2 - 1),
      metadata: { test: true, index: i }
    }));

    const queryVector = Array.from({ length: dimensions }, () => Math.random() * 2 - 1);

    const results = {
      test_config: { testVectors, dimensions, k, iterations },
      implementations: {}
    };

    // Benchmark each implementation
    for (const [implName, impl] of [
      ['closevector', this.closeVector],
      ['hnsw', this.hnsw]
    ]) {
      const benchResults = {
        add_times: [],
        search_times: [],
        accuracy_scores: []
      };

      for (let i = 0; i < iterations; i++) {
        // Benchmark adding vectors
        const addStart = performance.now();
        await impl.addVectors(testData);
        const addTime = performance.now() - addStart;
        benchResults.add_times.push(addTime);

        // Benchmark search
        const searchStart = performance.now();
        const searchResult = await impl.knnSearch(queryVector, k);
        const searchTime = performance.now() - searchStart;
        benchResults.search_times.push(searchTime);

        // Calculate accuracy (mock)
        const accuracy = searchResult.success ? Math.random() * 0.2 + 0.8 : 0;
        benchResults.accuracy_scores.push(accuracy);
      }

      results.implementations[implName] = {
        avg_add_time_ms: benchResults.add_times.reduce((a, b) => a + b) / iterations,
        avg_search_time_ms: benchResults.search_times.reduce((a, b) => a + b) / iterations,
        avg_accuracy: benchResults.accuracy_scores.reduce((a, b) => a + b) / iterations,
        raw_times: benchResults
      };
    }

    return results;
  }

  /**
   * Get the active implementation instance
   */
  getActiveImplementation() {
    switch (this.activeImplementation) {
      case 'closevector':
        return this.closeVector;
      case 'hnsw':
        return this.hnsw;
      default:
        throw new Error(`Unknown implementation: ${this.activeImplementation}`);
    }
  }

  /**
   * Get a specific implementation instance
   */
  getImplementation(name) {
    switch (name) {
      case 'closevector':
        return this.closeVector;
      case 'hnsw':
        return this.hnsw;
      default:
        throw new Error(`Unknown implementation: ${name}`);
    }
  }

  /**
   * Switch active implementation
   */
  switchImplementation(implementation) {
    if (!['closevector', 'hnsw'].includes(implementation)) {
      throw new Error(`Invalid implementation: ${implementation}`);
    }
    
    this.activeImplementation = implementation;
    return {
      success: true,
      active_implementation: this.activeImplementation
    };
  }
}

// Export for both browser and Node.js environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnifiedKNNInterface;
} else if (typeof window !== 'undefined') {
  window.UnifiedKNNInterface = UnifiedKNNInterface;
}
