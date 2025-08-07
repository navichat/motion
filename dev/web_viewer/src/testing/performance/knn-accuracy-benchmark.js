/**
 * KNN Accuracy Benchmark - Comprehensive Testing Against Ground Truth
 * Measures actual accuracy, memory usage, and CPU time for KNN implementations
 */

class KNNAccuracyBenchmark {
    constructor() {
        this.testVectors = [];
        this.queryVectors = [];
        this.groundTruth = new Map();
        this.results = {};
    }

    /**
     * Generate deterministic test dataset for reproducible benchmarks
     */
    generateTestDataset(numVectors = 4096, dimensions = 512, numQueries = 20) {
        console.log(`📊 Generating ${numVectors} vectors with ${dimensions} dimensions...`);
        
        // Use seeded random for reproducibility
        let seed = 42;
        const seededRandom = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };

        // Generate test vectors with some structure (clusters)
        this.testVectors = [];
        const numClusters = 8;
        const clusterCenters = Array.from({ length: numClusters }, () => 
            Array.from({ length: dimensions }, () => seededRandom() * 4 - 2)
        );

        for (let i = 0; i < numVectors; i++) {
            const clusterId = i % numClusters;
            const center = clusterCenters[clusterId];
            const noise = 0.5; // Cluster spread
            
            const vector = center.map(val => val + (seededRandom() - 0.5) * noise);
            
            this.testVectors.push({
                id: `vector_${i}`,
                vector: vector,
                metadata: { 
                    cluster: clusterId,
                    index: i,
                    norm: Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0))
                }
            });
        }

        // Generate query vectors (some from clusters, some random)
        this.queryVectors = [];
        for (let i = 0; i < numQueries; i++) {
            let vector;
            if (i < numQueries / 2) {
                // Half from cluster centers (should have good neighbors)
                const clusterId = i % numClusters;
                const center = clusterCenters[clusterId];
                vector = center.map(val => val + (seededRandom() - 0.5) * 0.2);
            } else {
                // Half completely random (harder queries)
                vector = Array.from({ length: dimensions }, () => seededRandom() * 4 - 2);
            }
            
            this.queryVectors.push({
                id: `query_${i}`,
                vector: vector,
                metadata: { type: i < numQueries / 2 ? 'cluster_based' : 'random' }
            });
        }

        console.log(`✅ Generated ${this.testVectors.length} test vectors and ${this.queryVectors.length} queries`);
    }

    /**
     * Calculate exact distance between two vectors
     */
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
                return 1 - (dot / (norm1 * norm2));
            
            case 'manhattan':
            case 'l1':
                return vec1.reduce((sum, v, i) => sum + Math.abs(v - vec2[i]), 0);
            
            default:
                throw new Error(`Unknown distance metric: ${metric}`);
        }
    }

    /**
     * Exhaustive brute-force search for ground truth
     */
    exhaustiveSearch(queryVector, k = 8, metric = 'euclidean') {
        const startTime = performance.now();
        const startMemory = this.getMemoryUsage();

        // Calculate distances to all vectors
        const distances = this.testVectors.map(item => ({
            ...item,
            distance: this.calculateDistance(queryVector, item.vector, metric)
        }));

        // Sort by distance and take top k
        distances.sort((a, b) => a.distance - b.distance);
        const topK = distances.slice(0, k);

        const endTime = performance.now();
        const endMemory = this.getMemoryUsage();

        return {
            results: topK,
            search_time_ms: endTime - startTime,
            memory_used_mb: endMemory - startMemory,
            total_comparisons: this.testVectors.length,
            method: 'exhaustive_brute_force'
        };
    }

    /**
     * Calculate ground truth for all queries
     */
    async calculateGroundTruth(k = 8, metric = 'euclidean') {
        console.log(`🎯 Calculating ground truth with exhaustive search (k=${k}, metric=${metric})...`);
        
        this.groundTruth.clear();
        const startTime = performance.now();

        for (let i = 0; i < this.queryVectors.length; i++) {
            const query = this.queryVectors[i];
            const truth = this.exhaustiveSearch(query.vector, k, metric);
            this.groundTruth.set(query.id, truth);
            
            if ((i + 1) % 5 === 0) {
                console.log(`   Processed ${i + 1}/${this.queryVectors.length} queries...`);
            }
        }

        const totalTime = performance.now() - startTime;
        console.log(`✅ Ground truth calculated in ${totalTime.toFixed(2)}ms`);
    }

    /**
     * Calculate accuracy metrics
     */
    calculateAccuracy(predicted, groundTruth, k) {
        const predictedIds = new Set(predicted.slice(0, k).map(r => r.id));
        const trueIds = new Set(groundTruth.results.slice(0, k).map(r => r.id));
        
        // Precision@K: How many of the predicted top-k are actually in true top-k
        const intersection = new Set([...predictedIds].filter(id => trueIds.has(id)));
        const precisionAtK = intersection.size / k;

        // Recall@K: How many of the true top-k were found
        const recallAtK = intersection.size / Math.min(k, trueIds.size);

        // NDCG@K: Normalized Discounted Cumulative Gain
        let dcg = 0;
        let idcg = 0;
        
        for (let i = 0; i < k; i++) {
            const discount = Math.log2(i + 2); // i+2 because log2(1) = 0
            
            // DCG: score based on actual results
            if (i < predicted.length) {
                const isRelevant = trueIds.has(predicted[i].id) ? 1 : 0;
                dcg += isRelevant / discount;
            }
            
            // IDCG: ideal score (perfect ranking)
            if (i < groundTruth.results.length) {
                idcg += 1 / discount;
            }
        }
        
        const ndcg = idcg > 0 ? dcg / idcg : 0;

        // Average distance error
        const trueDistances = groundTruth.results.slice(0, k).map(r => r.distance);
        const predDistances = predicted.slice(0, k).map(r => r.distance || 0);
        const avgTrueDistance = trueDistances.reduce((sum, d) => sum + d, 0) / trueDistances.length;
        const avgPredDistance = predDistances.reduce((sum, d) => sum + d, 0) / predDistances.length;
        const distanceError = Math.abs(avgPredDistance - avgTrueDistance) / avgTrueDistance;

        return {
            precision_at_k: precisionAtK,
            recall_at_k: recallAtK,
            ndcg_at_k: ndcg,
            exact_matches: intersection.size,
            distance_error: distanceError,
            avg_true_distance: avgTrueDistance,
            avg_pred_distance: avgPredDistance
        };
    }

    /**
     * Get memory usage (approximate)
     */
    getMemoryUsage() {
        if (typeof process !== 'undefined' && process.memoryUsage) {
            return process.memoryUsage().heapUsed / 1024 / 1024; // MB
        }
        // Browser fallback - very approximate
        return (performance.memory?.usedJSHeapSize || 0) / 1024 / 1024;
    }

    /**
     * Benchmark a KNN implementation
     */
    async benchmarkImplementation(implementation, name, config = {}) {
        console.log(`\n🧪 Benchmarking ${name}...`);
        
        const startSetupTime = performance.now();
        const startSetupMemory = this.getMemoryUsage();

        // Initialize implementation
        const initResult = await implementation.initialize(
            config.dimensions || 512,
            config.maxElements || 4096,
            config.metric || 'euclidean',
            config.params || {}
        );

        if (!initResult.success) {
            throw new Error(`Failed to initialize ${name}: ${initResult.error}`);
        }

        // Add all test vectors
        const addResult = await implementation.addVectors(this.testVectors);
        
        const setupTime = performance.now() - startSetupTime;
        const setupMemory = this.getMemoryUsage() - startSetupMemory;

        console.log(`   Setup: ${setupTime.toFixed(2)}ms, Memory: ${setupMemory.toFixed(2)}MB`);

        // Benchmark queries
        const queryResults = [];
        const accuracyMetrics = [];
        let totalSearchTime = 0;
        let totalSearchMemory = 0;

        for (const query of this.queryVectors) {
            const startQueryTime = performance.now();
            const startQueryMemory = this.getMemoryUsage();

            // Perform search
            const searchResult = await implementation.knnSearch(
                query.vector, 
                8, 
                config.searchParams || {}
            );

            const queryTime = performance.now() - startQueryTime;
            const queryMemory = this.getMemoryUsage() - startQueryMemory;

            totalSearchTime += queryTime;
            totalSearchMemory += queryMemory;

            // Calculate accuracy against ground truth
            const groundTruth = this.groundTruth.get(query.id);
            if (groundTruth) {
                const accuracy = this.calculateAccuracy(searchResult.results, groundTruth, 8);
                accuracyMetrics.push(accuracy);
            }

            queryResults.push({
                query_id: query.id,
                search_time_ms: queryTime,
                memory_used_mb: queryMemory,
                results_found: searchResult.k_returned,
                ...searchResult
            });
        }

        // Calculate aggregate metrics
        const avgAccuracy = accuracyMetrics.reduce((sum, acc) => ({
            precision_at_k: sum.precision_at_k + acc.precision_at_k,
            recall_at_k: sum.recall_at_k + acc.recall_at_k,
            ndcg_at_k: sum.ndcg_at_k + acc.ndcg_at_k,
            exact_matches: sum.exact_matches + acc.exact_matches,
            distance_error: sum.distance_error + acc.distance_error
        }), { precision_at_k: 0, recall_at_k: 0, ndcg_at_k: 0, exact_matches: 0, distance_error: 0 });

        const numQueries = accuracyMetrics.length;
        Object.keys(avgAccuracy).forEach(key => {
            avgAccuracy[key] /= numQueries;
        });

        const result = {
            implementation: name,
            config: config,
            setup_time_ms: setupTime,
            setup_memory_mb: setupMemory,
            vectors_added: addResult.added,
            total_queries: numQueries,
            avg_search_time_ms: totalSearchTime / numQueries,
            total_search_time_ms: totalSearchTime,
            avg_search_memory_mb: totalSearchMemory / numQueries,
            total_search_memory_mb: totalSearchMemory,
            accuracy_metrics: avgAccuracy,
            detailed_queries: queryResults
        };

        console.log(`   Results: ${avgAccuracy.precision_at_k.toFixed(3)} precision, ${avgAccuracy.ndcg_at_k.toFixed(3)} NDCG, ${(totalSearchTime / numQueries).toFixed(2)}ms avg search`);

        return result;
    }

    /**
     * Run comprehensive benchmark comparing all implementations
     */
    async runComprehensiveBenchmark() {
        console.log('🚀 Starting Comprehensive KNN Accuracy Benchmark\n');
        console.log('Dataset: 4096 vectors × 512 dimensions, finding top 8 neighbors\n');

        // Generate test dataset
        this.generateTestDataset(4096, 512, 20);

        // Calculate ground truth
        await this.calculateGroundTruth(8, 'euclidean');

        const results = [];

        try {
            // Load implementations
            const CloseVectorInterface = require('./close-vector/closevector-interface.cjs');
            const HNSWInterface = require('./hsnwlib/hnsw-interface.cjs');
            const UnifiedKNNInterface = require('./close-vector/unified-knn-interface.js');

            // Benchmark CloseVector
            const closeVector = new CloseVectorInterface();
            const cvResult = await this.benchmarkImplementation(closeVector, 'CloseVector', {
                dimensions: 512,
                maxElements: 4096,
                metric: 'euclidean'
            });
            results.push(cvResult);

            // Benchmark HNSW with different configurations
            const hnsw1 = new HNSWInterface();
            const hnswResult1 = await this.benchmarkImplementation(hnsw1, 'HNSW (M=16, ef=200)', {
                dimensions: 512,
                maxElements: 4096,
                metric: 'l2',
                params: { M: 16, efConstruction: 200 },
                searchParams: { ef: 200 }
            });
            results.push(hnswResult1);

            const hnsw2 = new HNSWInterface();
            const hnswResult2 = await this.benchmarkImplementation(hnsw2, 'HNSW (M=32, ef=400)', {
                dimensions: 512,
                maxElements: 4096,
                metric: 'l2',
                params: { M: 32, efConstruction: 400 },
                searchParams: { ef: 400 }
            });
            results.push(hnswResult2);

            // Benchmark Unified KNN
            const unified = new UnifiedKNNInterface();
            const unifiedResult = await this.benchmarkImplementation(unified, 'Unified KNN', {
                implementation: 'auto',
                dimensions: 512,
                maxElements: 4096,
                distanceMetric: 'euclidean'
            });
            results.push(unifiedResult);

        } catch (error) {
            console.error('❌ Benchmark failed:', error.message);
            throw error;
        }

        // Generate comparison report
        this.generateComparisonReport(results);

        return results;
    }

    /**
     * Generate detailed comparison report
     */
    generateComparisonReport(results) {
        console.log('\n📊 COMPREHENSIVE KNN BENCHMARK RESULTS');
        console.log('=' .repeat(80));

        // Accuracy comparison
        console.log('\n🎯 ACCURACY METRICS (Higher is better):');
        console.log('Implementation'.padEnd(25) + 'Precision@8'.padEnd(12) + 'NDCG@8'.padEnd(12) + 'Exact Matches'.padEnd(15) + 'Dist Error');
        console.log('-'.repeat(80));
        
        results.forEach(result => {
            const acc = result.accuracy_metrics;
            console.log(
                result.implementation.padEnd(25) +
                acc.precision_at_k.toFixed(3).padEnd(12) +
                acc.ndcg_at_k.toFixed(3).padEnd(12) +
                acc.exact_matches.toFixed(1).padEnd(15) +
                (acc.distance_error * 100).toFixed(1) + '%'
            );
        });

        // Performance comparison
        console.log('\n⚡ PERFORMANCE METRICS:');
        console.log('Implementation'.padEnd(25) + 'Setup Time'.padEnd(12) + 'Avg Search'.padEnd(12) + 'Setup Mem'.padEnd(12) + 'Search Mem');
        console.log('-'.repeat(80));
        
        results.forEach(result => {
            console.log(
                result.implementation.padEnd(25) +
                (result.setup_time_ms.toFixed(1) + 'ms').padEnd(12) +
                (result.avg_search_time_ms.toFixed(2) + 'ms').padEnd(12) +
                (result.setup_memory_mb.toFixed(1) + 'MB').padEnd(12) +
                (result.avg_search_memory_mb.toFixed(2) + 'MB')
            );
        });

        // Find best performers
        const bestAccuracy = results.reduce((best, curr) => 
            curr.accuracy_metrics.precision_at_k > best.accuracy_metrics.precision_at_k ? curr : best
        );
        
        const fastestSearch = results.reduce((fastest, curr) => 
            curr.avg_search_time_ms < fastest.avg_search_time_ms ? curr : fastest
        );

        const lowestMemory = results.reduce((lowest, curr) => 
            curr.total_search_memory_mb < lowest.total_search_memory_mb ? curr : lowest
        );

        console.log('\n🏆 WINNERS:');
        console.log(`   Best Accuracy: ${bestAccuracy.implementation} (${(bestAccuracy.accuracy_metrics.precision_at_k * 100).toFixed(1)}% precision)`);
        console.log(`   Fastest Search: ${fastestSearch.implementation} (${fastestSearch.avg_search_time_ms.toFixed(2)}ms avg)`);
        console.log(`   Lowest Memory: ${lowestMemory.implementation} (${lowestMemory.total_search_memory_mb.toFixed(2)}MB total)`);

        console.log('\n📝 INTERPRETATION:');
        console.log('• Precision@8: Fraction of returned neighbors that are actually in the true top-8');
        console.log('• NDCG@8: Normalized Discounted Cumulative Gain - considers ranking quality');
        console.log('• Exact Matches: Average number of exactly correct neighbors found');
        console.log('• Distance Error: Relative error in average distance vs ground truth');
        console.log('• All measurements are averages across 20 different query vectors');
    }
}

// Export for use in other modules
module.exports = { KNNAccuracyBenchmark };

// Run benchmark if called directly
if (require.main === module) {
    const benchmark = new KNNAccuracyBenchmark();
    benchmark.runComprehensiveBenchmark()
        .then(results => {
            console.log(`\n✅ Benchmark completed with ${results.length} implementations tested`);
            process.exit(0);
        })
        .catch(error => {
            console.error('\n❌ Benchmark failed:', error.message);
            console.error(error.stack);
            process.exit(1);
        });
}
