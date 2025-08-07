/**
 * Working KNN Accuracy Benchmark 
 * Measures real accuracy against exhaustive brute-force ground truth
 */

class WorkingKNNBenchmark {
    constructor() {
        this.testVectors = [];
        this.queryVectors = [];
        this.groundTruth = new Map();
    }

    /**
     * Generate deterministic test dataset
     */
    generateTestDataset(numVectors = 4096, dimensions = 512, numQueries = 10) {
        console.log(`📊 Generating ${numVectors} vectors with ${dimensions} dimensions...`);
        
        // Use seeded random for reproducibility
        let seed = 42;
        const seededRandom = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };

        // Generate test vectors with cluster structure for realistic data
        this.testVectors = [];
        const numClusters = 8;
        const clusterCenters = Array.from({ length: numClusters }, () => 
            Array.from({ length: dimensions }, () => seededRandom() * 4 - 2)
        );

        for (let i = 0; i < numVectors; i++) {
            const clusterId = i % numClusters;
            const center = clusterCenters[clusterId];
            const noise = 0.3; // Cluster spread
            
            const vector = center.map(val => val + (seededRandom() - 0.5) * noise);
            
            this.testVectors.push({
                id: `vector_${i}`,
                vector: vector,
                metadata: { 
                    cluster: clusterId,
                    index: i
                }
            });
        }

        // Generate query vectors 
        this.queryVectors = [];
        for (let i = 0; i < numQueries; i++) {
            let vector;
            if (i < numQueries / 2) {
                // Half from cluster centers (should have good neighbors)
                const clusterId = i % numClusters;
                const center = clusterCenters[clusterId];
                vector = center.map(val => val + (seededRandom() - 0.5) * 0.1);
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
        switch (metric) {
            case 'euclidean':
            case 'l2':
                return Math.sqrt(vec1.reduce((sum, v, i) => sum + Math.pow(v - vec2[i], 2), 0));
            case 'cosine':
                const dot = vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
                const norm1 = Math.sqrt(vec1.reduce((sum, v) => sum + v * v, 0));
                const norm2 = Math.sqrt(vec2.reduce((sum, v) => sum + v * v, 0));
                return 1 - (dot / (norm1 * norm2));
            default:
                return vec1.reduce((sum, v, i) => sum + Math.abs(v - vec2[i]), 0);
        }
    }

    /**
     * Exhaustive brute-force search for ground truth
     */
    exhaustiveSearch(queryVector, k = 8, metric = 'euclidean') {
        const startTime = performance.now();
        const startMemory = process.memoryUsage().heapUsed / 1024 / 1024;

        // Calculate distances to all vectors
        const distances = this.testVectors.map(item => ({
            ...item,
            distance: this.calculateDistance(queryVector, item.vector, metric)
        }));

        // Sort by distance and take top k
        distances.sort((a, b) => a.distance - b.distance);
        const topK = distances.slice(0, k);

        const endTime = performance.now();
        const endMemory = process.memoryUsage().heapUsed / 1024 / 1024;

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
        }

        const totalTime = performance.now() - startTime;
        console.log(`✅ Ground truth calculated in ${totalTime.toFixed(2)}ms`);
    }

    /**
     * Calculate accuracy metrics comparing predicted vs ground truth
     */
    calculateAccuracy(predicted, groundTruth, k) {
        const predictedIds = new Set(predicted.slice(0, k).map(r => r.id));
        const trueIds = new Set(groundTruth.results.slice(0, k).map(r => r.id));
        
        // Precision@K: How many of the predicted top-k are actually in true top-k
        const intersection = new Set([...predictedIds].filter(id => trueIds.has(id)));
        const precisionAtK = intersection.size / k;

        // Recall@K: How many of the true top-k were found
        const recallAtK = intersection.size / Math.min(k, trueIds.size);

        // NDCG@K: Normalized Discounted Cumulative Gain (measures ranking quality)
        let dcg = 0;
        let idcg = 0;
        
        for (let i = 0; i < k; i++) {
            const discount = Math.log2(i + 2);
            
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

        // Distance quality metrics
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
     * Create a mock KNN implementation that varies in quality for demonstration
     */
    createMockKNNImplementation(name, accuracyLevel = 0.8) {
        return {
            name: name,
            accuracyLevel: accuracyLevel,
            
            async initialize(dimensions, maxElements, metric) {
                return { success: true, dimensions, maxElements, metric };
            },

            async addVectors(vectors) {
                this.vectorStore = [...vectors];
                return { added: vectors.length };
            },

            async knnSearch(queryVector, k) {
                const startTime = performance.now();
                const startMemory = process.memoryUsage().heapUsed / 1024 / 1024;
                
                // Simulate different quality levels
                let results;
                
                if (this.accuracyLevel >= 0.9) {
                    // High quality: nearly perfect results with small noise
                    const distances = this.vectorStore.map(item => {
                        const distance = this.calculateDistance(queryVector, item.vector);
                        // Add tiny amount of noise to perfect results
                        return { ...item, distance: distance + Math.random() * 0.001 };
                    });
                    distances.sort((a, b) => a.distance - b.distance);
                    results = distances.slice(0, k);
                    
                } else if (this.accuracyLevel >= 0.7) {
                    // Medium quality: mostly correct with some errors
                    const distances = this.vectorStore.map(item => {
                        const distance = this.calculateDistance(queryVector, item.vector);
                        return { ...item, distance };
                    });
                    distances.sort((a, b) => a.distance - b.distance);
                    
                    // Take more candidates and introduce some randomness
                    const candidates = distances.slice(0, k * 2);
                    results = [];
                    for (let i = 0; i < k; i++) {
                        if (Math.random() < this.accuracyLevel) {
                            // Pick from top candidates
                            const idx = Math.floor(Math.random() * Math.min(candidates.length, k));
                            results.push(candidates.splice(idx, 1)[0]);
                        } else {
                            // Pick from broader set
                            const idx = Math.floor(Math.random() * candidates.length);
                            results.push(candidates.splice(idx, 1)[0]);
                        }
                    }
                    
                } else {
                    // Low quality: significant errors
                    const distances = this.vectorStore.map(item => {
                        let distance = this.calculateDistance(queryVector, item.vector);
                        // Add significant noise
                        distance += Math.random() * distance * 0.5;
                        return { ...item, distance };
                    });
                    distances.sort((a, b) => a.distance - b.distance);
                    results = distances.slice(0, k);
                }

                const endTime = performance.now();
                const endMemory = process.memoryUsage().heapUsed / 1024 / 1024;

                return {
                    results,
                    k_requested: k,
                    k_returned: results.length,
                    search_time_ms: endTime - startTime,
                    memory_used_mb: endMemory - startMemory
                };
            },

            calculateDistance(vec1, vec2) {
                return Math.sqrt(vec1.reduce((sum, v, i) => sum + Math.pow(v - vec2[i], 2), 0));
            }
        };
    }

    /**
     * Run comprehensive benchmark
     */
    async runBenchmark() {
        console.log('🚀 KNN Accuracy Benchmark: Measuring Real Performance vs Ground Truth\n');
        console.log('Dataset: 4096 vectors × 512 dimensions, finding top 8 neighbors\n');

        // Generate test dataset
        this.generateTestDataset(4096, 512, 10);

        // Calculate ground truth
        await this.calculateGroundTruth(8, 'euclidean');

        // Test different implementations with varying quality
        const implementations = [
            this.createMockKNNImplementation('Perfect Algorithm', 0.99),
            this.createMockKNNImplementation('High-Quality HNSW', 0.85),
            this.createMockKNNImplementation('Standard LSH', 0.72),
            this.createMockKNNImplementation('Fast Approximation', 0.55),
            this.createMockKNNImplementation('Random Search', 0.20)
        ];

        const results = [];

        for (const impl of implementations) {
            console.log(`\n🧪 Testing ${impl.name}...`);
            
            const startSetup = performance.now();
            const startSetupMem = process.memoryUsage().heapUsed / 1024 / 1024;
            
            await impl.initialize(512, 4096, 'euclidean');
            await impl.addVectors(this.testVectors);
            
            const setupTime = performance.now() - startSetup;
            const setupMemory = process.memoryUsage().heapUsed / 1024 / 1024 - startSetupMem;

            // Test all queries
            const queryResults = [];
            const accuracyMetrics = [];
            let totalSearchTime = 0;
            let totalSearchMemory = 0;

            for (const query of this.queryVectors) {
                const searchResult = await impl.knnSearch(query.vector, 8);
                
                totalSearchTime += searchResult.search_time_ms;
                totalSearchMemory += searchResult.memory_used_mb || 0;

                // Calculate accuracy against ground truth
                const groundTruth = this.groundTruth.get(query.id);
                if (groundTruth) {
                    const accuracy = this.calculateAccuracy(searchResult.results, groundTruth, 8);
                    accuracyMetrics.push(accuracy);
                }
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
                implementation: impl.name,
                setup_time_ms: setupTime,
                setup_memory_mb: setupMemory,
                avg_search_time_ms: totalSearchTime / numQueries,
                total_search_time_ms: totalSearchTime,
                avg_search_memory_mb: totalSearchMemory / numQueries,
                accuracy_metrics: avgAccuracy
            };

            results.push(result);
            console.log(`   ✅ Precision@8: ${(avgAccuracy.precision_at_k * 100).toFixed(1)}%, NDCG: ${avgAccuracy.ndcg_at_k.toFixed(3)}, Avg Search: ${(totalSearchTime / numQueries).toFixed(2)}ms`);
        }

        this.generateReport(results);
        return results;
    }

    /**
     * Generate detailed comparison report
     */
    generateReport(results) {
        console.log('\n📊 COMPREHENSIVE KNN ACCURACY ANALYSIS');
        console.log('=' .repeat(80));
        console.log('\n🎯 ACCURACY METRICS (What "average accuracy" really means):');
        console.log('Implementation'.padEnd(20) + 'Precision@8'.padEnd(12) + 'NDCG@8'.padEnd(10) + 'Exact Hits'.padEnd(12) + 'Dist Error');
        console.log('-'.repeat(80));
        
        results.forEach(result => {
            const acc = result.accuracy_metrics;
            console.log(
                result.implementation.padEnd(20) +
                (acc.precision_at_k * 100).toFixed(1).padEnd(11) + '%' +
                acc.ndcg_at_k.toFixed(3).padEnd(10) +
                acc.exact_matches.toFixed(1).padEnd(11) + '/8' +
                (acc.distance_error * 100).toFixed(1) + '%'
            );
        });

        console.log('\n⚡ PERFORMANCE METRICS:');
        console.log('Implementation'.padEnd(20) + 'Setup Time'.padEnd(12) + 'Avg Search'.padEnd(12) + 'Memory/Query');
        console.log('-'.repeat(70));
        
        results.forEach(result => {
            console.log(
                result.implementation.padEnd(20) +
                (result.setup_time_ms.toFixed(1) + 'ms').padEnd(12) +
                (result.avg_search_time_ms.toFixed(2) + 'ms').padEnd(12) +
                (result.avg_search_memory_mb.toFixed(2) + 'MB')
            );
        });

        console.log('\n📝 INTERPRETATION GUIDE:');
        console.log('• PRECISION@8: Of the 8 neighbors returned, what fraction are actually');
        console.log('  in the TRUE top-8 closest vectors (based on exhaustive search)');
        console.log('• NDCG@8: Normalized Discounted Cumulative Gain - measures ranking quality');
        console.log('  Perfect ranking = 1.0, random ranking ≈ 0.5');
        console.log('• EXACT HITS: Average number of exactly correct neighbors found out of 8');
        console.log('• DISTANCE ERROR: How much the average distance differs from ground truth');
        console.log('• All measurements are averages across 10 different query vectors');
        
        console.log('\n🔍 WHAT THIS TELLS US:');
        console.log('• A "90% accurate" algorithm means 90% of returned neighbors are truly');
        console.log('  in the optimal set - this is MUCH more meaningful than arbitrary scores');
        console.log('• NDCG shows if the algorithm gets the ORDER right, not just the set');
        console.log('• Distance error shows if the algorithm preserves actual similarity relationships');

        const best = results.reduce((best, curr) => 
            curr.accuracy_metrics.precision_at_k > best.accuracy_metrics.precision_at_k ? curr : best
        );
        console.log(`\n🏆 Best Accuracy: ${best.implementation} with ${(best.accuracy_metrics.precision_at_k * 100).toFixed(1)}% precision`);
    }
}

// Run the benchmark
const benchmark = new WorkingKNNBenchmark();
benchmark.runBenchmark()
    .then(results => {
        console.log(`\n✅ Benchmark completed successfully!`);
        process.exit(0);
    })
    .catch(error => {
        console.error('\n❌ Benchmark failed:', error.message);
        process.exit(1);
    });
