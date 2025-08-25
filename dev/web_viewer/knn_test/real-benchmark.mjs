/**
 * Real KNN Package Benchmark - ES Module Version
 * Testing actual hnswlib-wasm and closevector-web packages
 */

import { readFileSync, writeFileSync } from 'fs';
import { performance } from 'perf_hooks';

class RealKNNBenchmark {
    constructor() {
        this.testVectors = [];
        this.queryVectors = [];
        this.groundTruth = new Map();
        this.results = [];
    }

    /**
     * Generate deterministic test dataset
     */
    generateTestDataset(numVectors = 4096, dimensions = 512, numQueries = 20) {
        console.log(`📊 Generating ${numVectors} vectors with ${dimensions} dimensions...`);
        
        // Use seeded random for reproducibility
        let seed = 42;
        const seededRandom = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };

        // Generate test vectors with cluster structure
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
                metadata: { cluster: clusterId, index: i }
            });
        }

        // Generate query vectors 
        this.queryVectors = [];
        for (let i = 0; i < numQueries; i++) {
            let vector;
            if (i < numQueries / 2) {
                // Half from cluster centers (easier queries)
                const clusterId = i % numClusters;
                const center = clusterCenters[clusterId];
                vector = center.map(val => val + (seededRandom() - 0.5) * 0.1);
            } else {
                // Half random (harder queries)
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
     * Calculate euclidean distance
     */
    euclideanDistance(vec1, vec2) {
        return Math.sqrt(vec1.reduce((sum, v, i) => sum + Math.pow(v - vec2[i], 2), 0));
    }

    /**
     * Exhaustive brute-force search for ground truth
     */
    exhaustiveSearch(queryVector, k = 8) {
        const startTime = performance.now();
        const startMemory = process.memoryUsage().heapUsed / 1024 / 1024;

        // Calculate distances to all vectors
        const distances = this.testVectors.map(item => ({
            ...item,
            distance: this.euclideanDistance(queryVector, item.vector)
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
            method: 'exhaustive_brute_force'
        };
    }

    /**
     * Calculate ground truth for all queries
     */
    async calculateGroundTruth(k = 8) {
        console.log(`🎯 Calculating ground truth with exhaustive search (k=${k})...`);
        
        this.groundTruth.clear();
        const startTime = performance.now();

        for (let i = 0; i < this.queryVectors.length; i++) {
            const query = this.queryVectors[i];
            const truth = this.exhaustiveSearch(query.vector, k);
            this.groundTruth.set(query.id, truth);
            
            if ((i + 1) % 5 === 0) {
                console.log(`   Processed ${i + 1}/${this.queryVectors.length} queries...`);
            }
        }

        const totalTime = performance.now() - startTime;
        console.log(`✅ Ground truth calculated in ${totalTime.toFixed(2)}ms`);
        
        // Calculate average exhaustive search time per query
        const avgExhaustiveTime = totalTime / this.queryVectors.length;
        console.log(`📊 Exhaustive search baseline: ${avgExhaustiveTime.toFixed(2)}ms per query`);
    }

    /**
     * Calculate accuracy metrics
     */
    calculateAccuracy(predicted, groundTruth, k) {
        const predictedIds = new Set(predicted.slice(0, k).map(r => r.id || r));
        const trueIds = new Set(groundTruth.results.slice(0, k).map(r => r.id));
        
        // Precision@K: How many predicted are actually correct
        const intersection = new Set([...predictedIds].filter(id => trueIds.has(id)));
        const precisionAtK = intersection.size / k;

        // NDCG@K: Normalized Discounted Cumulative Gain
        let dcg = 0;
        let idcg = 0;
        
        for (let i = 0; i < k; i++) {
            const discount = Math.log2(i + 2);
            
            // DCG: score based on actual results
            if (i < predicted.length) {
                const predId = predicted[i].id || predicted[i];
                const isRelevant = trueIds.has(predId) ? 1 : 0;
                dcg += isRelevant / discount;
            }
            
            // IDCG: ideal score (perfect ranking)
            if (i < groundTruth.results.length) {
                idcg += 1 / discount;
            }
        }
        
        const ndcg = idcg > 0 ? dcg / idcg : 0;

        return {
            precision_at_k: precisionAtK,
            ndcg_at_k: ndcg,
            exact_matches: intersection.size
        };
    }

    /**
     * Test HNSWLib-WASM implementation
     */
    async testHNSWLib() {
        console.log('\n🧪 Testing HNSWLib-WASM...');
        
        try {
            const { HierarchicalNSW } = await import('hnswlib-wasm');
            
            const setupStartTime = performance.now();
            const setupStartMemory = process.memoryUsage().heapUsed / 1024 / 1024;
            
            // Initialize HNSW index
            const index = new HierarchicalNSW('l2', 512);
            index.initIndex(4096, 16, 200); // maxElements, M, efConstruction
            
            // Add vectors to index
            for (let i = 0; i < this.testVectors.length; i++) {
                index.addItem(this.testVectors[i].vector, i);
            }
            
            const setupTime = performance.now() - setupStartTime;
            const setupMemory = process.memoryUsage().heapUsed / 1024 / 1024 - setupStartMemory;
            
            console.log(`   Setup: ${setupTime.toFixed(2)}ms, Memory: ${setupMemory.toFixed(2)}MB`);
            
            // Test queries
            const queryResults = [];
            const accuracyMetrics = [];
            let totalSearchTime = 0;
            
            for (const query of this.queryVectors) {
                const searchStartTime = performance.now();
                
                // Search with ef parameter
                const results = index.searchKnn(query.vector, 8, { ef: 200 });
                
                const searchTime = performance.now() - searchStartTime;
                totalSearchTime += searchTime;
                
                // Convert results to our format
                const formattedResults = results.neighbors.map((idx, i) => ({
                    id: this.testVectors[idx].id,
                    distance: results.distances[i],
                    index: idx
                }));
                
                // Calculate accuracy
                const groundTruth = this.groundTruth.get(query.id);
                if (groundTruth) {
                    const accuracy = this.calculateAccuracy(formattedResults, groundTruth, 8);
                    accuracyMetrics.push(accuracy);
                }
                
                queryResults.push({
                    query_id: query.id,
                    search_time_ms: searchTime,
                    results_found: formattedResults.length,
                    results: formattedResults
                });
            }
            
            // Calculate averages
            const avgAccuracy = accuracyMetrics.reduce((sum, acc) => ({
                precision_at_k: sum.precision_at_k + acc.precision_at_k,
                ndcg_at_k: sum.ndcg_at_k + acc.ndcg_at_k,
                exact_matches: sum.exact_matches + acc.exact_matches
            }), { precision_at_k: 0, ndcg_at_k: 0, exact_matches: 0 });
            
            const numQueries = accuracyMetrics.length;
            Object.keys(avgAccuracy).forEach(key => {
                avgAccuracy[key] /= numQueries;
            });
            
            const result = {
                implementation: 'HNSWLib-WASM',
                setup_time_ms: setupTime,
                setup_memory_mb: setupMemory,
                avg_search_time_ms: totalSearchTime / numQueries,
                total_search_time_ms: totalSearchTime,
                accuracy_metrics: avgAccuracy,
                detailed_queries: queryResults
            };
            
            console.log(`   ✅ Precision@8: ${(avgAccuracy.precision_at_k * 100).toFixed(1)}%, NDCG: ${avgAccuracy.ndcg_at_k.toFixed(3)}, Avg Search: ${(totalSearchTime / numQueries).toFixed(2)}ms`);
            
            return result;
            
        } catch (error) {
            console.log(`   ❌ HNSWLib test failed: ${error.message}`);
            return null;
        }
    }

    /**
     * Test CloseVector implementation
     */
    async testCloseVector() {
        console.log('\n🧪 Testing CloseVector-Web...');
        
        try {
            const { CloseVectorHNSWWeb } = await import('closevector-web');
            
            const setupStartTime = performance.now();
            const setupStartMemory = process.memoryUsage().heapUsed / 1024 / 1024;
            
            // Initialize CloseVector HNSW
            const cv = new CloseVectorHNSWWeb(512, 'l2'); // dimensions, distance metric
            await cv.init(4096, 16, 200); // maxElements, M, efConstruction
            
            // Add vectors
            for (let i = 0; i < this.testVectors.length; i++) {
                cv.addItem(this.testVectors[i].vector, i);
            }
            
            const setupTime = performance.now() - setupStartTime;
            const setupMemory = process.memoryUsage().heapUsed / 1024 / 1024 - setupStartMemory;
            
            console.log(`   Setup: ${setupTime.toFixed(2)}ms, Memory: ${setupMemory.toFixed(2)}MB`);
            
            // Test queries
            const queryResults = [];
            const accuracyMetrics = [];
            let totalSearchTime = 0;
            
            for (const query of this.queryVectors) {
                const searchStartTime = performance.now();
                
                // Search for nearest neighbors
                const results = cv.searchKnn(query.vector, 8, 200); // vector, k, ef
                
                const searchTime = performance.now() - searchStartTime;
                totalSearchTime += searchTime;
                
                // Format results
                const formattedResults = results.neighbors.map((idx, i) => ({
                    id: this.testVectors[idx].id,
                    distance: results.distances[i],
                    index: idx
                }));
                
                // Calculate accuracy
                const groundTruth = this.groundTruth.get(query.id);
                if (groundTruth) {
                    const accuracy = this.calculateAccuracy(formattedResults, groundTruth, 8);
                    accuracyMetrics.push(accuracy);
                }
                
                queryResults.push({
                    query_id: query.id,
                    search_time_ms: searchTime,
                    results_found: formattedResults.length,
                    results: formattedResults
                });
            }
            
            // Calculate averages
            const avgAccuracy = accuracyMetrics.reduce((sum, acc) => ({
                precision_at_k: sum.precision_at_k + acc.precision_at_k,
                ndcg_at_k: sum.ndcg_at_k + acc.ndcg_at_k,
                exact_matches: sum.exact_matches + acc.exact_matches
            }), { precision_at_k: 0, ndcg_at_k: 0, exact_matches: 0 });
            
            const numQueries = accuracyMetrics.length;
            Object.keys(avgAccuracy).forEach(key => {
                avgAccuracy[key] /= numQueries;
            });
            
            const result = {
                implementation: 'CloseVector-Web (HNSW)',
                setup_time_ms: setupTime,
                setup_memory_mb: setupMemory,
                avg_search_time_ms: totalSearchTime / numQueries,
                total_search_time_ms: totalSearchTime,
                accuracy_metrics: avgAccuracy,
                detailed_queries: queryResults
            };
            
            console.log(`   ✅ Precision@8: ${(avgAccuracy.precision_at_k * 100).toFixed(1)}%, NDCG: ${avgAccuracy.ndcg_at_k.toFixed(3)}, Avg Search: ${(totalSearchTime / numQueries).toFixed(2)}ms`);
            
            return result;
            
        } catch (error) {
            console.log(`   ❌ CloseVector test failed: ${error.message}`);
            return null;
        }
    }

    /**
     * Run comprehensive benchmark
     */
    async runBenchmark() {
        console.log('🚀 Real KNN Package Benchmark: Exact Accuracy & CPU Time Measurements\n');
        console.log('Testing: hnswlib-wasm vs closevector-web vs exhaustive search');
        console.log('Dataset: 4096 vectors × 512 dimensions, finding top 8 neighbors\n');

        // Generate test dataset
        this.generateTestDataset(4096, 512, 20);

        // Calculate ground truth
        await this.calculateGroundTruth(8);

        // Test implementations
        const results = [];
        
        // Test HNSW
        const hnswResult = await this.testHNSWLib();
        if (hnswResult) results.push(hnswResult);
        
        // Test CloseVector
        const cvResult = await this.testCloseVector();
        if (cvResult) results.push(cvResult);

        // Generate comparison report
        this.generateReport(results);
        
        return results;
    }

    /**
     * Generate detailed comparison report
     */
    generateReport(results) {
        console.log('\n📊 REAL KNN PACKAGE BENCHMARK RESULTS');
        console.log('=' .repeat(80));

        if (results.length === 0) {
            console.log('❌ No successful benchmark results to display');
            return;
        }

        // Get exhaustive search baseline
        const exhaustiveTimes = Array.from(this.groundTruth.values()).map(gt => gt.search_time_ms);
        const avgExhaustiveTime = exhaustiveTimes.reduce((sum, t) => sum + t, 0) / exhaustiveTimes.length;

        console.log('\n🎯 ACCURACY vs GROUND TRUTH:');
        console.log('Implementation'.padEnd(25) + 'Precision@8'.padEnd(12) + 'NDCG@8'.padEnd(10) + 'Exact Hits'.padEnd(12) + 'Quality');
        console.log('-'.repeat(80));
        
        results.forEach(result => {
            const acc = result.accuracy_metrics;
            const quality = acc.precision_at_k >= 0.9 ? 'Excellent' : 
                           acc.precision_at_k >= 0.8 ? 'Very Good' :
                           acc.precision_at_k >= 0.7 ? 'Good' :
                           acc.precision_at_k >= 0.6 ? 'Fair' : 'Poor';
            
            console.log(
                result.implementation.padEnd(25) +
                (acc.precision_at_k * 100).toFixed(1).padEnd(11) + '%' +
                acc.ndcg_at_k.toFixed(3).padEnd(10) +
                acc.exact_matches.toFixed(1).padEnd(11) + '/8' +
                quality
            );
        });

        console.log('\n⚡ PERFORMANCE vs EXHAUSTIVE SEARCH:');
        console.log('Implementation'.padEnd(25) + 'Setup Time'.padEnd(12) + 'Avg Search'.padEnd(12) + 'Speedup'.padEnd(10) + 'Memory');
        console.log('-'.repeat(80));
        
        // Add exhaustive search baseline
        console.log(
            'Exhaustive Search'.padEnd(25) +
            '0.0ms'.padEnd(12) +
            (avgExhaustiveTime.toFixed(2) + 'ms').padEnd(12) +
            '1.0x'.padEnd(10) +
            '~0MB'
        );
        
        results.forEach(result => {
            const speedup = avgExhaustiveTime / result.avg_search_time_ms;
            console.log(
                result.implementation.padEnd(25) +
                (result.setup_time_ms.toFixed(1) + 'ms').padEnd(12) +
                (result.avg_search_time_ms.toFixed(2) + 'ms').padEnd(12) +
                (speedup.toFixed(1) + 'x').padEnd(10) +
                (result.setup_memory_mb.toFixed(1) + 'MB')
            );
        });

        console.log('\n📊 DETAILED ANALYSIS:');
        
        const best = results.reduce((best, curr) => 
            curr.accuracy_metrics.precision_at_k > best.accuracy_metrics.precision_at_k ? curr : best
        );
        
        const fastest = results.reduce((fastest, curr) => 
            curr.avg_search_time_ms < fastest.avg_search_time_ms ? curr : fastest
        );

        console.log(`🏆 Best Accuracy: ${best.implementation}`);
        console.log(`   - ${(best.accuracy_metrics.precision_at_k * 100).toFixed(1)}% of returned neighbors are actually in the true top-8`);
        console.log(`   - ${best.accuracy_metrics.exact_matches.toFixed(1)} out of 8 neighbors are exactly correct on average`);
        
        console.log(`⚡ Fastest Search: ${fastest.implementation}`);
        console.log(`   - ${fastest.avg_search_time_ms.toFixed(2)}ms average search time`);
        const fastestSpeedup = avgExhaustiveTime / fastest.avg_search_time_ms;
        console.log(`   - ${fastestSpeedup.toFixed(1)}x faster than exhaustive brute-force search`);

        console.log('\n📝 WHAT THESE NUMBERS MEAN:');
        console.log('• Precision@8: If you ask for 8 similar vectors, this % will actually be in the true top-8');
        console.log('• NDCG@8: Ranking quality - how well does the algorithm order the results? (1.0 = perfect)');
        console.log('• Exact Hits: On average, how many of the 8 returned are exactly right?');
        console.log('• Speedup: How much faster than checking every single vector individually');
        console.log('• These are REAL measurements using actual hnswlib-wasm and closevector-web packages');

        // Save detailed results
        const reportData = {
            timestamp: new Date().toISOString(),
            dataset: {
                vectors: this.testVectors.length,
                dimensions: this.testVectors[0]?.vector.length || 512,
                queries: this.queryVectors.length
            },
            exhaustive_baseline_ms: avgExhaustiveTime,
            results: results
        };
        
        writeFileSync('real_knn_benchmark_results.json', JSON.stringify(reportData, null, 2));
        console.log('\n💾 Detailed results saved to: real_knn_benchmark_results.json');
    }
}

// Run the benchmark
const benchmark = new RealKNNBenchmark();
benchmark.runBenchmark()
    .then(results => {
        console.log(`\n✅ Real KNN benchmark completed with ${results.length} successful tests!`);
        process.exit(0);
    })
    .catch(error => {
        console.error('\n❌ Benchmark failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    });
