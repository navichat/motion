#!/usr/bin/env node

/**
 * Focused KNN Accuracy vs Speed Comparison
 * Compares: Exhaustive Search, CloseVector, and HNSW
 */

const CloseVectorInterface = require('./close-vector/closevector-interface.cjs');
const HNSWInterface = require('./hsnwlib/hnsw-interface.cjs');

class KNNSpeedAccuracyComparison {
    constructor() {
        this.testVectors = [];
        this.queryVectors = [];
    }

    generateTestDataset(numVectors = 1000, dimensions = 256, numQueries = 10) {
        console.log(`📊 Generating ${numVectors} vectors with ${dimensions} dimensions...`);
        
        // Seeded random for reproducibility
        let seed = 42;
        const seededRandom = () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };

        // Generate test vectors with clusters for realistic data
        this.testVectors = [];
        const numClusters = 5;
        const clusterCenters = Array.from({ length: numClusters }, () => 
            Array.from({ length: dimensions }, () => seededRandom() * 4 - 2)
        );

        for (let i = 0; i < numVectors; i++) {
            const clusterId = i % numClusters;
            const center = clusterCenters[clusterId];
            const noise = 0.6;
            
            const vector = center.map(val => val + (seededRandom() - 0.5) * noise);
            
            this.testVectors.push({
                id: `vector_${i}`,
                vector: vector,
                metadata: { cluster: clusterId }
            });
        }

        // Generate query vectors
        this.queryVectors = [];
        for (let i = 0; i < numQueries; i++) {
            const clusterId = i % numClusters;
            const center = clusterCenters[clusterId];
            const vector = center.map(val => val + (seededRandom() - 0.5) * 0.3);
            
            this.queryVectors.push({
                id: `query_${i}`,
                vector: vector
            });
        }

        console.log(`✅ Generated ${this.testVectors.length} test vectors and ${this.queryVectors.length} queries`);
    }

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
                throw new Error(`Unknown metric: ${metric}`);
        }
    }

    // Exhaustive brute-force search - 100% accurate but slow
    exhaustiveSearch(queryVector, k = 8) {
        const startTime = performance.now();

        const distances = this.testVectors.map(item => ({
            ...item,
            distance: this.calculateDistance(queryVector, item.vector, 'euclidean')
        }));

        distances.sort((a, b) => a.distance - b.distance);
        const results = distances.slice(0, k);

        const searchTime = performance.now() - startTime;

        return {
            results: results,
            search_time_ms: searchTime,
            method: 'exhaustive'
        };
    }

    // Calculate precision@k between two result sets
    calculatePrecision(predicted, groundTruth, k) {
        const predictedIds = new Set(predicted.slice(0, k).map(r => r.id));
        const trueIds = new Set(groundTruth.slice(0, k).map(r => r.id));
        const intersection = new Set([...predictedIds].filter(id => trueIds.has(id)));
        return intersection.size / k;
    }

    async testImplementation(implementation, name, config) {
        console.log(`\n🧪 Testing ${name}...`);
        
        const startSetup = performance.now();
        
        // Initialize
        const initResult = await implementation.initialize(
            config.dimensions,
            config.maxElements,
            config.metric,
            config.params || {}
        );

        if (!initResult.success) {
            throw new Error(`Failed to initialize ${name}`);
        }

        // Add vectors
        const addResult = await implementation.addVectors(this.testVectors);
        const setupTime = performance.now() - startSetup;

        console.log(`   Setup: ${setupTime.toFixed(2)}ms, Added: ${addResult.added} vectors`);

        // Test searches
        const searchTimes = [];
        const accuracies = [];

        for (const query of this.queryVectors) {
            // Get ground truth with exhaustive search
            const groundTruth = this.exhaustiveSearch(query.vector, 8);
            
            // Test implementation
            const startSearch = performance.now();
            const result = await implementation.knnSearch(query.vector, 8, config.searchParams || {});
            const searchTime = performance.now() - startSearch;

            searchTimes.push(searchTime);
            
            // Calculate accuracy
            const precision = this.calculatePrecision(result.results, groundTruth.results, 8);
            accuracies.push(precision);
        }

        const avgSearchTime = searchTimes.reduce((sum, t) => sum + t, 0) / searchTimes.length;
        const avgAccuracy = accuracies.reduce((sum, a) => sum + a, 0) / accuracies.length;

        return {
            name: name,
            setup_time_ms: setupTime,
            avg_search_time_ms: avgSearchTime,
            avg_accuracy: avgAccuracy,
            vectors_added: addResult.added,
            config: config
        };
    }

    async runComparison() {
        console.log('🎯 KNN ACCURACY vs SPEED COMPARISON');
        console.log('=====================================\n');

        // Generate smaller dataset for comparison
        this.generateTestDataset(1000, 256, 10);

        const results = [];

        // Test 1: Exhaustive Search (Ground Truth)
        console.log('\n🧪 Testing Exhaustive Search (Ground Truth)...');
        const exhaustiveResults = [];
        const exhaustiveTimes = [];

        for (const query of this.queryVectors) {
            const startTime = performance.now();
            const result = this.exhaustiveSearch(query.vector, 8);
            const searchTime = performance.now() - startTime;
            exhaustiveTimes.push(searchTime);
        }

        const avgExhaustiveTime = exhaustiveTimes.reduce((sum, t) => sum + t, 0) / exhaustiveTimes.length;
        
        results.push({
            name: 'Exhaustive Search',
            setup_time_ms: 0,
            avg_search_time_ms: avgExhaustiveTime,
            avg_accuracy: 1.0, // Perfect accuracy by definition
            vectors_added: this.testVectors.length,
            config: { method: 'brute_force' }
        });

        console.log(`   Search: ${avgExhaustiveTime.toFixed(2)}ms avg, Accuracy: 100% (ground truth)`);

        // Test 2: CloseVector (Mock Implementation)
        try {
            const closeVector = new CloseVectorInterface();
            const cvResult = await this.testImplementation(closeVector, 'CloseVector', {
                dimensions: 256,
                maxElements: 1000,
                metric: 'euclidean'
            });
            results.push(cvResult);
        } catch (error) {
            console.error(`   ❌ CloseVector failed: ${error.message}`);
        }

        // Test 3: HNSW (Approximate)
        try {
            const hnsw = new HNSWInterface();
            const hnswResult = await this.testImplementation(hnsw, 'HNSW', {
                dimensions: 256,
                maxElements: 1000,
                metric: 'l2',
                params: { M: 16, efConstruction: 200 },
                searchParams: { ef: 200 }
            });
            results.push(hnswResult);
        } catch (error) {
            console.error(`   ❌ HNSW failed: ${error.message}`);
        }

        // Generate comparison report
        this.generateReport(results);

        return results;
    }

    generateReport(results) {
        console.log('\n📊 ACCURACY vs SPEED TRADEOFF ANALYSIS');
        console.log('=' .repeat(70));

        console.log('\nImplementation'.padEnd(20) + 'Setup Time'.padEnd(12) + 'Search Time'.padEnd(14) + 'Accuracy'.padEnd(12) + 'Speedup');
        console.log('-'.repeat(70));

        const baselineTime = results.find(r => r.name === 'Exhaustive Search')?.avg_search_time_ms || 1;

        results.forEach(result => {
            const speedup = (baselineTime / result.avg_search_time_ms).toFixed(1) + 'x';
            console.log(
                result.name.padEnd(20) +
                (result.setup_time_ms.toFixed(1) + 'ms').padEnd(12) +
                (result.avg_search_time_ms.toFixed(2) + 'ms').padEnd(14) +
                (result.avg_accuracy * 100).toFixed(1).padEnd(8) + '%'.padEnd(4) +
                speedup
            );
        });

        console.log('\n🎯 KEY INSIGHTS:');
        
        const exhaustive = results.find(r => r.name === 'Exhaustive Search');
        const closeVector = results.find(r => r.name === 'CloseVector');
        const hnsw = results.find(r => r.name === 'HNSW');

        if (exhaustive) {
            console.log(`\n📍 EXHAUSTIVE SEARCH (Ground Truth):`);
            console.log(`   • Perfect accuracy: 100%`);
            console.log(`   • Search time: ${exhaustive.avg_search_time_ms.toFixed(2)}ms`);
            console.log(`   • Scales O(n) - linear with dataset size`);
            console.log(`   • Best for: Small datasets, when perfect accuracy is required`);
        }

        if (closeVector) {
            const cvSpeedup = exhaustive ? (exhaustive.avg_search_time_ms / closeVector.avg_search_time_ms).toFixed(1) : 'N/A';
            console.log(`\n🔍 CLOSEVECTOR (Brute Force Implementation):`);
            console.log(`   • Accuracy: ${(closeVector.avg_accuracy * 100).toFixed(1)}%`);
            console.log(`   • Search time: ${closeVector.avg_search_time_ms.toFixed(2)}ms`);
            console.log(`   • Speedup: ${cvSpeedup}x over exhaustive`);
            console.log(`   • Tradeoff: Similar to exhaustive search (mock implementation)`);
            console.log(`   • Best for: When you need exact results with interface compatibility`);
        }

        if (hnsw) {
            const hnswSpeedup = exhaustive ? (exhaustive.avg_search_time_ms / hnsw.avg_search_time_ms).toFixed(1) : 'N/A';
            console.log(`\n⚡ HNSW (Approximate Nearest Neighbors):`);
            console.log(`   • Accuracy: ${(hnsw.avg_accuracy * 100).toFixed(1)}%`);
            console.log(`   • Search time: ${hnsw.avg_search_time_ms.toFixed(2)}ms`);
            console.log(`   • Speedup: ${hnswSpeedup}x over exhaustive`);
            console.log(`   • Tradeoff: Faster search but may miss some true neighbors`);
            console.log(`   • Best for: Large datasets where speed > perfect accuracy`);
        }

        console.log(`\n📈 SUMMARY:`);
        if (exhaustive && closeVector && hnsw) {
            console.log(`   • Exhaustive: ${exhaustive.avg_search_time_ms.toFixed(1)}ms, 100% accuracy - SLOW but PERFECT`);
            console.log(`   • CloseVector: ${closeVector.avg_search_time_ms.toFixed(1)}ms, ${(closeVector.avg_accuracy * 100).toFixed(1)}% accuracy - BALANCED`);
            console.log(`   • HNSW: ${hnsw.avg_search_time_ms.toFixed(1)}ms, ${(hnsw.avg_accuracy * 100).toFixed(1)}% accuracy - FAST but APPROXIMATE`);
        }

        console.log(`\n💡 RECOMMENDATIONS:`);
        console.log(`   • < 1K vectors: Use Exhaustive Search for perfect results`);
        console.log(`   • 1K-10K vectors: Use CloseVector for good balance`);
        console.log(`   • > 10K vectors: Use HNSW for speed, tune parameters for accuracy`);
        console.log(`   • Real-time apps: HNSW with lower accuracy tolerance`);
        console.log(`   • Batch processing: Exhaustive or CloseVector for better accuracy`);
    }
}

// Run the comparison
const comparison = new KNNSpeedAccuracyComparison();
comparison.runComparison()
    .then(results => {
        console.log(`\n✅ Comparison completed with ${results.length} implementations tested`);
        process.exit(0);
    })
    .catch(error => {
        console.error('\n❌ Comparison failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    });
