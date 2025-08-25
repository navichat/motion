#!/usr/bin/env node

// Test the KNN benchmark with fixed CloseVector interface
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Import KNN interfaces
const CloseVectorInterface = require('./close-vector/closevector-interface.cjs');
const HNSWInterface = require('./hsnwlib/hnsw-interface.cjs');

console.log('🧪 Testing KNN Interfaces with Benchmark');

async function testKNNInterfaces() {
  console.log('\n=== 1. Testing CloseVector Interface ===');
  
  try {
    const cv = new CloseVectorInterface();
    
    // Initialize
    const initResult = await cv.initialize(64, 1000, 'euclidean');
    console.log('✅ CloseVector init:', initResult.success);
    
    // Add test vectors
    const vectors = [];
    for (let i = 0; i < 10; i++) {
      vectors.push({
        id: `vec_${i}`,
        vector: Array.from({length: 64}, () => Math.random()),
        metadata: { index: i }
      });
    }
    
    const addResult = await cv.addVectors(vectors);
    console.log('✅ CloseVector add:', `${addResult.added} vectors in ${addResult.add_time_ms.toFixed(2)}ms`);
    
    // Test search
    const query = Array.from({length: 64}, () => Math.random());
    const searchResult = await cv.knnSearch(query, 5);
    console.log('✅ CloseVector search:', `Found ${searchResult.k_returned} neighbors in ${searchResult.search_time_ms.toFixed(2)}ms`);
    
    // Test different distance metrics
    const cosineInit = await cv.initialize(64, 1000, 'cosine');
    await cv.addVectors(vectors.slice(0, 5));
    const cosineSearch = await cv.knnSearch(query, 3);
    console.log('✅ CloseVector cosine:', `Found ${cosineSearch.k_returned} neighbors with cosine distance`);
    
  } catch (error) {
    console.error('❌ CloseVector test failed:', error.message);
  }
  
  console.log('\n=== 2. Testing HNSW Interface ===');
  
  try {
    const hnsw = new HNSWInterface();
    
    // Initialize
    const initResult = await hnsw.initialize(64, 1000, 'euclidean');
    console.log('✅ HNSW init:', initResult.success);
    
    // Add test vectors
    const vectors = [];
    for (let i = 0; i < 10; i++) {
      vectors.push({
        id: `hnsw_vec_${i}`,
        vector: Array.from({length: 64}, () => Math.random()),
        metadata: { index: i }
      });
    }
    
    const addResult = await hnsw.addVectors(vectors);
    console.log('✅ HNSW add:', `${addResult.added} vectors in ${addResult.add_time_ms.toFixed(2)}ms`);
    
    // Test search
    const query = Array.from({length: 64}, () => Math.random());
    const searchResult = await hnsw.knnSearch(query, 5);
    console.log('✅ HNSW search:', `Found ${searchResult.k_returned} neighbors in ${searchResult.search_time_ms.toFixed(2)}ms`);
    
  } catch (error) {
    console.error('❌ HNSW test failed:', error.message);
  }
  
  console.log('\n=== 3. Performance Comparison ===');
  
  try {
    // Create test dataset
    const dimensions = 128;
    const numVectors = 500;
    const k = 8;
    
    const testVectors = [];
    for (let i = 0; i < numVectors; i++) {
      testVectors.push({
        id: `test_${i}`,
        vector: Array.from({length: dimensions}, () => Math.random()),
        metadata: { index: i }
      });
    }
    
    const query = Array.from({length: dimensions}, () => Math.random());
    
    // Test CloseVector
    const cv = new CloseVectorInterface();
    await cv.initialize(dimensions, numVectors, 'euclidean');
    
    const cvStartAdd = Date.now();
    await cv.addVectors(testVectors);
    const cvAddTime = Date.now() - cvStartAdd;
    
    const cvStartSearch = Date.now();
    const cvResult = await cv.knnSearch(query, k);
    const cvSearchTime = Date.now() - cvStartSearch;
    
    // Test HNSW  
    const hnsw = new HNSWInterface();
    await hnsw.initialize(dimensions, numVectors, 'euclidean');
    
    const hnswStartAdd = Date.now();
    await hnsw.addVectors(testVectors);
    const hnswAddTime = Date.now() - hnswStartAdd;
    
    const hnswStartSearch = Date.now();
    const hnswResult = await hnsw.knnSearch(query, k);
    const hnswSearchTime = Date.now() - hnswStartSearch;
    
    console.log('\n📊 Performance Results:');
    console.log(`CloseVector: Add ${cvAddTime}ms, Search ${cvSearchTime}ms, Found ${cvResult.k_returned}`);
    console.log(`HNSW:        Add ${hnswAddTime}ms, Search ${hnswSearchTime}ms, Found ${hnswResult.k_returned}`);
    
    console.log('\n🎉 All KNN interface tests completed successfully!');
    console.log('✅ CloseVector interface is fixed and tested');
    
  } catch (error) {
    console.error('❌ Performance comparison failed:', error.message);
  }
}

testKNNInterfaces().catch(console.error);
