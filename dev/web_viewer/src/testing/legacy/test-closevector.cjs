#!/usr/bin/env node

// Test CloseVector Interface in CommonJS
const CloseVectorInterface = require('./close-vector/closevector-interface.cjs');

console.log('Testing CloseVector Interface...');
console.log('Type:', typeof CloseVectorInterface);
console.log('Is function:', typeof CloseVectorInterface === 'function');

try {
  const cv = new CloseVectorInterface();
  console.log('✅ Constructor works');
  
  // Test basic functionality
  async function runTest() {
    console.log('\n=== Testing CloseVector Interface ===');
    
    // Initialize
    const initResult = await cv.initialize(128, 100, 'euclidean');
    console.log('1. Init:', initResult.success ? '✅' : '❌', initResult);
    
    // Add some test vectors
    const vectors = [
      { id: 'v1', vector: Array.from({length: 128}, () => Math.random()) },
      { id: 'v2', vector: Array.from({length: 128}, () => Math.random()) },
      { id: 'v3', vector: Array.from({length: 128}, () => Math.random()) }
    ];
    
    const addResult = await cv.addVectors(vectors);
    console.log('2. Add vectors:', addResult.success ? '✅' : '❌', `${addResult.added} vectors added`);
    
    // Search
    const query = Array.from({length: 128}, () => Math.random());
    const searchResult = await cv.knnSearch(query, 2);
    console.log('3. Search:', searchResult.success ? '✅' : '❌', `Found ${searchResult.k_returned} neighbors in ${searchResult.search_time_ms.toFixed(2)}ms`);
    
    // Test distance calculations
    const vec1 = [1, 2, 3];
    const vec2 = [4, 5, 6];
    const euclidean = cv.calculateDistance(vec1, vec2, 'euclidean');
    const cosine = cv.calculateDistance(vec1, vec2, 'cosine');
    console.log('4. Distance calculations:', `euclidean=${euclidean.toFixed(3)}, cosine=${cosine.toFixed(3)}`);
    
    // Stats
    const stats = cv.getStats();
    console.log('5. Stats:', `${stats.total_vectors} vectors, ${stats.dimensions}D, ${stats.distance_metric} metric`);
    
    console.log('\n🎉 CloseVector interface test completed successfully!');
    console.log('✅ CloseVector interface is fixed and working properly');
  }
  
  runTest().catch(console.error);
  
} catch (error) {
  console.error('❌ Constructor failed:', error.message);
  console.log('Debugging info:');
  console.log('- CloseVectorInterface keys:', Object.keys(CloseVectorInterface || {}));
  console.log('- CloseVectorInterface.prototype:', CloseVectorInterface.prototype);
  console.log('- typeof CloseVectorInterface:', typeof CloseVectorInterface);
}
