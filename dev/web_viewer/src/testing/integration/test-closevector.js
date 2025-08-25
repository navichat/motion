#!/usr/bin/env node

// Test CloseVector Interface
const CloseVectorInterface = require('./close-vector/closevector-interface.js');

console.log('Testing CloseVector Interface...');
console.log('Type:', typeof CloseVectorInterface);

try {
  const cv = new CloseVectorInterface();
  console.log('✅ Constructor works');
  
  // Test basic functionality
  async function runTest() {
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
    
    // Stats
    const stats = cv.getStats();
    console.log('4. Stats:', `${stats.total_vectors} vectors, ${stats.dimensions}D, ${stats.distance_metric} metric`);
    
    console.log('\n🎉 CloseVector interface test completed successfully!');
  }
  
  runTest().catch(console.error);
  
} catch (error) {
  console.error('❌ Constructor failed:', error.message);
  console.log('CloseVectorInterface keys:', Object.keys(CloseVectorInterface));
  console.log('CloseVectorInterface.prototype:', CloseVectorInterface.prototype);
}
