// Test the simple KNN interface
const SimpleKNNInterface = require('./simple-knn-test.cjs');

console.log('SimpleKNNInterface type:', typeof SimpleKNNInterface);
console.log('SimpleKNNInterface name:', SimpleKNNInterface.name);

try {
    const knn = new SimpleKNNInterface();
    console.log('✅ SimpleKNNInterface instance created successfully');
    
    // Test initialization
    knn.initialize(512, 4096, 'euclidean').then(result => {
        console.log('✅ Initialization result:', result);
    });
    
} catch (e) {
    console.log('❌ Failed to create SimpleKNNInterface:', e.message);
}
