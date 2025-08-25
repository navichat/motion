// Simple test to check module loading
const CloseVectorInterface = require('./close-vector/closevector-interface.js');
const HNSWInterface = require('./hsnwlib/hnsw-interface.js');

console.log('CloseVectorInterface type:', typeof CloseVectorInterface);
console.log('CloseVectorInterface constructor:', CloseVectorInterface.constructor.name);
console.log('CloseVectorInterface prototype:', Object.getPrototypeOf(CloseVectorInterface));
console.log('CloseVectorInterface keys:', Object.keys(CloseVectorInterface));

console.log('\nHNSWInterface type:', typeof HNSWInterface);
console.log('HNSWInterface keys:', Object.keys(HNSWInterface));

// Try to create instances
try {
    const cv = new CloseVectorInterface();
    console.log('\n✅ CloseVectorInterface instance created successfully');
} catch (e) {
    console.log('\n❌ Failed to create CloseVectorInterface:', e.message);
}

try {
    const hnsw = new HNSWInterface();
    console.log('✅ HNSWInterface instance created successfully');
} catch (e) {
    console.log('❌ Failed to create HNSWInterface:', e.message);
}
