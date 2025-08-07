const { Audio2GestureChunkedGenerator } = require('./audio2gesture_chunked_generator');

async function testWithoutOriginals() {
    console.log('🧪 Testing chunked system without original files...');
    
    try {
        const generator = new Audio2GestureChunkedGenerator();
        
        // Test loading non-existent file that should fallback to chunks
        await generator.initialize('./motion_generator.onnx');
        console.log('✅ Successfully loaded from chunks when original missing!');
        
        const inputs = generator.createSampleInputs('full');
        const results = await generator.generateSequence(inputs, 3);
        console.log(`✅ Generated ${results.frames.length} frames successfully!`);
        
        console.log('🎉 Chunked system working perfectly without original files!');
        return true;
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        return false;
    }
}

testWithoutOriginals();
