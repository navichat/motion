// Test script for the minimal FaceFormer model
async function testMinimalFaceFormer() {
    console.log('🧪 Testing Minimal FaceFormer model...');
    
    try {
        // Load the minimal FaceFormer generator
        const generator = new FaceFormerWebGeneratorFixed('./faceformer_minimal.onnx');
        
        console.log('📥 Initializing model...');
        await generator.initialize();
        
        console.log('🧪 Running model test...');
        const testResult = await generator.testModel();
        
        if (testResult) {
            console.log('✅ Model test passed!');
            
            // Test actual generation
            console.log('🎭 Testing generation...');
            const audioFeatures = new Array(768 * 10).fill(0.01); // 10 frames of audio
            const template = new Array(15069).fill(0.01);
            const oneHot = [1, 0, 0];
            
            const result = await generator.generateSequence(audioFeatures, template, oneHot, 5);
            
            console.log(`✅ Generation test passed! Created ${result.length} frames`);
            return true;
        } else {
            console.log('❌ Model test failed');
            return false;
        }
    } catch (error) {
        console.error('❌ Test failed:', error);
        return false;
    }
}

// Run test when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Add test button
    const testButton = document.createElement('button');
    testButton.textContent = 'Test Minimal FaceFormer';
    testButton.onclick = testMinimalFaceFormer;
    testButton.style.margin = '10px';
    testButton.style.padding = '10px';
    testButton.style.backgroundColor = '#4CAF50';
    testButton.style.color = 'white';
    testButton.style.border = 'none';
    testButton.style.borderRadius = '4px';
    
    document.body.appendChild(testButton);
});
