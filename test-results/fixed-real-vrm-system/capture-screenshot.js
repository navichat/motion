// Simple screenshot capture for VRM system
console.log('📸 Starting screenshot capture...');

setTimeout(() => {
    // Initialize system
    const initButton = document.getElementById('init-button');
    if (initButton) {
        console.log('⚡ Initializing VRM system...');
        initButton.click();
        
        setTimeout(() => {
            console.log('✅ VRM system initialization complete');
            
            // Test voice
            const testVoiceButton = document.getElementById('test-voice');
            if (testVoiceButton) {
                console.log('🎤 Testing voice system...');
                testVoiceButton.click();
            }
            
            setTimeout(() => {
                // System ready for screenshot
                console.log('🎯 System ready for screenshot capture');
                document.title = 'READY_FOR_SCREENSHOT';
            }, 3000);
            
        }, 8000);
    }
}, 2000);
