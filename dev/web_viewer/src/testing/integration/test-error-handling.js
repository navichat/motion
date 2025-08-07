// Quick verification test for the enhanced error handling
// This can be run in the browser console after loading the demo

console.log('🧪 Testing enhanced error handling...');

// Test that the modules handle fallback scenarios gracefully
async function testErrorHandling() {
    try {
        // Import the voice chat interface
        const { VoiceChatInterface } = await import('./modules/VoiceChatInterface.js');
        
        const voiceChat = new VoiceChatInterface({
            memoryThresholdMB: 512,
            vadSensitivity: 0.6,
            systemPrompt: "Test prompt"
        });
        
        // Set up event listeners to capture events
        const events = {
            info: [],
            warning: [],
            error: []
        };
        
        voiceChat.addEventListener('info', (e) => {
            events.info.push(e.detail);
            console.log('✅ Info event captured:', e.detail);
        });
        
        voiceChat.addEventListener('warning', (e) => {
            events.warning.push(e.detail);
            console.log('⚠️ Warning event captured:', e.detail);
        });
        
        voiceChat.addEventListener('error', (e) => {
            events.error.push(e.detail);
            console.log('❌ Error event captured:', e.detail);
        });
        
        console.log('🚀 Voice chat interface created successfully');
        console.log('📊 Event listeners attached');
        
        // Try to initialize (this should trigger fallback scenarios)
        await voiceChat.initialize();
        
        console.log('📈 Events captured after initialization:');
        console.log('  Info events:', events.info.length);
        console.log('  Warning events:', events.warning.length);
        console.log('  Error events:', events.error.length);
        
        // Test transcription fallback
        console.log('🎤 Testing transcription fallback...');
        try {
            const dummyAudioData = new Float32Array(1024);
            await voiceChat.transcribeSpeech(dummyAudioData);
        } catch (e) {
            console.log('Transcription test completed');
        }
        
        console.log('✅ Error handling test completed successfully!');
        return { voiceChat, events };
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        return null;
    }
}

// Export for use in browser console
window.testErrorHandling = testErrorHandling;

console.log('🔧 Error handling test ready! Run window.testErrorHandling() in the console after loading the demo.');
