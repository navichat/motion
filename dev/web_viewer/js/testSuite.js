/**
 * Debug and Test Script for VRM AI Conversation System
 * This script validates all components and provides comprehensive testing
 */

// Test configurations
const TEST_CONFIG = {
    enableFullLogging: true,
    testModelLoading: true,
    testAudioPipeline: true,
    testConversation: true,
    simulateUserInput: true
};

// Test results tracking
const testResults = {
    workerInitialization: false,
    transformersLoading: false,
    modelLoading: false,
    audioSetup: false,
    conversationFlow: false,
    ttsOutput: false
};

// Comprehensive logging
function debugLog(message, category = 'INFO', data = null) {
    if (!TEST_CONFIG.enableFullLogging) return;
    
    const timestamp = new Date().toISOString();
    const logEntry = {
        timestamp,
        category,
        message,
        data
    };
    
    console.log(`[${timestamp}] [${category}] ${message}`, data || '');
    
    // Send to UI if available
    if (typeof window !== 'undefined' && window.postMessage) {
        window.postMessage({
            type: 'debug_log',
            entry: logEntry
        }, '*');
    }
}

// Worker message handler with detailed logging
function createEnhancedWorkerHandler(worker) {
    worker.onmessage = (event) => {
        const { type, message, text, error, status, result } = event.data;
        
        debugLog(`Worker message: ${type}`, 'WORKER', event.data);
        
        switch (type) {
            case 'info':
                debugLog(message, 'WORKER_INFO');
                
                // Track specific achievements
                if (message.includes('Transformers.js loaded')) {
                    testResults.transformersLoading = true;
                    debugLog('✅ Transformers.js loading successful', 'SUCCESS');
                } else if (message.includes('VAD loaded')) {
                    debugLog('✅ Voice Activity Detection loaded', 'SUCCESS');
                } else if (message.includes('Whisper loaded')) {
                    debugLog('✅ Speech-to-Text loaded', 'SUCCESS');
                } else if (message.includes('Language model loaded')) {
                    debugLog('✅ Language Model loaded', 'SUCCESS');
                    testResults.modelLoading = true;
                }
                break;
                
            case 'error':
                debugLog(`Worker error: ${error || message}`, 'ERROR');
                break;
                
            case 'status':
                debugLog(`Status update: ${message}`, 'STATUS');
                if (status === 'ready') {
                    testResults.workerInitialization = true;
                    debugLog('✅ Worker initialization complete', 'SUCCESS');
                }
                break;
                
            case 'transcript':
                debugLog(`User speech: "${text}"`, 'CONVERSATION');
                break;
                
            case 'response':
                debugLog(`AI response: "${text}"`, 'CONVERSATION');
                testResults.conversationFlow = true;
                break;
                
            case 'output':
                if (result?.audio) {
                    debugLog('Audio output generated', 'TTS');
                    testResults.ttsOutput = true;
                }
                break;
        }
    };
    
    worker.onerror = (error) => {
        debugLog(`Worker error: ${error.message}`, 'ERROR', error);
    };
    
    return worker;
}

// Audio setup testing
async function testAudioSetup() {
    debugLog('Testing audio setup...', 'TEST');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });
        
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000,
        });
        
        testResults.audioSetup = true;
        debugLog('✅ Audio setup successful', 'SUCCESS');
        
        // Clean up
        stream.getTracks().forEach(track => track.stop());
        
        return true;
    } catch (error) {
        debugLog(`❌ Audio setup failed: ${error.message}`, 'ERROR');
        return false;
    }
}

// TTS testing
function testTTS() {
    debugLog('Testing Text-to-Speech...', 'TEST');
    
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance('Testing text to speech functionality');
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;
        
        utterance.onstart = () => {
            debugLog('✅ TTS playback started', 'SUCCESS');
        };
        
        utterance.onend = () => {
            debugLog('✅ TTS playback completed', 'SUCCESS');
        };
        
        utterance.onerror = (error) => {
            debugLog(`❌ TTS error: ${error.error}`, 'ERROR');
        };
        
        speechSynthesis.speak(utterance);
        return true;
    } else {
        debugLog('❌ Speech Synthesis not supported', 'ERROR');
        return false;
    }
}

// Conversation flow testing
async function testConversationFlow(worker) {
    if (!worker) {
        debugLog('❌ No worker available for conversation test', 'ERROR');
        return false;
    }
    
    debugLog('Testing conversation flow...', 'TEST');
    
    // Start a call
    worker.postMessage({ type: 'start_call' });
    
    // Wait a moment then simulate speech
    setTimeout(() => {
        const testPhrases = [
            "Hello, how are you?",
            "What's your name?",
            "Tell me about yourself",
            "What can you help me with?",
            "Thank you for talking with me"
        ];
        
        const phrase = testPhrases[Math.floor(Math.random() * testPhrases.length)];
        debugLog(`Simulating user speech: "${phrase}"`, 'TEST');
        
        // Simulate audio data
        const fakeAudioData = new Float32Array(1024).map(() => Math.random() * 0.1);
        worker.postMessage({
            type: 'audio',
            data: fakeAudioData
        });
    }, 1000);
    
    return true;
}

// Comprehensive system test
async function runComprehensiveTest() {
    debugLog('🚀 Starting comprehensive system test...', 'TEST');
    
    // Test 1: Audio setup
    if (TEST_CONFIG.testAudioPipeline) {
        await testAudioSetup();
    }
    
    // Test 2: TTS
    if (TEST_CONFIG.testConversation) {
        testTTS();
    }
    
    // Test 3: Worker initialization
    if (TEST_CONFIG.testModelLoading) {
        debugLog('Initializing worker for testing...', 'TEST');
        
        try {
            const worker = createEnhancedWorkerHandler(
                new Worker('./js/conversationWorkerWorking.js')
            );
            
            // Wait for initialization
            setTimeout(() => {
                if (TEST_CONFIG.testConversation) {
                    testConversationFlow(worker);
                }
            }, 3000);
            
            // Generate test report after 10 seconds
            setTimeout(() => {
                generateTestReport();
            }, 10000);
            
        } catch (error) {
            debugLog(`❌ Worker creation failed: ${error.message}`, 'ERROR');
        }
    }
}

// Generate test report
function generateTestReport() {
    debugLog('📊 Generating test report...', 'REPORT');
    
    const report = {
        timestamp: new Date().toISOString(),
        results: testResults,
        score: Object.values(testResults).filter(r => r).length,
        total: Object.keys(testResults).length
    };
    
    debugLog('Test Results:', 'REPORT', report);
    
    console.table(testResults);
    
    const successRate = (report.score / report.total * 100).toFixed(1);
    debugLog(`📈 Overall Success Rate: ${successRate}% (${report.score}/${report.total})`, 'REPORT');
    
    // Recommendations
    if (report.score === report.total) {
        debugLog('🎉 All tests passed! System is fully functional.', 'SUCCESS');
    } else {
        debugLog('⚠️ Some tests failed. Check the detailed logs above.', 'WARNING');
        
        if (!testResults.transformersLoading) {
            debugLog('💡 Recommendation: Check internet connection and CDN availability', 'ADVICE');
        }
        if (!testResults.audioSetup) {
            debugLog('💡 Recommendation: Grant microphone permissions and check audio devices', 'ADVICE');
        }
        if (!testResults.conversationFlow) {
            debugLog('💡 Recommendation: Verify worker message handling and model loading', 'ADVICE');
        }
    }
    
    return report;
}

// Browser compatibility check
function checkBrowserCompatibility() {
    debugLog('🔍 Checking browser compatibility...', 'TEST');
    
    const features = {
        'Web Workers': typeof Worker !== 'undefined',
        'Audio Worklet': window.AudioContext && 'audioWorklet' in AudioContext.prototype,
        'Web Audio API': typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined',
        'getUserMedia': !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        'Speech Synthesis': 'speechSynthesis' in window,
        'ES6 Modules': typeof window.importScripts !== 'undefined' || typeof module !== 'undefined',
        'WebAssembly': typeof WebAssembly !== 'undefined',
        'WebGPU': 'gpu' in navigator
    };
    
    debugLog('Browser Feature Support:', 'COMPATIBILITY', features);    const supportedFeatures = Object.values(features).filter(f => f).length;
    const totalFeatures = Object.keys(features).length;
    
    debugLog(`🌐 Browser Compatibility: ${supportedFeatures}/${totalFeatures} features supported`, 'COMPATIBILITY');
    
    return features;
}

// Export for use in HTML pages
if (typeof window !== 'undefined') {
    window.VRMTestSuite = {
        runComprehensiveTest,
        generateTestReport,
        checkBrowserCompatibility,
        testAudioSetup,
        testTTS,
        testConversationFlow,
        createEnhancedWorkerHandler,
        debugLog,
        testResults,
        TEST_CONFIG
    };
    
    debugLog('🛠️ VRM Test Suite loaded and ready', 'INIT');
}

// Auto-run compatibility check
if (typeof window !== 'undefined') {
    checkBrowserCompatibility();
}
