/**
 * Enhanced Audio-to-Audio Conversation Worker
 * With graceful fallback for TTS loading
 */

// Global state variables
let messages = [];
let voice = "af_heart";
let isPlaying = false;
let isRecording = false;
let callActive = false;
let lastSpeechTime = 0;
let speechCooldown = 5000;
let tts = null;
let voices = {};
let KokoroTTS = null;
let TextSplitterStream = null;

// Try to load Kokoro TTS with fallback
async function loadTTSModules() {
    try {
        const kokoroModule = await import("kokoro-js");
        KokoroTTS = kokoroModule.KokoroTTS;
        TextSplitterStream = kokoroModule.TextSplitterStream;
        return true;
    } catch (error) {
        console.warn("Failed to load kokoro-js:", error);
        return false;
    }
}

// Initialize TTS and voices
async function initializeModels() {
    try {
        self.postMessage({ type: "info", message: "🚀 Starting worker initialization..." });
        
        // Try to load TTS modules
        const ttsLoaded = await loadTTSModules();
        
        if (ttsLoaded && KokoroTTS) {
            try {
                // Load Kokoro TTS
                self.postMessage({ type: "info", message: "Loading Kokoro TTS model..." });
                const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
                tts = await KokoroTTS.from_pretrained(model_id, {
                    dtype: "fp32",
                    device: "webgpu",
                });
                
                voices = tts.voices || {
                    "af_heart": { name: "Heart", language: "en-us", gender: "female" },
                    "am_adam": { name: "Adam", language: "en-us", gender: "male" },
                    "af_sarah": { name: "Sarah", language: "en-us", gender: "female" }
                };
                
                self.postMessage({ type: "info", message: "✅ Kokoro TTS loaded successfully" });
            } catch (ttsError) {
                console.warn("Failed to load Kokoro TTS model:", ttsError);
                tts = null;
                self.postMessage({ type: "info", message: "⚠️ TTS model loading failed, using fallback" });
            }
        } else {
            self.postMessage({ type: "info", message: "⚠️ Kokoro TTS not available, using Web Speech API" });
        }
        
        // Set fallback voices if TTS failed
        if (!tts || !voices || Object.keys(voices).length === 0) {
            voices = {
                "af_heart": { name: "Heart", language: "en-us", gender: "female" },
                "am_adam": { name: "Adam", language: "en-us", gender: "male" },
                "af_sarah": { name: "Sarah", language: "en-us", gender: "female" }
            };
        }
        
        // Set default voice
        voice = Object.keys(voices)[0] || "af_heart";
        
        // Initialize conversation context
        messages = [{
            role: "system",
            content: "You're a helpful and conversational AI avatar assistant. Keep your responses short, clear, and casual."
        }];

        const statusMessage = tts ? "System ready with Kokoro TTS!" : "System ready (Web Speech API fallback)!";
        self.postMessage({
            type: "status",
            status: "ready",
            message: statusMessage,
            voices: voices
        });

    } catch (error) {
        console.error('Model initialization error:', error);
        self.postMessage({ error: `Initialization failed: ${error.message}` });
    }
}

// Simplified speech processing for testing
function processSimpleResponse(userInput = "user input") {
    const responses = [
        "That's interesting! Tell me more about that.",
        "I understand what you're saying. Thanks for sharing!",
        "That's a great point. What do you think about it?",
        "I'm listening! Please continue.",
        "Thanks for talking with me. I enjoy our conversation!",
        "Could you elaborate on that?",
        "That sounds fascinating! What happened next?",
        "I see what you mean. How did that make you feel?",
        "That's a really good observation.",
        "I appreciate you sharing that with me."
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
}

// Generate audio using Kokoro TTS or fallback
async function generateRealAudio(text) {
    try {
        if (tts && TextSplitterStream && voice) {
            self.postMessage({ type: "info", message: "🎤 Generating TTS audio..." });
            
            // Use streaming TTS like in the reference implementation
            const splitter = new TextSplitterStream();
            const stream = tts.stream(splitter, { voice });
            
            // Collect all audio chunks
            const audioChunks = [];
            splitter.push(text);
            splitter.close();
            
            for await (const { audio } of stream) {
                if (audio && audio.length > 0) {
                    audioChunks.push(audio);
                }
            }
            
            // Concatenate all audio chunks
            if (audioChunks.length > 0) {
                const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
                const combinedAudio = new Float32Array(totalLength);
                let offset = 0;
                for (const chunk of audioChunks) {
                    combinedAudio.set(chunk, offset);
                    offset += chunk.length;
                }
                self.postMessage({ type: "info", message: "✅ TTS audio generated" });
                return combinedAudio;
            }
        }
        
        // Fallback: return flag for Web Speech API
        self.postMessage({ type: "info", message: "🔊 Using Web Speech API fallback" });
        return { useWebSpeech: true, text: text };
        
    } catch (error) {
        console.error('TTS generation error:', error);
        self.postMessage({ type: "info", message: "⚠️ TTS failed, using Web Speech API" });
        return { useWebSpeech: true, text: text };
    }
}

// Simulate speech detection (only when call is active)
function simulateSpeechDetection() {
    if (!callActive) return false;
    if (isRecording || isPlaying) return false;
    
    const now = Date.now();
    if (now - lastSpeechTime < speechCooldown) return false;
    
    // Very reduced chance - only 5% chance to avoid constant triggering
    if (Math.random() > 0.95) {
        lastSpeechTime = now;
        return true;
    }
    return false;
}

// Main message handler
self.onmessage = async (event) => {
    const { type } = event.data;

    try {
        switch (type) {
            case "start_call":
                callActive = true;
                const greeting = `Hello! I'm ${voices[voice]?.name || 'your AI assistant'}. How can I help you today?`;
                const greetingAudio = await generateRealAudio(greeting);
                self.postMessage({ 
                    type: "output", 
                    text: greeting, 
                    result: { audio: greetingAudio }
                });
                break;
                
            case "end_call":
                callActive = false;
                isRecording = false;
                isPlaying = false;
                self.postMessage({ type: "info", message: "Call ended" });
                break;
                
            case "set_voice":
                voice = event.data.voice || "af_heart";
                self.postMessage({ type: "info", message: `Voice changed to: ${voices[voice]?.name || voice}` });
                break;
                
            case "audio":
                // Only process audio when call is active and not currently playing
                if (!callActive || isPlaying) {
                    return;
                }
                
                // Simulate speech detection and response
                if (simulateSpeechDetection()) {
                    isRecording = true;
                    self.postMessage({ type: "status", status: "recording_start", message: "Listening..." });
                    
                    // Simulate processing delay
                    setTimeout(async () => {
                        if (!callActive) return;
                        
                        isRecording = false;
                        self.postMessage({ type: "status", status: "recording_end", message: "Processing..." });
                        
                        isPlaying = true;
                        const response = processSimpleResponse("user input");
                        
                        // Generate real audio
                        const responseAudio = await generateRealAudio(response);
                        
                        // Add to conversation history
                        messages.push({ role: "user", content: "User said something..." });
                        messages.push({ role: "assistant", content: response });
                        
                        self.postMessage({ 
                            type: "output", 
                            text: response, 
                            result: { audio: responseAudio }
                        });
                    }, 1000 + Math.random() * 500);
                }
                break;
                
            case "playback_started":
                isPlaying = true;
                break;
                
            case "playback_ended":
                isPlaying = false;
                self.postMessage({ type: "info", message: "Ready for next input" });
                break;
                
            default:
                self.postMessage({ type: "info", message: `Unknown message type: ${type}` });
        }
    } catch (error) {
        console.error('Worker message handling error:', error);
        self.postMessage({ error: `Worker error: ${error.message}` });
    }
};

// Error handler
self.addEventListener('error', (error) => {
    console.error('Worker error:', error);
    self.postMessage({ error: `Worker error: ${error.message}` });
});

// Unhandled rejection handler
self.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection in worker:', event.reason);
    self.postMessage({ error: `Promise rejection: ${event.reason}` });
});

// Initialize when worker starts
self.postMessage({ type: "info", message: "🚀 Starting conversation worker..." });
initializeModels().catch(error => {
    console.error('Failed to initialize models:', error);
    self.postMessage({ error: `Failed to initialize: ${error.message}` });
});
