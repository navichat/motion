/**
 * Enhanced Audio-to-Audio Conversation Worker
 * Simplified version with better error handling
 */

// Global state variables
let messages = [];
let voice = "af_heart";
let isPlaying = false;
let tts = null;
let isRecording = false;
let past_key_values_cache = null;
let bufferPointer = 0;

// Constants for audio processing
const INPUT_SAMPLE_RATE = 16000;
const SPEECH_THRESHOLD = 0.3;
const EXIT_THRESHOLD = 0.1;
const MIN_SILENCE_DURATION_SAMPLES = 400 * INPUT_SAMPLE_RATE / 1000;
const SPEECH_PAD_SAMPLES = 80 * INPUT_SAMPLE_RATE / 1000;
const MIN_SPEECH_DURATION_SAMPLES = 250 * INPUT_SAMPLE_RATE / 1000;
const MAX_BUFFER_DURATION = 30;
const BUFFER = new Float32Array(MAX_BUFFER_DURATION * INPUT_SAMPLE_RATE);

// Initialize all AI models
async function initializeModels() {
    try {
        self.postMessage({ type: "info", message: "🚀 Starting worker initialization..." });
        
        // For now, simulate model loading to test the interface
        self.postMessage({ type: "info", message: "⚠️ Using simplified mode for testing" });
        
        // Simulate loading delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Initialize basic conversation context
        messages = [{
            role: "system",
            content: "You're a helpful and conversational AI avatar assistant. Keep your responses short, clear, and casual."
        }];

        // Send ready status with mock voices
        self.postMessage({
            type: "status",
            status: "ready",
            message: "System ready (simplified mode)!",
            voices: {
                "af_heart": { name: "Heart", language: "en-us", gender: "female" },
                "am_adam": { name: "Adam", language: "en-us", gender: "male" },
                "af_sarah": { name: "Sarah", language: "en-us", gender: "female" }
            }
        });

    } catch (error) {
        console.error('Model initialization error:', error);
        self.postMessage({ error: error.message || "Unknown initialization error" });
    }
}

// Simplified speech processing for testing
function processSimpleResponse(text) {
    const responses = [
        "That's interesting! Tell me more about that.",
        "I understand what you're saying. Thanks for sharing!",
        "That's a great point. What do you think about it?",
        "I'm listening! Please continue.",
        "Thanks for talking with me. I enjoy our conversation!"
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
}

// Mock TTS function
function generateMockAudio(text) {
    // Generate a simple sine wave for testing
    const sampleRate = 24000;
    const duration = 2; // 2 seconds
    const samples = sampleRate * duration;
    const audio = new Float32Array(samples);
    
    for (let i = 0; i < samples; i++) {
        audio[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.1; // 440Hz tone
    }
    
    return audio;
}

// Main message handler
self.onmessage = async (event) => {
    const { type } = event.data;

    try {
        switch (type) {
            case "start_call":
                self.postMessage({ 
                    type: "output", 
                    text: "Hello! I'm your AI assistant. How can I help you today?", 
                    result: { audio: generateMockAudio("Hello") }
                });
                break;
                
            case "end_call":
                self.postMessage({ type: "info", message: "Call ended" });
                break;
                
            case "set_voice":
                voice = event.data.voice || "af_heart";
                self.postMessage({ type: "info", message: `Voice changed to: ${voice}` });
                break;
                
            case "audio":
                // Simulate speech detection and response
                if (Math.random() > 0.7) { // 30% chance to "detect" speech for testing
                    self.postMessage({ type: "status", status: "recording_start", message: "Listening..." });
                    
                    setTimeout(() => {
                        self.postMessage({ type: "status", status: "recording_end", message: "Processing..." });
                        
                        const response = processSimpleResponse("user input");
                        self.postMessage({ 
                            type: "output", 
                            text: response, 
                            result: { audio: generateMockAudio(response) }
                        });
                    }, 1000);
                }
                break;
                
            case "playback_ended":
                self.postMessage({ type: "info", message: "Audio playback finished" });
                break;
                
            default:
                self.postMessage({ type: "info", message: `Unknown message type: ${type}` });
        }
    } catch (error) {
        console.error('Worker message handling error:', error);
        self.postMessage({ error: error.message || "Unknown worker error" });
    }
};

// Initialize when worker starts
self.postMessage({ type: "info", message: "🚀 Starting conversation worker..." });
initializeModels();

// Voice Activity Detection
async function vad(buffer) {
    if (!silero_vad || !state) return false;
    
    try {
        const input = new Tensor("float32", buffer, [1, buffer.length]);
        const sr = new Tensor("int64", [INPUT_SAMPLE_RATE], []);
        
        const { stateN, output } = await silero_vad({ input, sr, state });
        state = stateN;
        
        const isSpeech = output.data[0];
        return isSpeech > SPEECH_THRESHOLD || (isRecording && isSpeech >= EXIT_THRESHOLD);
    } catch (error) {
        console.warn('VAD error:', error);
        return false;
    }
}

// Speech-to-Speech processing pipeline
async function speechToSpeech(buffer) {
    try {
        isPlaying = true;

        // 1. Transcribe the audio from the user
        self.postMessage({ type: "info", message: "🎤 Transcribing speech..." });
        const text = await transcriber(buffer).then(({ text }) => text.trim());
        
        if (["", "[BLANK_AUDIO]"].includes(text)) {
            self.postMessage({ type: "info", message: "No speech detected in audio" });
            isPlaying = false;
            return;
        }

        self.postMessage({ type: "info", message: `Heard: "${text}"` });
        messages.push({ role: "user", content: text });

        // 2. Generate AI response
        self.postMessage({ type: "info", message: "🧠 Generating response..." });
        let responseText = "";

        if (tts && TextSplitterStream) {
            // Use Kokoro TTS with streaming
            const splitter = new TextSplitterStream();
            const stream = tts.stream(splitter, { voice });
            
            // Start TTS streaming
            (async () => {
                try {
                    for await (const { text: chunkText, audio } of stream) {
                        self.postMessage({ 
                            type: "output", 
                            text: chunkText, 
                            result: { audio }
                        });
                    }
                } catch (ttsError) {
                    console.error('TTS streaming error:', ttsError);
                }
            })();

            // Generate response with LLM
            const inputs = tokenizer.apply_chat_template(messages, {
                add_generation_prompt: true,
                return_dict: true,
            });

            const streamer = new TextStreamer(tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true,
                callback_function: (text) => {
                    responseText += text;
                    splitter.push(text);
                },
            });

            stopping_criteria = new InterruptableStoppingCriteria();
            const { past_key_values } = await llm.generate({
                ...inputs,
                past_key_values: past_key_values_cache,
                do_sample: false,
                max_new_tokens: 256,
                streamer,
                stopping_criteria,
                return_dict_in_generate: true,
            });
            
            past_key_values_cache = past_key_values;
            splitter.close();
            
        } else {
            // Fallback: Generate simple response without TTS
            responseText = generateSimpleResponse(text);
            
            // Send as text-only output for Web Speech API handling
            self.postMessage({ 
                type: "output", 
                text: responseText, 
                result: { audio: new Float32Array(0), useWebSpeech: true }
            });
        }

        messages.push({ role: "assistant", content: responseText });
        self.postMessage({ type: "info", message: `Response: "${responseText}"` });

    } catch (error) {
        console.error('Speech-to-speech error:', error);
        self.postMessage({ type: "info", message: `Error: ${error.message}` });
        isPlaying = false;
    }
}

// Simple response generation fallback
function generateSimpleResponse(userText) {
    const text = userText.toLowerCase();
    
    if (text.includes('hello') || text.includes('hi') || text.includes('hey')) {
        return "Hello! It's great to meet you! How can I help you today?";
    } else if (text.includes('how are you')) {
        return "I'm doing wonderfully, thank you for asking! How are you doing?";
    } else if (text.includes('goodbye') || text.includes('bye')) {
        return "Goodbye! It was lovely talking with you!";
    } else if (text.includes('thank')) {
        return "You're very welcome! I'm happy to help.";
    } else if (text.includes('?')) {
        return "That's a great question! Let me think about that for you.";
    } else {
        return "That's really interesting! Tell me more about that.";
    }
}

// Audio buffer management
function resetAfterRecording(offset = 0) {
    self.postMessage({
        type: "status",
        status: "recording_end",
        message: "Processing speech...",
    });
    
    BUFFER.fill(0, offset);
    bufferPointer = offset;
    isRecording = false;
    postSpeechSamples = 0;
}

function dispatchForTranscriptionAndResetAudioBuffer(overflow) {
    // Get the speech segment with padding
    const buffer = BUFFER.slice(0, bufferPointer + SPEECH_PAD_SAMPLES);
    
    // Include previous buffers for context
    const prevLength = prevBuffers.reduce((acc, b) => acc + b.length, 0);
    const paddedBuffer = new Float32Array(prevLength + buffer.length);
    
    let offset = 0;
    for (const prev of prevBuffers) {
        paddedBuffer.set(prev, offset);
        offset += prev.length;
    }
    paddedBuffer.set(buffer, offset);
    
    // Process the audio
    speechToSpeech(paddedBuffer);
    
    // Reset buffer with overflow
    if (overflow) {
        BUFFER.set(overflow, 0);
    }
    resetAfterRecording(overflow?.length || 0);
}

// Greeting function for call start
function greet(text) {
    isPlaying = true;
    
    if (tts && TextSplitterStream) {
        const splitter = new TextSplitterStream();
        const stream = tts.stream(splitter, { voice });
        
        (async () => {
            for await (const { text: chunkText, audio } of stream) {
                self.postMessage({ type: "output", text: chunkText, result: { audio } });
            }
        })();
        
        splitter.push(text);
        splitter.close();
    } else {
        self.postMessage({ 
            type: "output", 
            text, 
            result: { audio: new Float32Array(0), useWebSpeech: true }
        });
    }
    
    messages.push({ role: "assistant", content: text });
}

// Main message handler
self.onmessage = async (event) => {
    const { type, buffer } = event.data;

    // Refuse new audio while playing back
    if (type === "audio" && isPlaying) return;

    switch (type) {
        case "start_call": {
            const voiceInfo = (tts && tts.voices[voice]) ? tts.voices[voice] : { name: "AI Assistant" };
            const greeting = `Hello! I'm ${voiceInfo.name}. How can I help you today?`;
            greet(greeting);
            return;
        }
        
        case "end_call":
            messages = messages.slice(0, 1); // Keep only system message
            past_key_values_cache = null;
            isRecording = false;
            isPlaying = false;
            bufferPointer = 0;
            postSpeechSamples = 0;
            prevBuffers = [];
            self.postMessage({ type: "info", message: "Call ended, context reset" });
            return;
            
        case "interrupt":
            stopping_criteria?.interrupt();
            isPlaying = false;
            return;
            
        case "set_voice":
            voice = event.data.voice;
            self.postMessage({ type: "info", message: `Voice changed to: ${voice}` });
            return;
            
        case "playback_ended":
            isPlaying = false;
            return;
    }

    // Handle audio buffer for VAD
    if (!buffer || !silero_vad) return;

    const wasRecording = isRecording;
    const isSpeech = await vad(buffer);

    if (!wasRecording && !isSpeech) {
        // Not recording and not speech - add to previous buffers
        if (prevBuffers.length >= MAX_NUM_PREV_BUFFERS) {
            prevBuffers.shift();
        }
        prevBuffers.push(buffer);
        return;
    }

    // Check if buffer fits in remaining space
    const remaining = BUFFER.length - bufferPointer;
    if (buffer.length >= remaining) {
        // Buffer too large - process current buffer and start new one
        BUFFER.set(buffer.subarray(0, remaining), bufferPointer);
        bufferPointer += remaining;
        
        const overflow = buffer.subarray(remaining);
        dispatchForTranscriptionAndResetAudioBuffer(overflow);
        return;
    } else {
        // Buffer fits - add to current buffer
        BUFFER.set(buffer, bufferPointer);
        bufferPointer += buffer.length;
    }

    if (isSpeech) {
        if (!isRecording) {
            // Start recording
            self.postMessage({
                type: "status",
                status: "recording_start",
                message: "Listening...",
            });
        }
        isRecording = true;
        postSpeechSamples = 0;
        return;
    }

    // Track silence after speech
    postSpeechSamples += buffer.length;

    // Check if we should end recording
    if (postSpeechSamples < MIN_SILENCE_DURATION_SAMPLES) {
        // Short pause - continue recording
        return;
    }

    if (bufferPointer < MIN_SPEECH_DURATION_SAMPLES) {
        // Speech too short - discard
        resetAfterRecording();
        return;
    }

    // End of speech detected - process the buffer
    dispatchForTranscriptionAndResetAudioBuffer();
};

// Initialize models when worker starts
self.postMessage({ type: "info", message: "🚀 Starting conversation worker..." });
initializeModels();
