// dev/web_viewer/modules/VRMConversationInterface.js
import { ConversationWorkerOrchestrator } from './ConversationWorkerOrchestrator.js';

export class VRMConversationInterface extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            device: 'webgpu',
            voice: 'af_heart',
            systemPrompt: 'educational',
            autoLipSync: true,
            autoGestures: true,
            autoEmotions: true,
            ...options
        };
        
        this.orchestrator = new ConversationWorkerOrchestrator(this.options);
        this.vrmCharacter = null;
        this.vrmAdapter = null;
        this.isInitialized = false;
        this.isConversing = false;
        
        this.setupOrchestratorEvents();
    }

    setupOrchestratorEvents() {
        this.orchestrator.addEventListener('ready', (event) => {
            console.log('Conversation system ready:', event.detail);
            this.isInitialized = true;
            this.dispatchEvent(new CustomEvent('ready', { detail: event.detail }));
        });

        this.orchestrator.addEventListener('status', (event) => {
            console.log('Status:', event.detail);
            this.dispatchEvent(new CustomEvent('status', { detail: event.detail }));
        });

        this.orchestrator.addEventListener('listening', (event) => {
            console.log('Listening:', event.detail.isListening);
            this.dispatchEvent(new CustomEvent('listening', { detail: event.detail }));
        });

        this.orchestrator.addEventListener('speaking', (event) => {
            console.log('Speaking:', event.detail.isSpeaking);
            this.dispatchEvent(new CustomEvent('speaking', { detail: event.detail }));
            
            // Trigger VRM animations based on speaking
            if (this.vrmCharacter && event.detail.isSpeaking) {
                this.triggerSpeakingAnimation();
            }
        });

        this.orchestrator.addEventListener('transcription', (event) => {
            console.log('User said:', event.detail.text);
            this.dispatchEvent(new CustomEvent('transcription', { detail: event.detail }));
            
            // Trigger VRM listening/acknowledgment animation
            if (this.vrmCharacter) {
                this.triggerListeningAnimation();
            }
        });

        this.orchestrator.addEventListener('response', (event) => {
            console.log('AI responded:', event.detail.text);
            this.dispatchEvent(new CustomEvent('response', { detail: event.detail }));
            
            // Analyze response for emotional content and trigger appropriate VRM expression
            if (this.vrmCharacter) {
                this.analyzeAndSetEmotion(event.detail.text);
            }
        });

        this.orchestrator.addEventListener('error', (event) => {
            console.error('Conversation error:', event.detail);
            this.dispatchEvent(new CustomEvent('error', { detail: event.detail }));
        });
    }

    // Set VRM character and adapter for animation integration
    setVRMCharacter(vrmCharacter, vrmAdapter) {
        this.vrmCharacter = vrmCharacter;
        this.vrmAdapter = vrmAdapter;
        console.log('VRM character set for conversation interface');
    }

    // Initialize the conversation system
    async initialize() {
        try {
            await this.orchestrator.initialize();
            return true;
        } catch (error) {
            console.error('Failed to initialize conversation system:', error);
            throw error;
        }
    }

    // Start voice conversation
    async startConversation() {
        if (!this.isInitialized) {
            throw new Error('Conversation system not initialized');
        }
        
        try {
            await this.orchestrator.startListening();
            this.isConversing = true;
            console.log('Voice conversation started');
        } catch (error) {
            console.error('Failed to start conversation:', error);
            throw error;
        }
    }

    // Stop voice conversation
    stopConversation() {
        this.orchestrator.stopListening();
        this.isConversing = false;
        console.log('Voice conversation stopped');
    }

    // Send text message
    sendTextMessage(text) {
        if (!this.isInitialized) {
            console.error('Conversation system not initialized');
            return;
        }
        
        this.orchestrator.sendTextMessage(text);
    }

    // Change voice
    setVoice(voice) {
        this.orchestrator.setVoice(voice);
    }

    // Change personality/system prompt
    setPersonality(personality) {
        this.orchestrator.setPersonality(personality);
    }

    // Reset conversation history
    clearHistory() {
        this.orchestrator.resetConversation();
    }

    // Test the system
    testSystem() {
        this.orchestrator.testSystem();
    }

    // VRM Animation Integration Methods
    
    triggerSpeakingAnimation() {
        if (!this.vrmCharacter || !this.options.autoGestures) return;
        
        try {
            // Set speaking expression
            if (this.vrmCharacter.expressionManager) {
                this.vrmCharacter.expressionManager.setValue('happy', 0.5);
            }
            
            // TODO: Add mouth movement animation for lip sync
            console.log('Triggered speaking animation');
        } catch (error) {
            console.log('Speaking animation failed:', error.message);
        }
    }
    
    triggerListeningAnimation() {
        if (!this.vrmCharacter || !this.options.autoEmotions) return;
        
        try {
            // Set attentive expression
            if (this.vrmCharacter.expressionManager) {
                this.vrmCharacter.expressionManager.setValue('surprised', 0.3);
            }
            
            console.log('Triggered listening animation');
        } catch (error) {
            console.log('Listening animation failed:', error.message);
        }
    }
    
    analyzeAndSetEmotion(text) {
        if (!this.vrmCharacter || !this.options.autoEmotions) return;
        
        const lowerText = text.toLowerCase();
        let emotion = 'neutral';
        let intensity = 0.7;
        
        if (lowerText.includes('happy') || lowerText.includes('great') || lowerText.includes('wonderful')) {
            emotion = 'happy';
        } else if (lowerText.includes('sad') || lowerText.includes('sorry') || lowerText.includes('unfortunately')) {
            emotion = 'sad';
        } else if (lowerText.includes('surprised') || lowerText.includes('wow') || lowerText.includes('amazing')) {
            emotion = 'surprised';
        } else if (lowerText.includes('angry') || lowerText.includes('annoyed')) {
            emotion = 'angry';
        }
        
        this.setVRMExpression(emotion, intensity);
    }
    
    setVRMExpression(expression, intensity = 1.0) {
        if (!this.vrmCharacter) return;
        
        try {
            if (this.vrmCharacter.expressionManager) {
                // Clear all expressions first
                const expressions = ['neutral', 'happy', 'sad', 'surprised', 'angry'];
                expressions.forEach(expr => {
                    try {
                        this.vrmCharacter.expressionManager.setValue(expr, 0);
                    } catch (e) {
                        // Expression might not exist, ignore
                    }
                });
                
                // Set target expression
                this.vrmCharacter.expressionManager.setValue(expression, intensity);
                console.log(`Set VRM expression: ${expression} (${intensity})`);
            }
        } catch (error) {
            console.log(`Failed to set expression ${expression}:`, error.message);
        }
    }
}
            this.sendMessageToScene('updateDebugInfo', { message: `🔊 Speaking: ${event.detail.isSpeaking ? 'ON' : 'OFF'}` });
        });

        this.orchestrator.addEventListener('error', (event) => {
            console.error('Orchestrator error:', event.detail);
            this.sendMessageToScene('updateDebugInfo', { message: `❌ Error: ${event.detail.error}` });
        });
    }

    setupSceneMessageListener() {
        window.addEventListener('message', (event) => {
            // Ensure the message is from our iframe and has the expected structure
            if (event.source === this.sceneIframe.contentWindow && event.data && event.data.type === 'vrmLoaded') {
                this.vrmCharacter = event.data.vrmCharacter;
                this.vrmAdapter = event.data.vrmAdapter;
                console.log('VRM character and adapter received from scene:', this.vrmCharacter, this.vrmAdapter);
                this.sendMessageToScene('updateDebugInfo', { message: '✅ VRM character linked to conversation interface.' });
            }
        });
    }

    // Methods to interact with the conversational system
    async initializeConversation() {
        await this.orchestrator.initialize();
    }

    async startListening() {
        await this.orchestrator.startListening();
    }

    stopListening() {
        this.orchestrator.stopListening();
    }

    sendTextMessage(text) {
        this.orchestrator.sendTextMessage(text);
    }

    setVoice(voice) {
        this.orchestrator.setVoice(voice);
    }

    resetConversation() {
        this.orchestrator.resetConversation();
    }

    getConversationHistory() {
        return this.orchestrator.getConversationHistory();
    }

    getAvailableVoices() {
        return this.orchestrator.voices;
    }

    // Methods to control VRM character in the scene
    setCharacterEmotion(emotion, options) {
        this.sendMessageToScene('setCharacterEmotion', { emotion, options });
    }

    addGestureToQueue(gesture, options) {
        this.sendMessageToScene('addGestureToQueue', { gesture, options });
    }

    setCharacterPose(pose) {
        this.sendMessageToScene('setCharacterPose', { pose });
    }

    setLightingManager(manager) {
        // This method is not directly called by the UI, but by the scene's init.
        // It's here for completeness if the UI ever needs to set it.
        this.sendMessageToScene('setLightingManager', { manager });
    }

    // Helper to send messages to the scene iframe
    sendMessageToScene(functionName, args = {}) {
        if (this.sceneWindow) {
            this.sceneWindow.postMessage({
                type: 'callFunction',
                function: functionName,
                args: args
            }, '*');
        }
    }

    dispose() {
        this.orchestrator.dispose();
    }
}
