/**
 * VRMConversationInterface_v2.js - CACHE-BYPASS VERSION
 * Interface for connecting VRM characters with conversation AI
 * Integrates with the ConversationWorkerOrchestrator for voice and text chat
 */

import { ConversationWorkerOrchestrator } from './ConversationWorkerOrchestrator.js';

/**
 * Main interface class for VRM conversation functionality
 */
export default class VRMConversationInterface extends EventTarget {
    constructor(options = {}) {
        super();
        
        this.options = {
            voiceEnabled: true,
            modelName: 'microsoft/DialoGPT-medium',
            temperature: 0.7,
            maxTokens: 100,
            ...options
        };
        
        this.isInitialized = false;
        this.isConversing = false;
        
        // VRM Character references
        this.vrmCharacter = null;
        this.vrmAdapter = null;
        
        // Conversation orchestrator
        console.log('🔧 Creating ConversationWorkerOrchestrator v2...');
        try {
            this.orchestrator = new ConversationWorkerOrchestrator(this.options);
            console.log('✅ ConversationWorkerOrchestrator v2 created successfully');
        } catch (error) {
            console.error('❌ Failed to create ConversationWorkerOrchestrator v2:', error);
            this.orchestrator = null;
        }
        
        console.log('🎭 VRM Conversation Interface v2 created - FRESH BUILD');
    }

    /**
     * Set up event listeners for the orchestrator
     */
    setupOrchestratorEvents() {
        console.log('🔧 Setting up orchestrator events v2 - FRESH VERSION');
        if (!this.orchestrator) {
            console.warn('No orchestrator available in setupOrchestratorEvents v2');
            return;
        }
        
        // Ready event
        this.orchestrator.addEventListener('ready', (event) => {
            console.log('🎤 Conversation orchestrator ready v2:', event.detail);
            this.dispatchEvent(new CustomEvent('ready', { detail: event.detail }));
        });

        // User message events
        this.orchestrator.addEventListener('userMessage', (event) => {
            console.log('👤 User said v2:', event.detail.text);
            this.dispatchEvent(new CustomEvent('userMessage', { detail: event.detail }));
            
            // Trigger VRM listening/acknowledgment animation
            if (this.vrmCharacter) {
                this.triggerListeningAnimation();
            }
        });

        // Assistant response events
        this.orchestrator.addEventListener('assistantMessage', (event) => {
            console.log('🤖 AI responded v2:', event.detail.text);
            this.dispatchEvent(new CustomEvent('assistantMessage', { detail: event.detail }));
            
            // Analyze response for emotional content and trigger appropriate VRM expression
            if (this.vrmCharacter) {
                this.analyzeAndSetEmotion(event.detail.text);
                this.triggerSpeakingAnimation();
            }
        });

        // Speaking state events
        this.orchestrator.addEventListener('speaking', (event) => {
            console.log('🗣️ Speaking state v2:', event.detail.isSpeaking);
            this.dispatchEvent(new CustomEvent('speaking', { detail: event.detail }));
            
            if (this.vrmCharacter) {
                if (event.detail.isSpeaking) {
                    this.triggerSpeakingAnimation();
                } else {
                    this.resetToNeutralExpression();
                }
            }
        });

        // Listening state events
        this.orchestrator.addEventListener('listening', (event) => {
            console.log('🎤 Listening state v2:', event.detail.isListening);
            this.dispatchEvent(new CustomEvent('listening', { detail: event.detail }));
            
            if (this.vrmCharacter && event.detail.isListening) {
                this.triggerListeningAnimation();
            }
        });

        // Processing state events
        this.orchestrator.addEventListener('processing', (event) => {
            console.log('🧠 Processing state v2:', event.detail.processing);
            this.dispatchEvent(new CustomEvent('processing', { detail: event.detail }));
        });

        // Error events
        this.orchestrator.addEventListener('error', (event) => {
            console.log('❌ Conversation error v2:', event.detail.error);
            this.dispatchEvent(new CustomEvent('error', { detail: event.detail }));
        });

        // Conversation lifecycle events
        this.orchestrator.addEventListener('conversationStarted', (event) => {
            this.isConversing = true;
            this.dispatchEvent(new CustomEvent('conversationStarted', { detail: event.detail }));
        });

        this.orchestrator.addEventListener('conversationStopped', (event) => {
            this.isConversing = false;
            this.dispatchEvent(new CustomEvent('conversationStopped', { detail: event.detail }));
        });
    }

    /**
     * Initialize the conversation system
     */
    async initialize() {
        try {
            console.log('🚀 Initializing VRM Conversation Interface v2...');
            
            // Set up event listeners for orchestrator first
            this.setupOrchestratorEvents();
            
            // Initialize the orchestrator (don't throw error if it fails, allow fallback)
            try {
                await this.orchestrator.initialize();
                console.log('✅ Full conversation system v2 initialized');
            } catch (initError) {
                console.warn('⚠️ Full system v2 initialization failed, using fallback mode:', initError);
                // Still mark as initialized to allow basic functionality
            }
            
            this.isInitialized = true;
            
            this.dispatchEvent(new CustomEvent('ready', {
                detail: { 
                    status: 'initialized',
                    hasVoice: this.options.voiceEnabled,
                    modelName: this.options.modelName,
                    mode: this.orchestrator.isInitialized ? 'full' : 'fallback'
                }
            }));
            
            console.log('✅ VRM Conversation Interface v2 ready');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize VRM Conversation Interface v2:', error);
            
            // Still allow fallback mode
            this.isInitialized = true;
            
            this.dispatchEvent(new CustomEvent('ready', {
                detail: { 
                    status: 'fallback',
                    hasVoice: false,
                    modelName: 'fallback',
                    mode: 'fallback'
                }
            }));
            
            console.log('✅ VRM Conversation Interface v2 ready (fallback mode)');
            return true;
        }
    }

    /**
     * Start voice conversation
     */
    async startConversation() {
        if (!this.isInitialized) {
            throw new Error('Conversation system not initialized');
        }
        
        try {
            this.isConversing = true;
            
            if (this.orchestrator && this.orchestrator.startConversation) {
                await this.orchestrator.startConversation();
            }
            
            console.log('Voice conversation started v2');
            
            this.dispatchEvent(new CustomEvent('conversationStarted', {
                detail: { timestamp: Date.now() }
            }));
            
            return true;
            
        } catch (error) {
            this.isConversing = false;
            console.error('Failed to start conversation v2:', error);
            throw error;
        }
    }

    /**
     * Stop voice conversation
     */
    async stopConversation() {
        try {
            this.isConversing = false;
            
            if (this.orchestrator && this.orchestrator.stopConversation) {
                await this.orchestrator.stopConversation();
            }
            
            console.log('Voice conversation stopped v2');
            
            this.dispatchEvent(new CustomEvent('conversationStopped', {
                detail: { timestamp: Date.now() }
            }));
            
            return true;
            
        } catch (error) {
            console.error('Failed to stop conversation v2:', error);
            throw error;
        }
    }

    /**
     * Send text message and get response
     */
    async sendTextMessage(text) {
        if (!this.isInitialized) {
            throw new Error('Conversation system not initialized');
        }
        
        try {
            let response;
            
            if (this.orchestrator && this.orchestrator.sendTextMessage) {
                response = await this.orchestrator.sendTextMessage(text);
            } else {
                // Fallback response
                response = {
                    text: "I received your message but the full conversation system isn't available. This is a fallback response.",
                    timestamp: Date.now()
                };
            }
            
            console.log('Text message sent and response received v2');
            return response;
            
        } catch (error) {
            console.error('Failed to send text message v2:', error);
            throw error;
        }
    }

    /**
     * Connect VRM character to the conversation interface
     */
    setVRMCharacter(vrmCharacter, vrmAdapter = null) {
        this.vrmCharacter = vrmCharacter;
        this.vrmAdapter = vrmAdapter;
        
        if (this.orchestrator && this.orchestrator.setVRMCharacter) {
            this.orchestrator.setVRMCharacter(vrmCharacter, vrmAdapter);
        }
        
        console.log('🎭 VRM character connected to conversation interface v2');
    }

    /**
     * Set voice configuration
     */
    setVoice(voiceName) {
        if (this.orchestrator && this.orchestrator.setVoice) {
            this.orchestrator.setVoice(voiceName);
        }
    }

    /**
     * Set personality/system prompt
     */
    setPersonality(personality) {
        if (this.orchestrator && this.orchestrator.setPersonality) {
            this.orchestrator.setPersonality(personality);
        }
    }

    /**
     * Trigger VRM speaking animation
     */
    triggerSpeakingAnimation() {
        if (this.vrmCharacter && this.vrmCharacter.expressionManager) {
            try {
                const expressions = this.vrmCharacter.expressionManager;
                
                // Animate mouth movements for speaking
                expressions.setValue('aa', 0.4);
                expressions.setValue('happy', 0.3);
                
                console.log('🗣️ VRM speaking animation triggered v2');
            } catch (error) {
                console.warn('Failed to trigger speaking animation v2:', error);
            }
        }
    }

    /**
     * Trigger VRM listening animation
     */
    triggerListeningAnimation() {
        if (this.vrmCharacter && this.vrmCharacter.expressionManager) {
            try {
                const expressions = this.vrmCharacter.expressionManager;
                
                // Subtle attention expression
                expressions.setValue('surprised', 0.2);
                expressions.setValue('happy', 0.1);
                
                console.log('👂 VRM listening animation triggered v2');
            } catch (error) {
                console.warn('Failed to trigger listening animation v2:', error);
            }
        }
    }

    /**
     * Analyze response text and set appropriate VRM emotion
     */
    analyzeAndSetEmotion(responseText) {
        if (!this.vrmCharacter || !this.vrmCharacter.expressionManager) {
            return;
        }
        
        try {
            const expressions = this.vrmCharacter.expressionManager;
            const text = responseText.toLowerCase();
            
            // Reset all expressions first
            ['happy', 'sad', 'angry', 'surprised', 'neutral'].forEach(expr => {
                try {
                    expressions.setValue(expr, 0);
                } catch (e) {
                    // Ignore if expression doesn't exist
                }
            });
            
            // Analyze text content for emotional cues
            if (text.includes('happy') || text.includes('great') || text.includes('wonderful') || text.includes('excellent')) {
                this.setVRMExpression('happy', 0.8);
            } else if (text.includes('sad') || text.includes('sorry') || text.includes('unfortunately')) {
                this.setVRMExpression('sad', 0.6);
            } else if (text.includes('surprised') || text.includes('wow') || text.includes('amazing')) {
                this.setVRMExpression('surprised', 0.7);
            } else if (text.includes('angry') || text.includes('frustrated')) {
                this.setVRMExpression('angry', 0.6);
            } else {
                this.setVRMExpression('happy', 0.3); // Default mild positive expression
            }
            
            console.log('😊 VRM emotion set based on response content v2');
            
        } catch (error) {
            console.warn('Failed to analyze and set emotion v2:', error);
        }
    }

    /**
     * Set specific VRM expression
     */
    setVRMExpression(expression, intensity = 1.0) {
        if (this.vrmCharacter && this.vrmCharacter.expressionManager) {
            try {
                const expressions = this.vrmCharacter.expressionManager;
                
                if (expressions.setValue) {
                    expressions.setValue(expression, intensity);
                    console.log(`😊 VRM expression set v2: ${expression} (${intensity})`);
                } else {
                    // Fallback for different expression manager APIs
                    const fallbackExpression = expression === 'happy' ? 'joy' : expression;
                    if (expressions[fallbackExpression]) {
                        expressions[fallbackExpression] = intensity;
                        console.log(`😊 Using fallback expression v2: ${fallbackExpression}`);
                    }
                }
            } catch (error) {
                console.warn('Failed to set VRM expression v2:', error);
            }
        }
    }

    /**
     * Reset VRM to neutral expression
     */
    resetToNeutralExpression() {
        if (this.vrmCharacter && this.vrmCharacter.expressionManager) {
            try {
                const expressions = this.vrmCharacter.expressionManager;
                
                // Reset all expressions to 0
                ['happy', 'sad', 'angry', 'surprised', 'aa', 'ih', 'ou', 'ee', 'oh'].forEach(expr => {
                    try {
                        expressions.setValue(expr, 0);
                    } catch (e) {
                        // Ignore if expression doesn't exist
                    }
                });
                
                // Set neutral expression
                expressions.setValue('neutral', 1.0);
                
                console.log('😐 VRM expression reset to neutral v2');
            } catch (error) {
                console.warn('Failed to reset VRM expression v2:', error);
            }
        }
    }

    /**
     * Reset conversation state
     */
    resetConversation() {
        this.isConversing = false;
        
        if (this.orchestrator && this.orchestrator.resetConversation) {
            this.orchestrator.resetConversation();
        }
        
        // Reset VRM to neutral
        this.resetToNeutralExpression();
    }

    /**
     * Dispose of resources
     */
    dispose() {
        this.isConversing = false;
        
        if (this.orchestrator && this.orchestrator.dispose) {
            this.orchestrator.dispose();
        }
        
        this.vrmCharacter = null;
        this.vrmAdapter = null;
        this.isInitialized = false;
        this.isConversing = false;
        
        console.log('🧹 VRM Conversation Interface v2 disposed');
    }
}
