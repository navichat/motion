/**
 * VRM Conversation Interface
 * Bridges VRM character animations with the conversation system
 */

class VRMConversationInterface {
    constructor(vrm, orchestrator) {
        this.vrm = vrm;
        this.orchestrator = orchestrator;
        this.isInitialized = false;
        this.currentEmotion = 'neutral';
        this.isSpeaking = false;
        this.animationQueue = [];
        
        // Animation timing
        this.speakingAnimationInterval = null;
        this.emotionTransitionDuration = 1000; // ms
        
        this.init();
    }
    
    init() {
        try {
            this.setupOrchestratorEvents();
            this.isInitialized = true;
            console.log('VRMConversationInterface initialized successfully');
        } catch (error) {
            console.error('Error initializing VRMConversationInterface:', error);
        }
    }
    
    setupOrchestratorEvents() {
        if (!this.orchestrator) {
            console.warn('No orchestrator provided to VRMConversationInterface');
            return;
        }
        
        // Listen for conversation events
        this.orchestrator.addEventListener('userSpeaking', (event) => {
            this.setListeningState(true);
        });
        
        this.orchestrator.addEventListener('userStoppedSpeaking', (event) => {
            this.setListeningState(false);
        });
        
        this.orchestrator.addEventListener('aiResponseStart', (event) => {
            this.startSpeakingAnimation();
        });
        
        this.orchestrator.addEventListener('aiResponseEnd', (event) => {
            this.stopSpeakingAnimation();
        });
        
        this.orchestrator.addEventListener('conversationMessage', (event) => {
            if (event.detail && event.detail.text) {
                this.analyzeAndSetEmotion(event.detail.text);
            }
        });
        
        this.orchestrator.addEventListener('error', (event) => {
            this.setEmotion('concerned');
        });
    }
    
    setListeningState(isListening) {
        if (isListening) {
            this.setEmotion('attentive');
            this.triggerGesture('nod');
        } else {
            this.setEmotion('neutral');
        }
    }
    
    startSpeakingAnimation() {
        if (this.isSpeaking) return;
        
        this.isSpeaking = true;
        this.setEmotion('speaking');
        
        // Start mouth movement animation
        this.speakingAnimationInterval = setInterval(() => {
            this.animateMouthMovement();
        }, 100); // Update every 100ms
        
        // Trigger speaking gestures occasionally
        this.triggerSpeakingGestures();
    }
    
    stopSpeakingAnimation() {
        this.isSpeaking = false;
        
        if (this.speakingAnimationInterval) {
            clearInterval(this.speakingAnimationInterval);
            this.speakingAnimationInterval = null;
        }
        
        // Return to neutral expression
        this.setEmotion('neutral');
        this.setMouthShape('neutral');
    }
    
    animateMouthMovement() {
        if (!this.isSpeaking || !this.vrm) return;
        
        // Simulate mouth movements during speech
        const mouthShapes = ['a', 'i', 'u', 'e', 'o', 'neutral'];
        const randomShape = mouthShapes[Math.floor(Math.random() * mouthShapes.length)];
        this.setMouthShape(randomShape);
    }
    
    triggerSpeakingGestures() {
        if (!this.isSpeaking) return;
        
        const gestures = ['gesture', 'point', 'nod'];
        const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
        
        setTimeout(() => {
            if (this.isSpeaking) {
                this.triggerGesture(randomGesture);
                // Schedule next gesture
                setTimeout(() => this.triggerSpeakingGestures(), 2000 + Math.random() * 3000);
            }
        }, 1000 + Math.random() * 2000);
    }
    
    analyzeAndSetEmotion(text) {
        if (!text) return;
        
        const lowerText = text.toLowerCase();
        let emotion = 'neutral';
        
        // Simple emotion detection based on keywords
        if (lowerText.includes('happy') || lowerText.includes('joy') || lowerText.includes('great') || lowerText.includes('wonderful')) {
            emotion = 'happy';
        } else if (lowerText.includes('sad') || lowerText.includes('sorry') || lowerText.includes('unfortunately')) {
            emotion = 'sad';
        } else if (lowerText.includes('angry') || lowerText.includes('annoyed') || lowerText.includes('frustrated')) {
            emotion = 'angry';
        } else if (lowerText.includes('surprised') || lowerText.includes('wow') || lowerText.includes('amazing')) {
            emotion = 'surprised';
        } else if (lowerText.includes('confused') || lowerText.includes('don\'t understand') || lowerText.includes('unclear')) {
            emotion = 'confused';
        } else if (lowerText.includes('thinking') || lowerText.includes('considering') || lowerText.includes('hmm')) {
            emotion = 'thinking';
        }
        
        this.setEmotion(emotion);
    }
    
    setEmotion(emotion) {
        if (!this.vrm || this.currentEmotion === emotion) return;
        
        console.log(`Setting VRM emotion to: ${emotion}`);
        this.currentEmotion = emotion;
        
        try {
            // Map emotions to VRM expressions
            const expressionMap = {
                'neutral': { name: 'neutral', weight: 1.0 },
                'happy': { name: 'happy', weight: 1.0 },
                'sad': { name: 'sad', weight: 1.0 },
                'angry': { name: 'angry', weight: 1.0 },
                'surprised': { name: 'surprised', weight: 1.0 },
                'confused': { name: 'confused', weight: 0.8 },
                'thinking': { name: 'neutral', weight: 0.6 },
                'speaking': { name: 'happy', weight: 0.3 },
                'attentive': { name: 'neutral', weight: 1.0 },
                'concerned': { name: 'sad', weight: 0.5 }
            };
            
            const expression = expressionMap[emotion] || expressionMap['neutral'];
            this.applyExpression(expression.name, expression.weight);
            
        } catch (error) {
            console.error('Error setting VRM emotion:', error);
        }
    }
    
    setMouthShape(shape) {
        if (!this.vrm) return;
        
        try {
            const mouthMap = {
                'a': 'aa',
                'i': 'ih',
                'u': 'ou',
                'e': 'ee',
                'o': 'oh',
                'neutral': 'neutral'
            };
            
            const mouthExpression = mouthMap[shape] || 'neutral';
            this.applyExpression(mouthExpression, 0.8);
            
        } catch (error) {
            console.error('Error setting mouth shape:', error);
        }
    }
    
    triggerGesture(gestureType) {
        if (!this.vrm) return;
        
        console.log(`Triggering VRM gesture: ${gestureType}`);
        
        try {
            // Simple gesture animations
            switch (gestureType) {
                case 'nod':
                    this.animateNod();
                    break;
                case 'gesture':
                    this.animateGesture();
                    break;
                case 'point':
                    this.animatePoint();
                    break;
                default:
                    console.log('Unknown gesture type:', gestureType);
            }
        } catch (error) {
            console.error('Error triggering gesture:', error);
        }
    }
    
    applyExpression(expressionName, weight = 1.0) {
        if (!this.vrm || !this.vrm.expressionManager) return;
        
        try {
            // Reset all expressions first
            this.vrm.expressionManager.setValue('neutral', 0);
            this.vrm.expressionManager.setValue('happy', 0);
            this.vrm.expressionManager.setValue('sad', 0);
            this.vrm.expressionManager.setValue('angry', 0);
            this.vrm.expressionManager.setValue('surprised', 0);
            
            // Apply the target expression
            this.vrm.expressionManager.setValue(expressionName, weight);
            this.vrm.expressionManager.update();
            
        } catch (error) {
            console.error('Error applying VRM expression:', error);
        }
    }
    
    animateNod() {
        // Simple head nod animation
        if (!this.vrm || !this.vrm.humanoid) return;
        
        try {
            const head = this.vrm.humanoid.getNormalizedBoneNode('head');
            if (head) {
                const originalRotation = head.rotation.x;
                
                // Nod down
                head.rotation.x = originalRotation - 0.2;
                
                setTimeout(() => {
                    // Nod back up
                    head.rotation.x = originalRotation + 0.1;
                    
                    setTimeout(() => {
                        // Return to original
                        head.rotation.x = originalRotation;
                    }, 200);
                }, 200);
            }
        } catch (error) {
            console.error('Error animating nod:', error);
        }
    }
    
    animateGesture() {
        // Simple hand gesture
        if (!this.vrm || !this.vrm.humanoid) return;
        
        try {
            const rightArm = this.vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
            if (rightArm) {
                const originalRotation = rightArm.rotation.z;
                
                // Raise arm
                rightArm.rotation.z = originalRotation - 0.5;
                
                setTimeout(() => {
                    // Lower arm
                    rightArm.rotation.z = originalRotation;
                }, 1000);
            }
        } catch (error) {
            console.error('Error animating gesture:', error);
        }
    }
    
    animatePoint() {
        // Simple pointing gesture
        if (!this.vrm || !this.vrm.humanoid) return;
        
        try {
            const rightArm = this.vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
            const rightForearm = this.vrm.humanoid.getNormalizedBoneNode('rightLowerArm');
            
            if (rightArm && rightForearm) {
                const originalArmRotation = rightArm.rotation.z;
                const originalForearmRotation = rightForearm.rotation.y;
                
                // Point gesture
                rightArm.rotation.z = originalArmRotation - 1.0;
                rightForearm.rotation.y = originalForearmRotation + 0.5;
                
                setTimeout(() => {
                    // Return to original
                    rightArm.rotation.z = originalArmRotation;
                    rightForearm.rotation.y = originalForearmRotation;
                }, 1500);
            }
        } catch (error) {
            console.error('Error animating point:', error);
        }
    }
    
    // Manual control methods
    speakText(text) {
        this.analyzeAndSetEmotion(text);
        this.startSpeakingAnimation();
        
        // Stop speaking after estimated duration
        const estimatedDuration = text.length * 100; // Rough estimate
        setTimeout(() => {
            this.stopSpeakingAnimation();
        }, Math.max(estimatedDuration, 2000));
    }
    
    setPersonality(personality) {
        console.log(`Setting VRM personality to: ${personality}`);
        
        // Adjust default expressions based on personality
        const personalityMap = {
            'friendly': 'happy',
            'professional': 'neutral',
            'energetic': 'happy',
            'calm': 'neutral',
            'mysterious': 'thinking'
        };
        
        const defaultEmotion = personalityMap[personality] || 'neutral';
        this.setEmotion(defaultEmotion);
    }
    
    // Cleanup
    destroy() {
        if (this.speakingAnimationInterval) {
            clearInterval(this.speakingAnimationInterval);
        }
        
        if (this.orchestrator) {
            // Remove event listeners if needed
            // Note: EventTarget doesn't have removeAllListeners, so we'd need to track them
        }
        
        this.isInitialized = false;
        console.log('VRMConversationInterface destroyed');
    }
}

// Export for ES6 modules
export { VRMConversationInterface };

// Also export as default for compatibility
export default VRMConversationInterface;

// Export for CommonJS modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VRMConversationInterface;
}

// Make available globally
if (typeof window !== 'undefined') {
    window.VRMConversationInterface = VRMConversationInterface;
}
