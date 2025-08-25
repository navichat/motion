/**
 * LlamaModule - Text generation using LLM models
 * Supports model loading/unloading for memory management
 * Uses dependency injection for GPU context and resource management
 */

import { BaseModel } from '../utils/ResourceManager.js';

export class LlamaModule extends BaseModel {
    constructor(options = {}) {
        super(options);
        
        this.options = {
            modelName: options.llamaModel || 'Xenova/TinyLlama-1.1B-Chat-v0.4',
            maxTokens: options.maxTokens || 150,
            temperature: options.temperature || 0.8,
            topP: options.topP || 0.9,
            device: options.device || 'wasm',
            quantized: options.quantized || true,
            systemPrompt: options.systemPrompt || "You are a helpful AI assistant. Keep responses concise and engaging. Include emotion markers like [happy], [thoughtful], [excited] in your responses.",
            ...options
        };

        this.model = null;
        this.tokenizer = null;
    }

    /**
     * Load the LLM model using dependency injection
     */
    async load() {
        if (this.isModelLoaded) {
            return true;
        }

        if (this.loadingPromise) {
            return this.loadingPromise;
        }

        this.loadingPromise = this._loadModel();
        return this.loadingPromise;
    }

    async _loadModel() {
        try {
            this.emit('loading', { module: 'llama', status: 'starting' });

            this.emit('loading', { 
                module: 'llama', 
                status: 'downloading', 
                model: this.options.modelName 
            });

            // Use shared pipeline from resource manager
            this.model = await this.getPipeline('text-generation', this.options.modelName, {
                device: this.options.device,
                dtype: this.options.quantized ? 'q4' : 'fp16',
                model_file_name: this.options.quantized ? 'model_q4.onnx' : 'model.onnx'
            });

            this.isModelLoaded = true;
            this.loadingPromise = null;
            
            this.emit('loaded', { 
                module: 'llama', 
                model: this.options.modelName,
                device: this.options.device 
            });

            return true;
        } catch (error) {
            this.isModelLoaded = false;
            this.loadingPromise = null;
            this.emit('error', { module: 'llama', error });
            throw error;
        }
    }

    /**
     * Unload the model to free memory
     */
    async unload() {
        if (!this.isModelLoaded) {
            return;
        }

        try {
            // Clear model references
            this.model = null;
            this.tokenizer = null;
            
            // Call base class unload
            await super.unload();

            // Force garbage collection if available
            if (global.gc) {
                global.gc();
            }

            this.emit('unloaded', { module: 'llama' });
        } catch (error) {
            this.emit('error', { module: 'llama', error });
        }
    }

    /**
     * Generate response from conversation history
     */
    async generateResponse(conversationHistory) {
        if (!this.isModelLoaded) {
            throw new Error('LLM model not loaded');
        }

        try {
            // Build prompt from conversation history
            const prompt = this._buildPrompt(conversationHistory);
            
            // Generate response
            const result = await this.model(prompt, {
                max_new_tokens: this.options.maxTokens,
                temperature: this.options.temperature,
                top_p: this.options.topP,
                do_sample: true,
                pad_token_id: 50256 // Common pad token
            });

            // Extract and clean response
            const rawResponse = result[0].generated_text;
            const response = this._cleanResponse(rawResponse.replace(prompt, ''));

            return response;
        } catch (error) {
            this.emit('error', { module: 'llama', operation: 'generate', error });
            
            // Return fallback response
            return this._getFallbackResponse(conversationHistory);
        }
    }

    /**
     * Build prompt from conversation history
     */
    _buildPrompt(conversationHistory) {
        let prompt = `${this.options.systemPrompt}\n\n`;
        
        // Add conversation context (last few messages)
        const recentHistory = conversationHistory.slice(-6); // Keep last 6 messages
        
        for (const message of recentHistory) {
            const role = message.role === 'user' ? 'Human' : 'Assistant';
            prompt += `${role}: ${message.content}\n`;
        }
        
        prompt += 'Assistant:';
        return prompt;
    }

    /**
     * Clean up model response
     */
    _cleanResponse(response) {
        // Remove any trailing prompts or incomplete sentences
        const lines = response.split('\n');
        let cleanedResponse = '';
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed && !trimmed.startsWith('Human:') && !trimmed.startsWith('Assistant:')) {
                cleanedResponse += trimmed + ' ';
            }
        }
        
        cleanedResponse = cleanedResponse.trim();
        
        // Ensure response ends properly
        if (cleanedResponse && !cleanedResponse.match(/[.!?]$/)) {
            cleanedResponse += '.';
        }
        
        // Limit response length
        if (cleanedResponse.length > 300) {
            cleanedResponse = cleanedResponse.substring(0, 297) + '...';
        }
        
        return cleanedResponse || "I'm here to help! What would you like to talk about? [friendly]";
    }

    /**
     * Get fallback response when model fails
     */
    _getFallbackResponse(conversationHistory) {
        const fallbackResponses = [
            "That's interesting! Tell me more about that. [curious]",
            "I see what you mean. What do you think about it? [thoughtful]",
            "Thanks for sharing that with me! [appreciative]",
            "That's a great point! Could you elaborate? [engaged]",
            "I'm listening! Please continue. [attentive]",
            "How fascinating! What made you think of that? [interested]"
        ];
        
        const lastMessage = conversationHistory[conversationHistory.length - 1]?.content?.toLowerCase() || '';
        
        // Simple keyword-based responses
        if (lastMessage.includes('hello') || lastMessage.includes('hi')) {
            return "Hello! It's great to chat with you! How are you doing today? [friendly]";
        } else if (lastMessage.includes('help')) {
            return "I'm here to help! What can I assist you with? [helpful]";
        } else if (lastMessage.includes('thank')) {
            return "You're very welcome! I'm glad I could help. [warm]";
        }
        
        // Random fallback
        return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    }

    /**
     * Get model info
     */
    getModelInfo() {
        return {
            loaded: this.isModelLoaded,
            model: this.options.modelName,
            maxTokens: this.options.maxTokens,
            temperature: this.options.temperature,
            device: this.options.device,
            quantized: this.options.quantized
        };
    }

    /**
     * Update generation parameters
     */
    updateParameters(params) {
        this.options = { ...this.options, ...params };
    }
}
