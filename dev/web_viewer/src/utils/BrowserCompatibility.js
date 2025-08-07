/**
 * Browser Compatibility Utilities
 * Provides feature detection and polyfills for better cross-browser support
 */

export class BrowserCompatibility {
    /**
     * Detect browser capabilities
     */
    static detectFeatures() {
        const features = {
            // Audio APIs
            audioContext: !!(window.AudioContext || window.webkitAudioContext),
            scriptProcessorNode: true, // Deprecated but widely supported
            audioWorklet: !!(window.AudioWorkletNode),
            mediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            webAudio: !!(window.AudioContext || window.webkitAudioContext),
            
            // Speech APIs
            speechRecognition: !!(window.SpeechRecognition || window.webkitSpeechRecognition),
            speechSynthesis: !!(window.speechSynthesis),
            
            // Modern JavaScript features
            esModules: 'noModule' in HTMLScriptElement.prototype,
            asyncAwait: true, // Modern browsers support this
            dynamicImport: (() => {
                try {
                    return typeof Function('return import')() === 'object';
                } catch (e) {
                    return false;
                }
            })(),
            
            // Browser info
            isMobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),
            isIOS: /iPad|iPhone|iPod/.test(navigator.userAgent),
            isAndroid: /Android/.test(navigator.userAgent),
            isSafari: /^((?!chrome|android).)*safari/i.test(navigator.userAgent),
            isChrome: /Chrome/.test(navigator.userAgent),
            isFirefox: /Firefox/.test(navigator.userAgent),
            
            // WebAssembly
            webAssembly: typeof WebAssembly === 'object',
            
            // Storage
            localStorage: (() => {
                try {
                    const test = '__storage_test__';
                    localStorage.setItem(test, test);
                    localStorage.removeItem(test);
                    return true;
                } catch(e) {
                    return false;
                }
            })(),
            
            // Worker support
            webWorkers: typeof Worker !== 'undefined',
            sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined'
        };
        
        return features;
    }
    
    /**
     * Create cross-browser AudioContext
     */
    static createAudioContext(options = {}) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        
        if (!AudioContextClass) {
            throw new Error('Web Audio API not supported in this browser');
        }
        
        // iOS Safari requires specific handling
        if (BrowserCompatibility.detectFeatures().isIOS && !options.sampleRate) {
            // iOS prefers 44.1kHz
            options.sampleRate = 44100;
        }
        
        return new AudioContextClass(options);
    }
    
    /**
     * Get user media with fallbacks
     */
    static async getUserMedia(constraints) {
        const features = BrowserCompatibility.detectFeatures();
        
        if (!features.mediaDevices) {
            throw new Error('getUserMedia not supported in this browser');
        }
        
        // Handle iOS specific constraints
        if (features.isIOS) {
            // iOS works better with specific audio constraints
            if (constraints.audio && typeof constraints.audio === 'object') {
                // Remove problematic constraints on iOS
                delete constraints.audio.noiseSuppression;
                delete constraints.audio.echoCancellation;
            }
        }
        
        try {
            return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (error) {
            // Fallback with simpler constraints
            if (constraints.audio && typeof constraints.audio === 'object') {
                console.warn('getUserMedia failed with advanced constraints, trying simpler constraints');
                return await navigator.mediaDevices.getUserMedia({
                    audio: true,
                    video: constraints.video || false
                });
            }
            throw error;
        }
    }
    
    /**
     * Create SpeechRecognition with vendor prefixes
     */
    static createSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            throw new Error('Speech Recognition not supported in this browser');
        }
        
        const recognition = new SpeechRecognition();
        
        // Set default properties for better compatibility
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        
        return recognition;
    }
    
    /**
     * Check if HTTPS is required for a feature
     */
    static requiresHTTPS(feature) {
        const httpsFeatures = [
            'getUserMedia',
            'devicemotion',
            'geolocation',
            'camera',
            'microphone'
        ];
        
        if (httpsFeatures.includes(feature)) {
            return location.protocol !== 'https:' && location.hostname !== 'localhost';
        }
        
        return false;
    }
    
    /**
     * Get optimal audio settings for the current browser
     */
    static getOptimalAudioSettings() {
        const features = BrowserCompatibility.detectFeatures();
        
        const settings = {
            sampleRate: 48000, // Default modern rate
            bufferSize: 4096,  // Balanced latency/performance
            channelCount: 1,   // Mono for speech
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
        };
        
        // Browser-specific optimizations
        if (features.isSafari) {
            settings.sampleRate = 44100; // Safari prefers 44.1kHz
            settings.bufferSize = 2048;  // Smaller buffer for better latency
        } else if (features.isFirefox) {
            settings.bufferSize = 8192;  // Firefox handles larger buffers better
        } else if (features.isMobile) {
            settings.bufferSize = 2048;  // Mobile prefers smaller buffers
            // Some mobile browsers have issues with advanced audio processing
            if (features.isAndroid) {
                settings.echoCancellation = false;
                settings.noiseSuppression = false;
            }
        }
        
        return settings;
    }
    
    /**
     * Handle audio context suspension/resumption for mobile
     */
    static async ensureAudioContextResumed(audioContext) {
        if (!audioContext) return;
        
        if (audioContext.state === 'suspended') {
            try {
                await audioContext.resume();
                console.log('Audio context resumed');
            } catch (error) {
                console.warn('Failed to resume audio context:', error);
            }
        }
    }
    
    /**
     * Create a compatibility report
     */
    static getCompatibilityReport() {
        const features = BrowserCompatibility.detectFeatures();
        const audioSettings = BrowserCompatibility.getOptimalAudioSettings();
        
        const report = {
            browser: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                language: navigator.language,
                mobile: features.isMobile,
                os: features.isIOS ? 'iOS' : features.isAndroid ? 'Android' : 'Desktop'
            },
            features,
            audioSettings,
            recommendations: []
        };
        
        // Add recommendations based on detected capabilities
        if (!features.audioWorklet) {
            report.recommendations.push('Consider upgrading browser for AudioWorklet support (better audio performance)');
        }
        
        if (!features.speechRecognition) {
            report.recommendations.push('Speech Recognition not available - voice input will use Whisper model');
        }
        
        if (BrowserCompatibility.requiresHTTPS('getUserMedia')) {
            report.recommendations.push('HTTPS required for microphone access in this browser');
        }
        
        if (features.isMobile) {
            report.recommendations.push('Mobile device detected - audio latency may be higher');
        }
        
        return report;
    }
    
    /**
     * Initialize browser compatibility fixes
     */
    static initialize() {
        const features = BrowserCompatibility.detectFeatures();
        
        // Add global compatibility info
        window.browserCompatibility = {
            features,
            createAudioContext: BrowserCompatibility.createAudioContext,
            getUserMedia: BrowserCompatibility.getUserMedia,
            getOptimalAudioSettings: BrowserCompatibility.getOptimalAudioSettings
        };
        
        // Add CSS classes for browser-specific styling
        document.documentElement.classList.add(
            features.isMobile ? 'mobile' : 'desktop',
            features.isIOS ? 'ios' : features.isAndroid ? 'android' : 'other-os'
        );
        
        console.log('Browser compatibility initialized:', features);
        
        return features;
    }
}

// Auto-initialize when loaded
if (typeof window !== 'undefined') {
    BrowserCompatibility.initialize();
}
