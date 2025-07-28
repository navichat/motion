/**
 * ONNX Runtime Dependency Injector
 * Provides centralized ort dependency management for workers and worklets
 * Eliminates runtime imports in favor of dependency injection
 */

class ONNXRuntimeDependencyInjector {
    constructor() {
        this.ortInstance = null;
        this.initialized = false;
        this.workers = new Set();
    }

    /**
     * Initialize the ONNX Runtime instance once
     */
    async initialize() {
        if (this.initialized) return this.ortInstance;

        try {
            // Load ONNX Runtime on main thread
            if (typeof ort === 'undefined') {
                console.log('[ORTInjector] Loading ONNX Runtime on main thread...');
                
                // Dynamic import for modern browsers
                try {
                    const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.esm.min.js');
                    this.ortInstance = ortModule.default || ortModule;
                } catch (importError) {
                    console.warn('[ORTInjector] ESM import failed, falling back to script tag');
                    
                    // Fallback to script tag
                    await this.loadOrtViaScript();
                    this.ortInstance = window.ort;
                }
            } else {
                this.ortInstance = window.ort;
            }

            if (this.ortInstance) {
                console.log('[ORTInjector] ONNX Runtime loaded successfully');
                this.initialized = true;
                return this.ortInstance;
            } else {
                throw new Error('ONNX Runtime not available');
            }
        } catch (error) {
            console.error('[ORTInjector] Failed to initialize ONNX Runtime:', error);
            throw error;
        }
    }

    /**
     * Load ONNX Runtime via script tag (fallback method)
     */
    async loadOrtViaScript() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js';
            script.onload = () => {
                console.log('[ORTInjector] ONNX Runtime script loaded');
                resolve();
            };
            script.onerror = (error) => {
                console.error('[ORTInjector] Failed to load ONNX Runtime script:', error);
                reject(error);
            };
            document.head.appendChild(script);
        });
    }

    /**
     * Inject ONNX Runtime into a worker
     */
    async injectIntoWorker(worker) {
        if (!this.initialized) {
            await this.initialize();
        }

        if (this.ortInstance && worker && typeof worker.postMessage === 'function') {
            // Send the ort instance to the worker
            worker.postMessage({
                type: 'injectOrt',
                ortInstance: this.ortInstance
            });

            // Track the worker
            this.workers.add(worker);
            console.log('[ORTInjector] Injected ONNX Runtime into worker');
            return true;
        }
        return false;
    }

    /**
     * Create a worker with automatic ort injection
     */
    async createWorkerWithOrt(workerScript) {
        await this.initialize();
        
        const worker = new Worker(workerScript);
        await this.injectIntoWorker(worker);
        
        return worker;
    }

    /**
     * Get the current ort instance
     */
    getOrtInstance() {
        return this.ortInstance;
    }

    /**
     * Clean up workers
     */
    cleanup() {
        this.workers.forEach(worker => {
            if (worker.terminate) {
                worker.terminate();
            }
        });
        this.workers.clear();
    }
}

// Create singleton instance
const ortInjector = new ONNXRuntimeDependencyInjector();

// Export for use in modules
export { ortInjector, ONNXRuntimeDependencyInjector };

// Also make available globally
window.ortInjector = ortInjector;
