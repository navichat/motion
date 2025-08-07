/**
 * ONNX Runtime Version Manager
 * Handles loading different ONNX Runtime versions for compatibility
 */

class ONNXVersionManager {
    static versions = [
        { version: '1.17.3', url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js' },
        { version: '1.18.0', url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js' },
        { version: '1.19.0', url: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js' },
    ];
    
    static currentVersion = null;
    static loadedVersions = new Set();
    
    static async loadVersion(versionInfo) {
        return new Promise((resolve, reject) => {
            if (this.loadedVersions.has(versionInfo.version)) {
                console.log(`[ONNX Version Manager] Version ${versionInfo.version} already loaded`);
                resolve();
                return;
            }
            
            console.log(`[ONNX Version Manager] Loading ONNX Runtime ${versionInfo.version}...`);
            
            // Create a unique global name for this version
            const globalName = `ort_${versionInfo.version.replace(/\./g, '_')}`;
            
            const script = document.createElement('script');
            script.src = versionInfo.url;
            script.onload = () => {
                // Store this version under a unique name
                if (typeof ort !== 'undefined') {
                    window[globalName] = ort;
                    this.loadedVersions.add(versionInfo.version);
                    console.log(`[ONNX Version Manager] ✅ Version ${versionInfo.version} loaded as ${globalName}`);
                }
                resolve();
            };
            script.onerror = () => {
                console.error(`[ONNX Version Manager] ❌ Failed to load version ${versionInfo.version}`);
                reject(new Error(`Failed to load ONNX Runtime ${versionInfo.version}`));
            };
            
            document.head.appendChild(script);
        });
    }
    
    static async tryAllVersions(modelPath, modelName) {
        console.log(`[ONNX Version Manager] Trying all ONNX Runtime versions for ${modelName}...`);
        
        for (const versionInfo of this.versions) {
            try {
                console.log(`[ONNX Version Manager] 🔄 Attempting ${versionInfo.version}...`);
                
                // Load this version if not already loaded
                await this.loadVersion(versionInfo);
                
                // Get the runtime for this version
                const ortRuntime = window[`ort_${versionInfo.version.replace(/\./g, '_')}`];
                if (!ortRuntime) {
                    console.warn(`[ONNX Version Manager] Runtime not available for ${versionInfo.version}`);
                    continue;
                }
                
                // Try to create session with this version
                const session = await ortRuntime.InferenceSession.create(modelPath, {
                    executionProviders: ['cpu'],
                    graphOptimizationLevel: 'disabled',
                    sessionOptions: {
                        enableCpuMemArena: false,
                        enableMemPattern: false,
                        logSeverityLevel: 4
                    }
                });
                
                console.log(`[ONNX Version Manager] ✅ SUCCESS with version ${versionInfo.version}!`);
                this.currentVersion = versionInfo.version;
                
                // Make this the global ort for other code to use
                window.ort = ortRuntime;
                
                return {
                    session,
                    version: versionInfo.version,
                    runtime: ortRuntime
                };
                
            } catch (error) {
                console.warn(`[ONNX Version Manager] Version ${versionInfo.version} failed: ${error.message}`);
                continue;
            }
        }
        
        throw new Error(`All ONNX Runtime versions failed for ${modelName}`);
    }
    
    static async createCompatibleSession(modelPath, modelName) {
        // First try the standard approach with current runtime
        if (typeof ort !== 'undefined') {
            try {
                console.log(`[ONNX Version Manager] Trying current ONNX Runtime...`);
                const session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ['cpu'],
                    graphOptimizationLevel: 'disabled'
                });
                console.log(`[ONNX Version Manager] ✅ Current runtime works for ${modelName}`);
                return { session, version: 'current', runtime: ort };
            } catch (error) {
                console.warn(`[ONNX Version Manager] Current runtime failed: ${error.message}`);
                
                // If it's a wire type error, try different versions
                if (error.message.includes('wire type 4') || 
                    error.message.includes('invalid wire type') ||
                    error.message.includes('protobuf')) {
                    
                    console.log(`[ONNX Version Manager] 🔧 Detected compatibility issue, trying all versions...`);
                    return await this.tryAllVersions(modelPath, modelName);
                }
                
                throw error;
            }
        } else {
            // No runtime loaded, try loading versions
            return await this.tryAllVersions(modelPath, modelName);
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXVersionManager;
}
if (typeof window !== 'undefined') {
    window.ONNXVersionManager = ONNXVersionManager;
}
