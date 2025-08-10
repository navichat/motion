// Lightweight runtime configuration helpers for ML model initialization.
// Pure JS utilities; do not import onnxruntime-web here to keep serverless tests lean.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.ModelRuntimeConfig = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  function detectWebGPU(env) {
    const g = env && env.navigator ? env.navigator.gpu : (typeof navigator !== 'undefined' && navigator.gpu);
    return !!g;
  }

  function selectOrtProvider(options = {}, env) {
    const preferWebGPU = options.preferWebGPU !== false; // default true
    const hasWebGPU = detectWebGPU(env);
    if (preferWebGPU && hasWebGPU) return 'webgpu';
    return 'wasm';
  }

  function buildOrtSessionOptions(provider) {
    const common = { executionProviders: [provider] };
    if (provider === 'webgpu') {
      return {
        ...common,
        // Example WebGPU options; adapt when wiring ORT Web
        extra: {
          enableGraphCapture: true,
          preferredOutputBuffer: 'gpu'
        }
      };
    }
    // wasm
    return {
      ...common,
      wasm: { numThreads: 1, proxy: true }
    };
  }

  // Path helpers
  function getModelFsPath(name) {
    // Conservative default: repo-root artifacts checked via Node fs in tests
    const map = {
      audio2gesture: 'audio2gesture_step_fixed.onnx'
    };
    const f = map[name] || name;
    if (typeof process !== 'undefined' && process.cwd) {
      return require('path').join(process.cwd(), f);
    }
    return f;
  }

  return {
    detectWebGPU,
    selectOrtProvider,
    buildOrtSessionOptions,
    getModelFsPath,
  };
});
