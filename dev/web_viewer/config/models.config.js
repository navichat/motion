// Centralized model URL configuration for the web viewer.
// Keep paths relative to the dev server root (http://localhost:8080) when serving in tests.
// For local experiments you can update these to point to CDN or absolute URLs.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.ModelUrlConfig = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const MODELS = {
    // Example: serve models under dev/web_viewer/models/ when available
    // Provide fallbacks or leave undefined to resolve at runtime
    audio2gesture: {
      // Default is undefined because the artifact lives at repo root; tests should provide URL.
      // When hosting for the viewer, copy to dev/web_viewer/models/ and set:
      // url: '/models/audio2gesture_step_fixed.onnx'
      url: undefined
    },
  faceformer: { url: undefined }, // Viseme/face animation
  kokoro: { url: undefined },     // TTS model
  whisper: { url: undefined },    // ASR model
  sileroVad: { url: undefined },  // VAD model
  speecht5: { url: undefined },   // TTS model
  llama: { url: undefined },      // LLM endpoint/model URL
  diabloGpt: { url: undefined },  // Custom LLM endpoint/model URL
  easyvector: { endpoint: undefined }, // Vector DB endpoint
  hnsw: { url: undefined },       // ANN index/artifact
  hsnw: { url: undefined },       // alias for hnsw (user-provided spelling)
    rsmt: {
      deepPhase: { url: undefined },
      styleVAE: { url: undefined },
      transitionNet: { url: undefined }
    },
    deepmimic: {
      // Example: url: '/models/deepmimic_walk.onnx'
    }
  };

  function getModelUrl(keyPath) {
    // keyPath examples: 'audio2gesture', 'rsmt.deepPhase'
    const parts = String(keyPath).split('.');
    let node = MODELS;
    for (const p of parts) {
      node = node && node[p];
    }
    if (!node) return undefined;
    if (typeof node === 'string') return node;
    return node.url || undefined;
  }

  function setModelUrl(keyPath, url) {
    const parts = String(keyPath).split('.');
    let node = MODELS;
    for (let i = 0; i < parts.length - 1; i++) {
      const k = parts[i];
      node[k] = node[k] || {};
      node = node[k];
    }
    const leaf = parts[parts.length - 1];
    node[leaf] = node[leaf] || {};
    if (typeof node[leaf] === 'string') {
      node[leaf] = url;
    } else {
      // Prefer url property; if leaf has an 'endpoint' (e.g., easyvector), set endpoint when provided
      if (Object.prototype.hasOwnProperty.call(node[leaf], 'endpoint')) {
        node[leaf].endpoint = url;
      } else {
        node[leaf].url = url;
      }
    }
  }

  return { MODELS, getModelUrl, setModelUrl };
});
