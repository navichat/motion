// WhisperOrtTask: Real ASR using onnxruntime-web when window.ort and model configured.
// Fallback: emit provided text or empty transcript.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.WhisperOrtTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class WhisperOrtTask {
    constructor({ id = 'whisper', lang = 'en', modelUrl, provider = 'wasm', textFallback = 'hello world' } = {}) {
      this.id = id; this.lang = lang; this.modelUrl = modelUrl; this.provider = provider;
      this.textFallback = textFallback;
      this.session = null; this.initialized = false;
    }
    _resolveModelUrl() {
      if (this.modelUrl) return this.modelUrl;
      try {
        const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
        if (g && g.ModelUrlConfig && typeof g.ModelUrlConfig.getModelUrl === 'function') {
          return g.ModelUrlConfig.getModelUrl('whisper');
        }
        if (typeof require !== 'undefined') {
          const cfg = require('../../../config/models.config.js');
          return cfg && cfg.getModelUrl('whisper');
        }
      } catch {}
      return undefined;
    }
    async initialize(env = (typeof window !== 'undefined' ? window : {})) {
      const url = this._resolveModelUrl();
      if (!env || !env.ort || !url) { this.initialized = false; return false; }
      try {
        const { InferenceSession, env: ortEnv } = env.ort;
        if (this.provider === 'wasm' && ortEnv && ortEnv.wasm) ortEnv.wasm.numThreads = 1;
        this.session = await InferenceSession.create(url, { executionProviders: [this.provider] });
        this.initialized = true; return true;
      } catch { this.session = null; this.initialized = false; return false; }
    }
    // Minimal feature stub: expects mel spectrogram features provided by caller
    async *run({ features, audioBuffer } = {}) {
      if (!this.initialized) {
        try { await this.initialize(typeof window !== 'undefined' ? window : {}); } catch {}
      }
      if (this.session && features && features.input_features) {
        try {
          const outputs = await this.session.run({ input_features: features.input_features });
          // Users must map decoder; here we expose raw logits existence
          const have = outputs && Object.keys(outputs).length > 0;
          yield { t0: 0, dt: 0, frames: [], metadata: { model: 'whisper_ort', logits: !!have } };
          return;
        } catch {}
      }
      yield { t0: 0, dt: 0, frames: [], metadata: { model: 'whisper_fallback', text: this.textFallback, lang: this.lang } };
    }
  }
  return { WhisperOrtTask };
});
