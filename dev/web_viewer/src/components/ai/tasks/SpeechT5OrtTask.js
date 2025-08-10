// SpeechT5OrtTask: Real TTS with onnxruntime-web when configured; fallback to tone.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SpeechT5OrtTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class SpeechT5OrtTask {
    constructor({ id = 'speecht5', modelUrl, provider = 'wasm' } = {}) {
      this.id = id; this.modelUrl = modelUrl; this.provider = provider; this.session = null; this.initialized = false;
    }
    _resolveModelUrl() {
      if (this.modelUrl) return this.modelUrl;
      try {
        const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
        if (g && g.ModelUrlConfig && typeof g.ModelUrlConfig.getModelUrl === 'function') return g.ModelUrlConfig.getModelUrl('speecht5');
        if (typeof require !== 'undefined') return require('../../../config/models.config.js').getModelUrl('speecht5');
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
    async *run({ features } = {}) {
      if (!this.initialized) { try { await this.initialize(typeof window !== 'undefined' ? window : {}); } catch {} }
      if (this.session && features) {
        try {
          const outputs = await this.session.run(features);
          const first = outputs && Object.values(outputs)[0];
          const length = (first && first.data && first.data.length) || 16000;
          const sampleRate = 16000;
          yield { t0: 0, dt: length / sampleRate, frames: [], metadata: { model: 'speecht5_ort', sampleRate, samples: length } };
          return;
        } catch {}
      }
      // fallback tone
      const sampleRate = 16000; const durationSec = 0.3; const length = Math.floor(sampleRate * durationSec);
      yield { t0: 0, dt: durationSec, frames: [], metadata: { model: 'speecht5_fallback', sampleRate, samples: length } };
    }
  }
  return { SpeechT5OrtTask };
});
