// FaceformerOrtTask: Real FaceFormer viseme/head outputs when window.ort and model configured.
// Fallback: deterministic viseme-like blendshapes.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.FaceformerOrtTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class FaceformerOrtTask {
    constructor({ id = 'faceformer', trackId = 'face', framerate = 30, durationMs = 300, modelUrl, provider = 'wasm' } = {}) {
      this.id = id; this.trackId = trackId; this.framerate = framerate; this.durationMs = durationMs;
      this.modelUrl = modelUrl; this.provider = provider;
      this.session = null; this.initialized = false;
    }
    _resolveModelUrl() {
      if (this.modelUrl) return this.modelUrl;
      try {
        const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
        if (g && g.ModelUrlConfig && typeof g.ModelUrlConfig.getModelUrl === 'function') return g.ModelUrlConfig.getModelUrl('faceformer');
        if (typeof require !== 'undefined') return require('../../../../config/models.config.js').getModelUrl('faceformer');
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
    _fallbackFrames(t0, dt) {
      const frames = []; const n = Math.max(1, Math.floor(this.framerate * dt));
      for (let i = 0; i < n; i++) {
        const time = t0 + i / this.framerate;
        const mouthOpen = (Math.sin(time * 6.28 * 2) + 1) / 2;
        const smile = (Math.sin(time * 6.28) + 1) / 2;
        const motionData = { mouthOpen, smile };
        frames.push({ time, motionData, metadata: { model: 'faceformer_fallback' } });
      }
      return frames;
    }
    async *run(context = {}) {
      const dt = this.durationMs / 1000; let t0 = 0;
      if (!this.initialized) { try { await this.initialize(typeof window !== 'undefined' ? window : {}); } catch {} }
      if (!this.session) { yield { t0, dt, frames: this._fallbackFrames(t0, dt) }; return; }
      try {
        // Expect caller to provide embeddings/features; otherwise just yield fallback.
        const inputs = context && context.features ? context.features : null;
        if (!inputs) { yield { t0, dt, frames: this._fallbackFrames(t0, dt) }; return; }
        const outputs = await this.session.run(inputs);
        // Minimal mapping: turn any numeric tensor outputs into viseme scalars [0,1]
        const frames = []; const n = Math.max(1, Math.floor(this.framerate * dt));
        const vals = Object.values(outputs)[0];
        for (let i = 0; i < n; i++) {
          const time = t0 + i / this.framerate;
          const v = vals && vals.data && vals.data.length ? Math.abs(vals.data[i % vals.data.length]) : 0.5;
          frames.push({ time, motionData: { mouthOpen: Math.min(1, v), smile: Math.min(1, v*0.5) }, metadata: { model: 'faceformer_ort' } });
        }
        yield { t0, dt, frames };
      } catch {
        yield { t0, dt, frames: this._fallbackFrames(t0, dt) };
      }
    }
  }
  return { FaceformerOrtTask };
});
