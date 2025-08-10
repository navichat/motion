// Audio2GestureOrtTask: Streams gesture BVH chunks using ONNX Runtime Web when available.
// Falls back to a deterministic stub when ORT or model URL is not configured.
// Does not import onnxruntime-web directly; expects window.ort injected in web-backed runs.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.Audio2GestureOrtTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class Audio2GestureOrtTask {
    constructor({ framerate = 30, chunkMs = 200, modelUrl, provider = 'wasm' } = {}) {
      this.framerate = framerate;
      this.chunkMs = chunkMs;
      // Resolve model URL from ModelUrlConfig when not provided explicitly
      this.modelUrl = modelUrl || (function () {
        try {
          // Prefer global first (browser tests), then CommonJS require fallback
          const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
          if (g && g.ModelUrlConfig && typeof g.ModelUrlConfig.getModelUrl === 'function') {
            return g.ModelUrlConfig.getModelUrl('audio2gesture');
          }
          if (typeof require !== 'undefined') {
            // tasks -> timeline -> animation -> components -> src -> dev/web_viewer -> config
            const cfg = require('../../../../../config/models.config.js');
            if (cfg && typeof cfg.getModelUrl === 'function') {
              return cfg.getModelUrl('audio2gesture');
            }
          }
        } catch (_) {}
        return undefined;
      })();
      this.provider = provider;
      this.session = null;
      this.initialized = false;
    }

    async initialize(env = (typeof window !== 'undefined' ? window : {})) {
      // Only initialize if ORT and a model URL are present
      if (!env || !env.ort || !this.modelUrl) {
        this.initialized = false;
        return false;
      }
      try {
        const { InferenceSession, env: ortEnv } = env.ort;
        if (this.provider === 'wasm' && ortEnv && ortEnv.wasm) {
          ortEnv.wasm.numThreads = 1;
        }
        this.session = await InferenceSession.create(this.modelUrl, { executionProviders: [this.provider] });
        this.initialized = true;
        return true;
      } catch (e) {
        // Fail soft; remain in fallback mode
        this.session = null;
        this.initialized = false;
        return false;
      }
    }

    // Placeholder: convert audio features to model inputs
    _buildInputs(_featureWindow) {
      // Implement once feature extractor is wired; return a dummy map for now
      return {};
    }

    // Placeholder: convert model outputs to BVH frames
    _outputsToFrames(_outputs, t0, dt) {
      // TODO: Map tensors to VRM bone rotations
      const frames = [];
      const n = Math.max(1, Math.floor(this.framerate * dt));
      for (let i = 0; i < n; i++) frames.push({ time: t0 + i / this.framerate, motionData: [], metadata: { model: 'a2g_ort' } });
      return frames;
    }

    // Fallback deterministic frames (similar to stub)
    _fallbackFrames(t0, dt) {
      const frames = [];
      const n = Math.max(1, Math.floor(this.framerate * dt));
      for (let i = 0; i < n; i++) {
        const time = t0 + i / this.framerate;
        const motionData = [];
        for (let j = 0; j < 20; j++) {
          const sway = Math.sin(time * 2 * Math.PI) * 5;
          motionData.push([0, 0, 0, j % 2 === 0 ? sway : -sway, 0, 0]);
        }
        frames.push({ time, motionData, metadata: { model: 'a2g_fallback' } });
      }
      return frames;
    }

    async *run(context = {}) {
      const { clock = { now: () => 0 }, abortSignal, featureProvider } = context;
      const dt = this.chunkMs / 1000;
      let t0 = clock.now ? clock.now() : 0;

  // Try to initialize if not done yet, ignore errors
      if (!this.initialized) {
        try { await this.initialize(typeof window !== 'undefined' ? window : {}); } catch {}
      }

      while (!abortSignal || !abortSignal.aborted) {
        if (this.session && featureProvider && featureProvider.next) {
          // Real path when features and ORT are available
          const featureWindow = await featureProvider.next();
          const inputs = this._buildInputs(featureWindow);
          let outputs;
          try {
            outputs = await this.session.run(inputs);
          } catch {
            outputs = null;
          }
          const frames = outputs ? this._outputsToFrames(outputs, t0, dt) : this._fallbackFrames(t0, dt);
          yield { t0, dt, frames };
        } else {
          // Fallback path without ORT or features
          const frames = this._fallbackFrames(t0, dt);
          yield { t0, dt, frames };
        }
        t0 += dt;
        await new Promise(r => setTimeout(r));
      }
    }
  }

  return { Audio2GestureOrtTask };
});
