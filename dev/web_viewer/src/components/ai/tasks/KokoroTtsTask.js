// KokoroTtsTask: Real TTS using kokoro-js when available, fallback to sine if not.
// Requires network or locally cached weights when using kokoro-js.
// UMD export: module.exports and window.KokoroTtsTask

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KokoroTtsTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class KokoroTtsTask {
    constructor({ id = 'kokoro', text = 'hello world', voice = 'bm_daniel', speed = 1.0 } = {}) {
      this.id = id;
      this.text = text;
      this.voice = voice;
      this.speed = speed;
      this.impl = null;
      this.ready = false;
    }

    async initialize(env = (typeof window !== 'undefined' ? window : {})) {
      // Prefer global kokoro-js if already bundled; else try require('kokoro-js') in Node
      try {
        if (env && env.kokoro && env.kokoro.KokoroTTS) {
          this.impl = env.kokoro.KokoroTTS;
          this.ready = true;
          return true;
        }
        // Also support legacy global export window.KokoroTTS (pre-refactor path)
        if (env && env.KokoroTTS) {
          this.impl = env.KokoroTTS;
          this.ready = true;
          return true;
        }
      } catch {}
      try {
        // Attempt dynamic import of kokoro-js when available as dependency
        // eslint-disable-next-line global-require
        const lib = (typeof require !== 'undefined') ? require('kokoro-js') : null;
        if (lib && lib.KokoroTTS) {
          this.impl = lib.KokoroTTS;
          this.ready = true;
          return true;
        }
      } catch {}
      this.ready = false;
      return false;
    }

    async *run({ text, voice, speed } = {}) {
      const phrase = text || this.text;
      const v = voice || this.voice;
      const rate = speed || this.speed;

      if (!this.ready) {
        try { await this.initialize(typeof window !== 'undefined' ? window : {}); } catch {}
      }

      if (this.impl) {
        try {
          // Use default public model id; can be overridden by ModelUrlConfig.kokoro url
          const cfg = (typeof window !== 'undefined' && window.ModelUrlConfig) ? window.ModelUrlConfig : (typeof module !== 'undefined' ? module.exports : undefined);
          const modelId = cfg && typeof cfg.getModelUrl === 'function' && cfg.getModelUrl('kokoro') ? cfg.getModelUrl('kokoro') : 'onnx-community/Kokoro-82M-v1.0-ONNX';
          const TTS = this.impl;
          const tts = await TTS.from_pretrained(modelId);
          // Stream one segment for simplicity
          const result = await tts.generate(phrase, { voice: v, speed: rate }).next();
          const audio = result && result.value ? result.value.audio : null;
          if (audio && audio.sampleRate && audio.getChannelData) {
            const samples = audio.getChannelData(0);
            yield { t0: 0, dt: samples.length / audio.sampleRate, frames: [], metadata: { model: 'kokoro', sampleRate: audio.sampleRate, samples: samples.length, voice: v } };
            return;
          }
        } catch (e) {
          // fall through to sine fallback
        }
      }

      // Fallback: deterministic sine tone
      const sampleRate = 16000;
      const durationSec = 0.5;
      const length = Math.floor(sampleRate * durationSec);
      const pcm = new Float32Array(length);
      for (let i = 0; i < length; i++) pcm[i] = Math.sin(i / sampleRate * 2 * Math.PI * 220) * 0.05;
      yield { t0: 0, dt: durationSec, frames: [], metadata: { model: 'kokoro_fallback', sampleRate, samples: pcm.length, voice: v } };
    }
  }

  return { KokoroTtsTask };
});
