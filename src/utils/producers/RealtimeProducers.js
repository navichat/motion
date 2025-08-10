// Simple real-time producers for visemes and gestures used in tests and demos.
// Exports CommonJS and attaches to window for browser consumption.

(function (root, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory();
  } else {
    const mod = factory();
    root.RealtimeProducers = mod;
  }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  class TextToVisemeProducer {
    constructor(text) { this.text = text || ''; }
    async start(targetAdapter, trackId = 'face', opts = {}) {
      const visemes = opts.visemes || ['sil','A','E','I','O','U'];
      const framesPerChunk = Math.max(1, opts.framesPerChunk || 6); // ~200ms at 30fps
      const fadeInMs = opts.fadeInMs ?? 40;
      const delayMs = opts.delayMs ?? 50;
      let tCursor = opts.t0 || 0;
      for (let i = 0; i < this.text.length; i++) {
        const v = visemes[(i % (visemes.length - 1)) + 1];
        const frames = Array.from({ length: framesPerChunk }, (_, j) => ({ time: j/30, motionData: [], metadata: { viseme: v } }));
        targetAdapter.appendChunk(trackId, { t0: tCursor, dt: framesPerChunk/30, frames }, { fadeInMs });
        tCursor += framesPerChunk/30;
        if (delayMs > 0) await sleep(delayMs);
      }
    }
  }

  class Audio2GestureProducerStub {
    constructor(durationSec = 0.8) { this.durationSec = Math.max(0, durationSec); }
    async start(targetAdapter, trackId = 'audio', opts = {}) {
      const chunkDt = opts.chunkDt || 0.2; // seconds per chunk
      const framesPerChunk = Math.max(1, opts.framesPerChunk || 6);
      const fadeInMs = opts.fadeInMs ?? 30;
      const delayMs = opts.delayMs ?? 60;
      const steps = Math.max(1, Math.ceil(this.durationSec / chunkDt));
      let tCursor = opts.t0 || 0;
      for (let c = 0; c < steps; c++) {
        const frames = [];
        for (let i = 0; i < framesPerChunk; i++) {
          const t = i/30;
          const energy = 0.3 + 0.7 * Math.abs(Math.sin((c*framesPerChunk+i) * 0.5));
          const md = new Array(20);
          const yaw = 8 + 4 * Math.sin((c*framesPerChunk+i) * 0.3);
          md[5] = [0,0,0, 0, yaw, 0];
          frames.push({ time: t, motionData: md, metadata: { energy, boneMask: ['bone_5'] } });
        }
        targetAdapter.appendChunk(trackId, { t0: tCursor, dt: chunkDt, frames }, { blendMode: 'replace', fadeInMs });
        tCursor += chunkDt;
        if (delayMs > 0) await sleep(delayMs);
      }
    }
  }

  return { TextToVisemeProducer, Audio2GestureProducerStub };
});
