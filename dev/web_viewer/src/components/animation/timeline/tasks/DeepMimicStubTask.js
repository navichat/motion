// DeepMimicStubTask: Generates locomotion-like chunks (e.g., walking) deterministically.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.DeepMimicStubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class DeepMimicStubTask {
    constructor({ id = 'deepmimic', trackId = 'locomotion', framerate = 30, durationMs = 600 } = {}) {
      this.id = id;
      this.trackId = trackId;
      this.framerate = framerate;
      this.durationMs = durationMs;
    }

    _generateFrames(t0, dt) {
      const frames = [];
      const n = Math.max(1, Math.floor(this.framerate * dt));
      for (let i = 0; i < n; i++) {
        const time = t0 + i / this.framerate;
        const motionData = [];
        // 12 joints with a simple alternating gait pattern
        for (let j = 0; j < 12; j++) {
          const phase = (j % 2 === 0 ? 1 : -1);
          const ry = Math.sin(time * 2 * Math.PI * 1.2 + j * 0.2) * 3 * phase;
          motionData.push([0, 0, 0, 0, ry, 0]);
        }
        frames.push({ time, motionData, metadata: { model: 'deepmimic_stub' } });
      }
      return frames;
    }

    async *run() {
      const dt = this.durationMs / 1000;
      const t0 = 0;
      yield { t0, dt, frames: this._generateFrames(t0, dt) };
    }
  }

  return { DeepMimicStubTask };
});
