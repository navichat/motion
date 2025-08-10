// RsmtStubTask: Generates a short transition chunk blending from pose A to pose B.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.RsmtStubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class RsmtStubTask {
    constructor({ id = 'rsmt', trackId = 'transition', framerate = 30, durationMs = 400 } = {}) {
      this.id = id;
      this.trackId = trackId;
      this.framerate = framerate;
      this.durationMs = durationMs;
    }

    _lerp(a, b, t) { return a + (b - a) * t; }

    _generateFrames(t0, dt) {
      const frames = [];
      const n = Math.max(1, Math.floor(this.framerate * dt));
      for (let i = 0; i < n; i++) {
        const time = t0 + i / this.framerate;
        const alpha = i / Math.max(1, n - 1);
        const motionData = [];
        // Simple 10-joint rig: x,y,z,rx,ry,rz; we interpolate rx between -10 -> +10 deg
        for (let j = 0; j < 10; j++) {
          const rx = this._lerp(-10, 10, alpha) * (j % 2 === 0 ? 1 : -1);
          motionData.push([0, 0, 0, rx, 0, 0]);
        }
        frames.push({ time, motionData, metadata: { model: 'rsmt_stub', alpha } });
      }
      return frames;
    }

    async *run() {
      const dt = this.durationMs / 1000;
      const t0 = 0;
      yield { t0, dt, frames: this._generateFrames(t0, dt) };
    }
  }

  return { RsmtStubTask };
});
