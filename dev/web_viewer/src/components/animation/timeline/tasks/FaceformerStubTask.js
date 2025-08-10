// FaceformerStubTask: Generates head/viseme-like blendshapes deterministically.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.FaceformerStubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class FaceformerStubTask {
    constructor({ id = 'faceformer', trackId = 'face', framerate = 30, durationMs = 300 } = {}) {
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
        // Minimal face bones: head, jaw, leftEye, rightEye
        const motionData = [];
        const headPitch = Math.sin(time * 2 * Math.PI * 0.5) * 2;
        const jawOpen = (Math.sin(time * 2 * Math.PI) * 0.5 + 0.5) * 10; // 0..10 deg
        // bones with x,y,z,rx,ry,rz
        motionData.push([0,0,0, headPitch, 0, 0]); // head
        motionData.push([0,0,0, jawOpen, 0, 0]);   // jaw
        motionData.push([0,0,0, 0, 0, 0]);         // leftEye
        motionData.push([0,0,0, 0, 0, 0]);         // rightEye
        frames.push({ time, motionData, metadata: { model: 'faceformer_stub' } });
      }
      return frames;
    }

    async *run() {
      const dt = this.durationMs / 1000;
      const t0 = 0;
      yield { t0, dt, frames: this._generateFrames(t0, dt) };
    }
  }

  return { FaceformerStubTask };
});
