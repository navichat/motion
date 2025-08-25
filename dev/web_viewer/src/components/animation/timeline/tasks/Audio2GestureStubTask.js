// Audio2GestureStubTask: Emits deterministic BVH chunks from a fake audio clock.
// This is a stub for tests until the real model pipeline is wired.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.Audio2GestureStubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class Audio2GestureStubTask {
    constructor({ id = 'a2g_stub', trackId = 'audio', framerate = 30, chunkMs = 200 } = {}) {
      this.id = id;
      this.trackId = trackId;
      this.framerate = framerate;
      this.chunkMs = chunkMs;
      this.running = false;
    }

    async *run(context = {}) {
      const { clock = { now: () => 0 }, abortSignal } = context;
      this.running = true;
      const dt = this.chunkMs / 1000;
      let t0 = clock.now ? clock.now() : 0;
      while (!abortSignal || !abortSignal.aborted) {
        const frames = [];
        const n = Math.max(1, Math.floor(this.framerate * dt));
        for (let i = 0; i < n; i++) {
          const time = t0 + i / this.framerate;
          // Simple sway on shoulders/hands; positions 0, rotations vary
          const motionData = [];
          for (let j = 0; j < 20; j++) {
            const sway = Math.sin(time * 2 * Math.PI) * 5; // degrees
            motionData.push([0, 0, 0, j % 2 === 0 ? sway : -sway, 0, 0]);
          }
          frames.push({ time, motionData, metadata: { model: 'a2g_stub' } });
        }
        yield { t0, dt, frames };
        t0 += dt;
        await new Promise(r => setTimeout(r));
      }
      this.running = false;
    }
  }

  return { Audio2GestureStubTask };
});
