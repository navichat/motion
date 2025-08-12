// AudioEnergyGestureTask: map audio energy envelope to gesture frames for 'audio' track.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(
      require('../../../../audio/AudioEnvelope')
    );
  } else {
    root.AudioEnergyGestureTask = factory(root.AudioEnvelope || {});
  }
})(typeof self !== 'undefined' ? self : this, function (Env) {
  const computeRmsEnvelope = (Env && Env.computeRmsEnvelope) || (() => []);

  class AudioEnergyGestureTask {
    constructor({ id = 'a2g_energy', trackId = 'audio', framerate = 30, chunkMs = 200 } = {}) {
      this.id = id;
      this.trackId = trackId;
      this.framerate = framerate;
      this.chunkMs = chunkMs;
      this.running = false;
      this.abort = new AbortController();
    }

    // Provide PCM to drive gestures
    setPcm(float32Array, sampleRate) {
      this._pcm = float32Array;
      this._sr = sampleRate;
      this._env = computeRmsEnvelope(float32Array, sampleRate, 50, 33);
    }

    async *run({ endTime, quantumMs } = {}) {
      this.running = true;
      const dt = this.chunkMs / 1000;
      let t0 = 0;
      const fps = this.framerate;
      let cursor = 0; // index into envelope
      const hopMs = 33;
      while (!this.abort.signal.aborted) {
        const frames = [];
        const n = Math.max(1, Math.floor(fps * dt));
        for (let i = 0; i < n; i++) {
          const time = t0 + i / fps;
          const envVal = this._env ? (this._env[cursor] || 0) : 0;
          cursor += Math.max(1, Math.floor(hopMs / (1000 / fps)));
          // Scale energy into shoulder/hand sway amplitude
          const amp = Math.min(1, envVal * 3) * 10;
          const motionData = [];
          for (let j = 0; j < 20; j++) {
            const sway = (j % 2 === 0 ? amp : -amp);
            motionData.push([0, 0, 0, sway, 0, 0]);
          }
          frames.push({ time, motionData, metadata: { energy: envVal } });
        }
        yield { t0, dt, frames };
        t0 += dt;
        await new Promise(r => setTimeout(r));
      }
      this.running = false;
    }
  }

  return { AudioEnergyGestureTask };
});
