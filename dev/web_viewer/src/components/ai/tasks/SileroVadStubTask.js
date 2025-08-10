// SileroVadStubTask: emits a simple VAD decision window.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SileroVadStubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class SileroVadStubTask {
    constructor({ id = 'vad', isSpeech = true, chunkMs = 200 } = {}) {
      this.id = id;
      this.isSpeech = isSpeech;
      this.chunkMs = chunkMs;
    }
    async *run() {
      const dt = this.chunkMs / 1000;
      const t0 = 0;
      yield { t0, dt, frames: [], metadata: { model: 'silero_vad_stub', speech: this.isSpeech } };
    }
  }
  return { SileroVadStubTask };
});
