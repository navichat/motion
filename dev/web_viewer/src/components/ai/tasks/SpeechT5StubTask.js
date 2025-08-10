// SpeechT5StubTask: emits a TTS audio buffer placeholder.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SpeechT5StubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class SpeechT5StubTask {
    constructor({ id = 'speecht5', text = 'hello', sampleRate = 16000, durationSec = 0.5 } = {}) {
      this.id = id; this.text = text; this.sampleRate = sampleRate; this.durationSec = durationSec;
    }
    async *run() {
      const length = Math.floor(this.sampleRate * this.durationSec);
      const pcm = new Float32Array(length);
      for (let i = 0; i < length; i++) pcm[i] = Math.sin(i / this.sampleRate * 2 * Math.PI * 220) * 0.1;
      yield { t0: 0, dt: this.durationSec, frames: [], metadata: { model: 'speecht5_stub', sampleRate: this.sampleRate, samples: pcm.length } };
    }
  }
  return { SpeechT5StubTask };
});
