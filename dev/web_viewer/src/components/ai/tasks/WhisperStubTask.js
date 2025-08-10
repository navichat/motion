// WhisperStubTask: emits a single ASR transcript segment deterministically.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.WhisperStubTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class WhisperStubTask {
    constructor({ id = 'whisper', lang = 'en', text = 'hello world', chunkMs = 500 } = {}) {
      this.id = id;
      this.lang = lang;
      this.text = text;
      this.chunkMs = chunkMs;
    }
    async *run() {
      const dt = this.chunkMs / 1000;
      const t0 = 0;
      yield { t0, dt, frames: [], metadata: { model: 'whisper_stub', lang: this.lang, text: this.text } };
    }
  }
  return { WhisperStubTask };
});
