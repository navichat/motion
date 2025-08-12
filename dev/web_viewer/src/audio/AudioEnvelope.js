// AudioEnvelope: compute simple RMS energy envelope from PCM samples.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.AudioEnvelope = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  function computeRmsEnvelope(float32, sampleRate, windowMs = 50, hopMs = 25) {
    if (!float32 || float32.length === 0 || !sampleRate) return [];
    const win = Math.max(1, Math.floor(sampleRate * (windowMs / 1000)));
    const hop = Math.max(1, Math.floor(sampleRate * (hopMs / 1000)));
    const out = [];
    for (let start = 0; start < float32.length; start += hop) {
      const end = Math.min(float32.length, start + win);
      let sumSq = 0;
      for (let i = start; i < end; i++) { const v = float32[i]; sumSq += v * v; }
      const meanSq = sumSq / (end - start);
      out.push(Math.sqrt(meanSq));
      if (end === float32.length) break;
    }
    return out;
  }

  return { computeRmsEnvelope };
});
