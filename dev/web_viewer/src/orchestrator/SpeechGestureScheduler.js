// SpeechGestureScheduler: build face/audio frames from a TTS result (stub)
// UMD export: window.SpeechGestureScheduler or module.exports
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SpeechGestureScheduler = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  function clamp01(x){ return Math.max(0, Math.min(1, x)); }
  function makeArray(n, f){ return Array.from({length:n}, (_,i)=>f(i)); }

  class SpeechGestureScheduler {
    static makeChunksFromTts(tts, opts = {}) {
      // tts: { startTime?: number, duration: number, visemes?: Array<{time:number,id:string|number}>, energy?: number[] }
      const fps = opts.fps || 30;
      const t0 = typeof tts.startTime === 'number' ? tts.startTime : 0;
      const framesCount = Math.max(1, Math.round((tts.duration || 0) * fps));

      // Face frames: encode viseme id as metadata; simple hold between keys
      const vis = Array.isArray(tts.visemes) ? tts.visemes.slice().sort((a,b)=>a.time-b.time) : [];
      let vi = 0;
      const faceFrames = makeArray(framesCount, (i) => {
        const t = i / fps;
        while (vi + 1 < vis.length && vis[vi + 1].time <= t) vi++;
        const v = vis[vi] || { id: 'rest' };
        return { motionData: [], metadata: { viseme: v.id } };
      });

      // Audio gesture frames: use energy envelope mapped to a simple shoulder swing signal in metadata
      const energy = Array.isArray(tts.energy) ? tts.energy : makeArray(framesCount, () => 0.2);
      const gestFrames = makeArray(framesCount, (i) => {
        const e = clamp01(energy[Math.min(i, energy.length - 1)] ?? 0);
        return { motionData: [], metadata: { energy: e } };
      });

      const dt = framesCount / fps;
      return {
        faceChunk: { t0, dt, frames: faceFrames },
        gestureChunk: { t0, dt, frames: gestFrames }
      };
    }
  }

  return { SpeechGestureScheduler };
});
