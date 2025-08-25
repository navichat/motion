/**
 * TaskFactory: wraps model runners to standardized chunked tasks.
 * Each task yields FramesChunk: { t0, dt, frames, meta }
 */
class TaskFactory {
  static createBVHPlaybackTask({ id, trackId, priority = 3, frames, startAt = 0, chunkSec = 0.2 }) {
    const abort = new AbortController();
    async function* run(ctx) {
      const fps = 30;
      const framesPerChunk = Math.max(1, Math.floor(fps * chunkSec));
      let idx = 0;
      while (idx < frames.length) {
        if (abort.signal.aborted) throw new Error('aborted');
        const slice = frames.slice(idx, idx + framesPerChunk);
        const t0 = startAt + idx / fps;
        const dt = slice.length / fps;
        yield { t0, dt, frames: slice, meta: { origin: 'bvh' } };
        idx += framesPerChunk;
        await Promise.resolve();
      }
      return { done: true };
    }
    return { id, kind: 'bvh', trackId, priority, abort, run };
  }

  // Placeholders for model-backed generators; wire to existing modules later.
  static createFaceFormerTask({ id, trackId, priority = 0, backend = 'auto', generator, startAt = 0, chunkSec = 0.2 }) {
    const abort = new AbortController();
    async function* run(ctx) {
      for await (const chunk of generator({ startAt, chunkSec, abort })) {
        if (abort.signal.aborted) throw new Error('aborted');
        yield { ...chunk, meta: { ...(chunk.meta||{}), origin: 'faceformer' } };
      }
      return { done: true };
    }
    return { id, kind: 'faceformer', trackId, priority, backend, abort, run };
  }

  static createAudio2GestureTask({ id, trackId, priority = 1, backend = 'auto', generator, startAt = 0, chunkSec = 0.2 }) {
    const abort = new AbortController();
    async function* run(ctx) {
      for await (const chunk of generator({ startAt, chunkSec, abort })) {
        if (abort.signal.aborted) throw new Error('aborted');
        yield { ...chunk, meta: { ...(chunk.meta||{}), origin: 'a2g' } };
      }
      return { done: true };
    }
    return { id, kind: 'a2g', trackId, priority, backend, abort, run };
  }

  static createRSMTTask({ id, trackId, priority = 2, backend = 'auto', generator, startAt = 0, chunkSec = 0.2 }) {
    const abort = new AbortController();
    async function* run(ctx) {
      for await (const chunk of generator({ startAt, chunkSec, abort })) {
        if (abort.signal.aborted) throw new Error('aborted');
        yield { ...chunk, meta: { ...(chunk.meta||{}), origin: 'rsmt' } };
      }
      return { done: true };
    }
    return { id, kind: 'rsmt', trackId, priority, backend, abort, run };
  }

  static createDeepMimicTask({ id, trackId, priority = 3, backend = 'auto', generator, startAt = 0, chunkSec = 0.2 }) {
    const abort = new AbortController();
    async function* run(ctx) {
      for await (const chunk of generator({ startAt, chunkSec, abort })) {
        if (abort.signal.aborted) throw new Error('aborted');
        yield { ...chunk, meta: { ...(chunk.meta||{}), origin: 'deepmimic' } };
      }
      return { done: true };
    }
    return { id, kind: 'deepmimic', trackId, priority, backend, abort, run };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TaskFactory };
} else {
  window.TaskFactory = TaskFactory;
}
