// ModelTaskRunner: drives a chunk-yielding task into a TimelineChunkAdapter with abort/preemption support.
// Works with tasks whose run(context) returns an async iterator yielding { t0, dt, frames }.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.ModelTaskRunner = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class ModelTaskRunner {
    constructor(adapter, options = {}) {
      if (!adapter) throw new Error('ModelTaskRunner requires a TimelineChunkAdapter');
      this.adapter = adapter;
      this.defaultTrack = options.defaultTrack || 'model-track';
      this.defaultFadeMs = options.defaultFadeMs ?? 180;
      this.defaultWeight = options.defaultWeight ?? 1.0;
      this._running = new Set();
    }

    // Start driving a task; returns a handle with abort()
    start(task, { track = this.defaultTrack, fadeInMs, weight, abortSignal, onChunk } = {}, context = {}) {
      const controller = new AbortController();
      const combinedAbort = this._combineAbortSignals(abortSignal, controller.signal);
      const runPromise = this._drive(task, { track, fadeInMs, weight, abortSignal: combinedAbort, onChunk }, context)
        .finally(() => this._running.delete(runPromise));
      this._running.add(runPromise);
      return { abort: () => controller.abort(), done: () => runPromise };
    }

    async _drive(task, { track, fadeInMs, weight, abortSignal, onChunk }, context) {
      const iter = task.run({ ...context, abortSignal });
      while (!(abortSignal && abortSignal.aborted)) {
        const { value, done } = await iter.next();
        if (done) break;
        if (!value) continue;
        const opts = {
          fadeInMs: fadeInMs ?? this.defaultFadeMs,
          weight: weight ?? this.defaultWeight,
        };
        this.adapter.appendChunk(track, value, opts);
        if (typeof onChunk === 'function') onChunk(value, opts);
      }
    }

    _combineAbortSignals(signalA, signalB) {
      if (!signalA) return signalB;
      if (!signalB) return signalA;
      const ctrl = new AbortController();
      const onAbort = () => ctrl.abort();
      if (signalA.aborted || signalB.aborted) {
        ctrl.abort();
      } else {
        signalA.addEventListener('abort', onAbort, { once: true });
        signalB.addEventListener('abort', onAbort, { once: true });
      }
      return ctrl.signal;
    }
  }

  return { ModelTaskRunner };
});
