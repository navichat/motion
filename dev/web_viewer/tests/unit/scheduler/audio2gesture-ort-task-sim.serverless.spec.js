// Simulate an ORT environment to exercise the non-fallback path in Audio2GestureOrtTask.

const { test, expect } = require('@playwright/test');

test('Audio2GestureOrtTask uses ORT path when window.ort and modelUrl are provided', async () => {
  const { Audio2GestureOrtTask } = require('../../../src/components/animation/timeline/tasks/Audio2GestureOrtTask.js');

  // Provide a fake ORT with an InferenceSession that returns a dummy outputs map
  const fakeWindow = {
    ort: {
      env: { wasm: { numThreads: 0 } },
      InferenceSession: class {
        static async create(url, opts) {
          // validate inputs minimally
          if (!url || !opts || !opts.executionProviders) throw new Error('bad init');
          return new this();
        }
        async run(_inputs) {
          return { y: { data: new Float32Array([0]) } };
        }
      }
    }
  };

  // Build a tiny feature provider yielding a single window
  const featureProvider = {
    _count: 0,
    async next() {
      this._count += 1;
      return { mfcc: new Float32Array(13) };
    }
  };

  const task = new Audio2GestureOrtTask({ framerate: 10, chunkMs: 100, modelUrl: '/models/fake.onnx', provider: 'wasm' });

  // Initialize against our fake window
  const ok = await task.initialize(fakeWindow);
  expect(ok).toBe(true);

  const it = task.run({ clock: { now: () => 0 }, featureProvider });
  const first = await it.next();
  expect(first.value).toBeTruthy();
  expect(first.value.frames && first.value.frames.length).toBeGreaterThan(0);
  // When ORT path is taken, metadata.model should be 'a2g_ort'
  expect(first.value.frames[0].metadata && first.value.frames[0].metadata.model).toBe('a2g_ort');
});
