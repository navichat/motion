import { test, expect } from '@playwright/test';

const TaskMod = require('../../../src/components/animation/timeline/tasks/AudioEnergyGestureTask.js');

function sine(length, freq, sr) {
  const out = new Float32Array(length);
  for (let i=0;i<length;i++) out[i] = Math.sin(2*Math.PI*freq*i/sr);
  return out;
}

test('AudioEnergyGestureTask emits frames with energy metadata', async () => {
  const { AudioEnergyGestureTask } = TaskMod;
  const task = new AudioEnergyGestureTask({ framerate: 30, chunkMs: 200 });
  const sr = 16000; const pcm = sine(sr * 1.0, 220, sr);
  task.setPcm(pcm, sr);

  const iter = task.run({});
  const r1 = await iter.next();
  expect(r1.value.frames.length).toBeGreaterThan(0);
  expect(r1.value.frames[0].metadata.energy).toBeGreaterThanOrEqual(0);
  // stop
  task.abort.abort();
});
