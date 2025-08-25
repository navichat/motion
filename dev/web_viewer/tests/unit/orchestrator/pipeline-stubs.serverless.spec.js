const { test, expect } = require('@playwright/test');

test('Orchestrator pipeline with Audio2Gesture, RSMT, DeepMimic stubs produces clips on tracks', async () => {
  const { IchikaOrchestrator } = require('../../../src/orchestrator/IchikaOrchestrator.js');
  const { BVHTimeline } = require('../../../src/models/bvh/BVHTimeline.js');
  const { Audio2GestureStubTask } = require('../../../src/components/animation/timeline/tasks/Audio2GestureStubTask.js');
  const { RsmtStubTask } = require('../../../src/components/animation/timeline/tasks/RsmtStubTask.js');
  const { DeepMimicStubTask } = require('../../../src/components/animation/timeline/tasks/DeepMimicStubTask.js');

  const timeline = new BVHTimeline({ framerate: 30 });
    const orch = new IchikaOrchestrator({ timeline, quantumMs: 20 });


    // Prepare stub tasks
    const a2g = new Audio2GestureStubTask({ id: 'a2g', trackId: 'audio', framerate: 30, chunkMs: 200 });
    const rsmt = new RsmtStubTask({ id: 'rsmt', trackId: 'transition', durationMs: 300 });
    const dm = new DeepMimicStubTask({ id: 'dm', trackId: 'locomotion', durationMs: 400 });

    // Deterministically feed one chunk from each task into the orchestrator's adapter
    async function feedOne(task) {
      const it = task.run();
      const res = await it.next();
      if (!res.done) {
        orch.adapter.appendChunk(task.trackId, res.value, { fadeInMs: 50 });
      }
    }

    await Promise.all([feedOne(a2g), feedOne(rsmt), feedOne(dm)]);

  const audioClips = (timeline.tracks.audio && timeline.tracks.audio.clips.length) || 0;
  const transitionClips = (timeline.tracks.transition && timeline.tracks.transition.clips.length) || 0;
  const locomotionClips = (timeline.tracks.locomotion && timeline.tracks.locomotion.clips.length) || 0;

  expect(audioClips).toBeGreaterThan(0);
  expect(transitionClips + locomotionClips).toBeGreaterThan(0);
});
