const { test, expect } = require('@playwright/test');

test('RSMT and DeepMimic stub tasks append chunks via TimelineChunkAdapter', async () => {
  const { BVHTimeline } = require('../../../src/models/bvh/BVHTimeline.js');
  const { TimelineChunkAdapter } = require('../../../src/components/animation/timeline/TimelineChunkAdapter.js');
  const { RsmtStubTask } = require('../../../src/components/animation/timeline/tasks/RsmtStubTask.js');
  const { DeepMimicStubTask } = require('../../../src/components/animation/timeline/tasks/DeepMimicStubTask.js');

  const timeline = new BVHTimeline({ framerate: 30 });
  const adapter = new TimelineChunkAdapter(timeline);

  async function drain(task) {
    for await (const chunk of task.run()) {
      adapter.appendChunk(task.trackId, chunk, { fadeInMs: 50 });
    }
  }

  const rsmt = new RsmtStubTask({ trackId: 'transition', durationMs: 400 });
  const dm = new DeepMimicStubTask({ trackId: 'locomotion', durationMs: 600 });

  await Promise.all([drain(rsmt), drain(dm)]);

  const hasTransition = !!timeline.tracks.transition && timeline.tracks.transition.clips.length > 0;
  const hasLocomotion = !!timeline.tracks.locomotion && timeline.tracks.locomotion.clips.length > 0;

  expect(hasTransition).toBeTruthy();
  expect(hasLocomotion).toBeTruthy();
});
