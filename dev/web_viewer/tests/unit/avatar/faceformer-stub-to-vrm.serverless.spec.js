const { test, expect } = require('@playwright/test');

test('Faceformer stub yields face frames and can be appended to face track', async () => {
  const { BVHTimeline } = require('../../../src/models/bvh/BVHTimeline.js');
  const { TimelineChunkAdapter } = require('../../../src/components/animation/timeline/TimelineChunkAdapter.js');
  const { FaceformerStubTask } = require('../../../src/components/animation/timeline/tasks/FaceformerStubTask.js');

  const timeline = new BVHTimeline({ framerate: 30 });
  const adapter = new TimelineChunkAdapter(timeline);

  const task = new FaceformerStubTask({ trackId: 'face', durationMs: 300 });
  for await (const chunk of task.run()) {
    adapter.appendChunk('face', chunk, { fadeInMs: 30 });
  }

  expect(!!timeline.tracks.face && timeline.tracks.face.clips.length > 0).toBeTruthy();
});
