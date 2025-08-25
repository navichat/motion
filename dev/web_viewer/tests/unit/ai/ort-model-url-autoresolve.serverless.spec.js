const { test, expect } = require('@playwright/test');

test('Audio2GestureOrtTask auto-resolves model URL from ModelUrlConfig', async () => {
  const cfg = require('../../../config/models.config.js');
  const { Audio2GestureOrtTask } = require('../../../src/components/animation/timeline/tasks/Audio2GestureOrtTask.js');

  // Ensure config returns a URL for audio2gesture
  cfg.setModelUrl('audio2gesture', '/models/a2g.onnx');

  const task = new Audio2GestureOrtTask({ framerate: 10, chunkMs: 100, provider: 'wasm' });
  // No explicit modelUrl; should pick up from config
  expect(task.modelUrl).toBe('/models/a2g.onnx');
});
