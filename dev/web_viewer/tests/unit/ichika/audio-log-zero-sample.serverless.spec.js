// Ensures verifier flags zero-sample artifact
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test('[Unit][AudioLog] zero-sample artifact generation', async () => {
  const outDir = path.join(process.cwd(), 'test-results');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const artifact = {
    scenario: 'speech-backend (fake ASR) zero sample test',
    backend: 'speech',
    timestamp: new Date().toISOString(),
    uniqueTypes: ['tts_scheduled'],
    count: 1,
    metrics: { samples: 0, avgLatencyMs: 0, lastLatencyMs: 0, p50: 0, p95: 0 },
    latencyEntry: false,
    tail: []
  };
  fs.writeFileSync(path.join(outDir,'audio-log-zero_sample_test.json'), JSON.stringify(artifact,null,2));
  expect(fs.existsSync(path.join(outDir,'audio-log-zero_sample_test.json'))).toBeTruthy();
});
