import { test, expect } from '@playwright/test';

// Serverless test via require to avoid web server
const Env = require('../../../src/audio/AudioEnvelope.js');

test('AudioEnvelope computes RMS envelope', async () => {
  const sr = 16000;
  const sec = 1.0;
  const N = sr * sec;
  const x = new Float32Array(N);
  for (let i = 0; i < N; i++) x[i] = Math.sin(2*Math.PI*440*i/sr);
  const env = Env.computeRmsEnvelope(x, sr, 50, 25);
  expect(env.length).toBeGreaterThan(10);
  // Envelope values should be positive and roughly consistent
  const avg = env.reduce((a,b)=>a+b,0)/env.length;
  expect(avg).toBeGreaterThan(0.1);
});
