import { test, expect } from '@playwright/test';

// Keep this file fast: HTTP-only checks, no navigation, short per-file timeout.
// Validates that ORT smoke fixtures are present and wired with expected scripts.

test.setTimeout(60_000);

const fixtures = [
  {
    path: '/tests/unit/ai/fixtures/ort_whisper_smoke.html',
    expectScripts: [
      '/config/models.config.js',
      '/src/testing/real_inference_bootstrap.js',
      '/src/components/ai/tasks/WhisperOrtTask.js',
    ],
    title: /Whisper ORT Smoke/i,
  },
  {
    path: '/tests/unit/ai/fixtures/ort_faceformer_smoke.html',
    expectScripts: [
      '/config/models.config.js',
      '/src/testing/real_inference_bootstrap.js',
      '/src/components/animation/timeline/tasks/FaceformerOrtTask.js',
    ],
    title: /Faceformer ORT Smoke/i,
  },
  {
    path: '/tests/unit/ai/fixtures/ort_silero_vad_smoke.html',
    expectScripts: [
      '/config/models.config.js',
      '/src/testing/real_inference_bootstrap.js',
      '/src/components/ai/tasks/SileroVadOrtTask.js',
    ],
    title: /Silero VAD ORT Smoke/i,
  },
  {
    path: '/tests/unit/ai/fixtures/ort_speecht5_smoke.html',
    expectScripts: [
      '/config/models.config.js',
      '/src/testing/real_inference_bootstrap.js',
      '/src/components/ai/tasks/SpeechT5OrtTask.js',
    ],
    title: /SpeechT5 ORT Smoke/i,
  },
];

for (const fx of fixtures) {
  test.describe(`[fixtures] ${fx.path}`, () => {
    test(`fixture served and contains expected scripts`, async ({ page }) => {
      test.info().annotations.push({ type: 'fixtures', description: 'HTTP-only probe of ORT smoke page' });
      const res = await page.request.get(fx.path);
      expect(res.status()).toBe(200);
      const body = await res.text();
      expect(body.length).toBeGreaterThan(50);
      if (fx.title) expect(body).toMatch(fx.title);
      for (const s of fx.expectScripts) {
        expect(body).toContain(`<script src="${s}">`);
      }
    });
  });
}
