import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const WHISPER = read('dev/web_viewer/src/components/ai/tasks/WhisperOrtTask.js');
const FACEFORMER = read('dev/web_viewer/src/components/animation/timeline/tasks/FaceformerOrtTask.js');
const VAD = read('dev/web_viewer/src/components/ai/tasks/SileroVadOrtTask.js');
const SPEECHT5 = read('dev/web_viewer/src/components/ai/tasks/SpeechT5OrtTask.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Sanity: the pre-refactor tasks expose UMD symbols in window when injected.

test('UMD exports for ORT tasks are present (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, WHISPER);
  await inject(page, FACEFORMER);
  await inject(page, VAD);
  await inject(page, SPEECHT5);
  const present = await page.evaluate(() => {
    return {
      whisper: !!(window.WhisperOrtTask && window.WhisperOrtTask.WhisperOrtTask),
      faceformer: !!(window.FaceformerOrtTask && window.FaceformerOrtTask.FaceformerOrtTask),
      vad: !!(window.SileroVadOrtTask && window.SileroVadOrtTask.SileroVadOrtTask),
      speecht5: !!(window.SpeechT5OrtTask && window.SpeechT5OrtTask.SpeechT5OrtTask),
    };
  });
  expect(present.whisper).toBeTruthy();
  expect(present.faceformer).toBeTruthy();
  expect(present.vad).toBeTruthy();
  expect(present.speecht5).toBeTruthy();
});
