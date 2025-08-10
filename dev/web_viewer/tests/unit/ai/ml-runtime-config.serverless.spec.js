import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROOT = process.cwd();
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');
const RUNTIME = read('dev/web_viewer/src/components/ml/ModelRuntimeConfig.js');

function inject(page, code) { return page.addScriptTag({ content: code }); }

// Serverless test: verify provider selection logic and path helper

test('ModelRuntimeConfig selects provider and resolves model path (serverless)', async ({ page }) => {
  await page.goto('about:blank');
  await inject(page, RUNTIME);

  const res = await page.evaluate(() => {
    const mod = (typeof module !== 'undefined' && module.exports) ? module.exports : window.ModelRuntimeConfig;
    if (!mod) throw new Error('ModelRuntimeConfig missing');

    const { selectOrtProvider, detectWebGPU, getModelFsPath } = mod;

    // Fake env without WebGPU
    const provider1 = selectOrtProvider({}, { navigator: {} });
    // Fake env with WebGPU
    const provider2 = selectOrtProvider({}, { navigator: { gpu: {} } });

    const hasGpuFalse = detectWebGPU({ navigator: {} });
    const hasGpuTrue = detectWebGPU({ navigator: { gpu: {} } });

    const audioModel = getModelFsPath('audio2gesture');

    return { provider1, provider2, hasGpuFalse, hasGpuTrue, audioModelEndsWith: audioModel.endsWith('audio2gesture_step_fixed.onnx') };
  });

  expect(res.provider1).toBe('wasm');
  expect(res.provider2).toBe('webgpu');
  expect(res.hasGpuFalse).toBeFalsy();
  expect(res.hasGpuTrue).toBeTruthy();
  expect(res.audioModelEndsWith).toBeTruthy();
});
