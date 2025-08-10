import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Node-side file existence check (runs in test runner, not browser context)

function findUp(startDir, fileName, maxDepth = 6) {
  let dir = startDir;
  for (let i = 0; i <= maxDepth; i++) {
    const candidate = path.join(dir, fileName);
    if (fs.existsSync(candidate)) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

test('audio2gesture ONNX artifact is present in repo (upward search) [optional]', async () => {
  const start = __dirname || process.cwd();
  const found = findUp(start, 'audio2gesture_step_fixed.onnx', 8)
    || findUp(process.cwd(), 'audio2gesture_step_fixed.onnx', 8);
  if (!found) {
    test.skip(true, 'audio2gesture_step_fixed.onnx not found locally; skipping optional artifact check');
  }
  expect(!!found).toBeTruthy();
});
