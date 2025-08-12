#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const ROOT = process.cwd();
const src = path.join(ROOT, 'audio2gesture_step_fixed.onnx');
const destDir = path.join(ROOT, 'dev/web_viewer/models');
const dest = path.join(destDir, 'audio2gesture_step_fixed.onnx');

if (!fs.existsSync(src)) {
  console.log('[prepare-local-models] Source ONNX not found, skipping:', src);
  process.exit(0);
}
fs.mkdirSync(destDir, { recursive: true });
if (!fs.existsSync(dest)) {
  fs.copyFileSync(src, dest);
  console.log('[prepare-local-models] Copied artifact to', dest);
} else {
  console.log('[prepare-local-models] Artifact already present at', dest);
}
