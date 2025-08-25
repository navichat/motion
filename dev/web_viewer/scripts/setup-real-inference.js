#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { download } = require('./download-file');

async function main() {
  const ROOT = process.cwd();
  const outDir = path.join(ROOT, 'dev/web_viewer/vendor');
  fs.mkdirSync(outDir, { recursive: true });

  // Tiny runtime assets (JS only; small footprint)
  const assets = {
    ort: {
      url: process.env.RUNTIME_ORT || 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
      dest: path.join(outDir, 'ort-wasm.min.js'),
      env: 'RUNTIME_ORT'
    },
    transformers: {
      url: process.env.TRANSFORMERS_URL || 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js',
      dest: path.join(outDir, 'transformers.min.js'),
      env: 'TRANSFORMERS_URL'
    },
    kokoroJs: {
      url: process.env.KOKORO_JS_URL || 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.min.js',
      dest: path.join(outDir, 'kokoro.min.js'),
      env: 'KOKORO_JS_URL'
    }
  };

  const envLines = [];
  for (const [key, cfg] of Object.entries(assets)) {
    try {
      console.log('[setup] fetching', key, 'from', cfg.url);
      await download(cfg.url, cfg.dest);
      envLines.push(`${cfg.env}=/${path.relative(path.join(ROOT, 'dev/web_viewer'), cfg.dest).replace(/\\/g,'/')}`);
    } catch (e) {
      console.warn('[setup] skipping', key, e.message);
    }
  }

  // Optional: VAD ONNX if provided via env; avoid defaulting to a large model
  const vadUrl = process.env.VAD_URL;
  if (vadUrl) {
    const vadDest = path.join(ROOT, 'dev/web_viewer/models/silero_vad.onnx');
    try {
      console.log('[setup] fetching silero VAD model from', vadUrl);
      await download(vadUrl, vadDest);
      envLines.push(`VAD_URL=/models/silero_vad.onnx`);
    } catch (e) {
      console.warn('[setup] skipping VAD model', e.message);
    }
  }

  // Optional: transformers VAD model (folder or model.json). If provided, wire for tests:
  // TRANSFORMERS_MODEL_VAD=/vendor/models/silero-vad (served path)
  const vadModelUrl = process.env.TRANSFORMERS_MODEL_VAD_URL;
  const vadModelPath = process.env.TRANSFORMERS_MODEL_VAD_PATH;
  if (vadModelUrl) {
    const dest = path.join(outDir, 'models/silero-vad/model.json');
    try {
      console.log('[setup] fetching transformers VAD model from', vadModelUrl);
      await download(vadModelUrl, dest);
      envLines.push('TRANSFORMERS_MODEL_VAD=/vendor/models/silero-vad');
    } catch (e) {
      console.warn('[setup] skipping transformers VAD model', e.message);
    }
  }
  if (vadModelPath) {
    try {
      const src = path.isAbsolute(vadModelPath) ? vadModelPath : path.join(ROOT, vadModelPath);
      const dest = path.join(outDir, 'models/silero-vad/model.json');
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(src, dest);
      console.log('[setup] copied local transformers VAD model.json from', src);
      envLines.push('TRANSFORMERS_MODEL_VAD=/vendor/models/silero-vad');
    } catch (e) {
      console.warn('[setup] failed to copy local VAD model.json', e.message);
    }
  }

  // Always attempt to fetch tiny VAD metadata so local folder exists (small files only)
  try {
    const vadMetaDir = path.join(outDir, 'models/silero-vad');
    fs.mkdirSync(vadMetaDir, { recursive: true });
    const cfgDest = path.join(vadMetaDir, 'config.json');
    const preDest = path.join(vadMetaDir, 'preprocessor_config.json');
    // Create minimal stub metadata if remote fetch is not possible
    if (!fs.existsSync(cfgDest)) {
      console.log('[setup] writing stub VAD config.json');
      fs.writeFileSync(cfgDest, JSON.stringify({
        _name_or_path: "Xenova/silero-vad",
        model_type: "silero-vad"
      }, null, 2));
    }
    if (!fs.existsSync(preDest)) {
      console.log('[setup] writing stub VAD preprocessor_config.json');
      fs.writeFileSync(preDest, JSON.stringify({
        feature_extractor_type: "Wav2Vec2FeatureExtractor",
        sampling_rate: 16000
      }, null, 2));
    }
    // Point tests to this local folder
    if (!envLines.find(l => l.startsWith('TRANSFORMERS_MODEL_VAD='))) {
      envLines.push('TRANSFORMERS_MODEL_VAD=/vendor/models/silero-vad');
    }
  } catch (e) {
    console.warn('[setup] VAD metadata fetch skipped', e.message);
  }

  const envDir = path.join(ROOT, 'dev/web_viewer/.env');
  fs.writeFileSync(envDir, envLines.join('\n') + '\n', 'utf8');
  console.log('[setup] wrote env to', envDir);

  console.log('[setup] done. To run, you can source the env then run real-inference tests.');
}

main().catch((e) => { console.error(e); process.exit(1); });
