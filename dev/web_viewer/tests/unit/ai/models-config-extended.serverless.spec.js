const { test, expect } = require('@playwright/test');

test('ModelUrlConfig supports extended models (faceformer, kokoro, whisper, vad, tts, llms, vector, hnsw)', async () => {
  const cfg = require('../../../config/models.config.js');

  // Set URLs/endpoints
  cfg.setModelUrl('faceformer', '/models/faceformer.onnx');
  cfg.setModelUrl('kokoro', '/models/kokoro.onnx');
  cfg.setModelUrl('whisper', '/models/whisper.onnx');
  cfg.setModelUrl('sileroVad', '/models/silero_vad.onnx');
  cfg.setModelUrl('speecht5', '/models/speecht5.onnx');
  cfg.setModelUrl('llama', '/models/llama.bin');
  cfg.setModelUrl('diabloGpt', '/models/diablo_gpt.bin');
  cfg.setModelUrl('easyvector', 'http://localhost:7700');
  cfg.setModelUrl('hnsw', '/indexes/hnsw.idx');
  cfg.setModelUrl('hsnw', '/indexes/hnsw.alias.idx');

  expect(cfg.getModelUrl('faceformer')).toBe('/models/faceformer.onnx');
  expect(cfg.getModelUrl('kokoro')).toBe('/models/kokoro.onnx');
  expect(cfg.getModelUrl('whisper')).toBe('/models/whisper.onnx');
  expect(cfg.getModelUrl('sileroVad')).toBe('/models/silero_vad.onnx');
  expect(cfg.getModelUrl('speecht5')).toBe('/models/speecht5.onnx');
  expect(cfg.getModelUrl('llama')).toBe('/models/llama.bin');
  expect(cfg.getModelUrl('diabloGpt')).toBe('/models/diablo_gpt.bin');
  // easyvector endpoint is stored under endpoint; getModelUrl won't read it; validate via internal MODELS
  expect(cfg.MODELS.easyvector.endpoint).toBe('http://localhost:7700');
  expect(cfg.getModelUrl('hnsw')).toBe('/indexes/hnsw.idx');
  expect(cfg.getModelUrl('hsnw')).toBe('/indexes/hnsw.alias.idx');
});
