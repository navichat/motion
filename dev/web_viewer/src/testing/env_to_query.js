// Utility to build a query string from environment variables in unit-web tests.
export function buildQueryFromEnv(env) {
  const params = new URLSearchParams();
  if (env.RUNTIME_ORT) params.set('runtime.ort', env.RUNTIME_ORT);
  if (env.WHISPER_URL) params.set('whisper', env.WHISPER_URL);
  if (env.VAD_URL) params.set('sileroVad', env.VAD_URL);
  if (env.FACEFORMER_URL) params.set('faceformer', env.FACEFORMER_URL);
  if (env.SPEECHT5_URL) params.set('speecht5', env.SPEECHT5_URL);
  if (env.KOKORO_ID) params.set('kokoro', env.KOKORO_ID);
  if (env.TRANSFORMERS_URL) params.set('transformers', env.TRANSFORMERS_URL);
  if (env.KOKORO_JS_URL) params.set('kokoroJs', env.KOKORO_JS_URL);
  if (env.LLAMA_ENDPOINT) params.set('llama', env.LLAMA_ENDPOINT);
  if (env.DIABLO_GPT_ENDPOINT) params.set('diabloGpt', env.DIABLO_GPT_ENDPOINT);
  if (env.EASYVECTOR_ENDPOINT) params.set('easyvector', env.EASYVECTOR_ENDPOINT);
  // Test-only overrides
  if (env.TRANSFORMERS_MODEL_VAD) params.set('vadModel', env.TRANSFORMERS_MODEL_VAD);
  // Legacy module exposure toggles
  if (env.LEGACY === '1') params.set('legacy', '1');
  if (env.LEGACY_WHISPER === '1') params.set('legacyWhisper', '1');
  if (env.LEGACY_KOKORO === '1') params.set('legacyKokoro', '1');
  if (env.LEGACY_LLAMA === '1') params.set('legacyLlama', '1');
  return params.toString();
}
