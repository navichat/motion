(function(){
  try {
    const g = window; if (!g || !g.ModelUrlConfig) return;
    const params = new URLSearchParams(location.search);
    const set = (key) => { const v = params.get(key); if (v) try { g.ModelUrlConfig.setModelUrl(key, v); } catch {} };
    // Known model keys
    ['runtime.ort','whisper','sileroVad','faceformer','speecht5','kokoro','llama','diabloGpt'].forEach(set);
    const ev = params.get('easyvector');
    if (ev) {
      g.ModelUrlConfig.MODELS = g.ModelUrlConfig.MODELS || {};
      g.ModelUrlConfig.MODELS.easyvector = { endpoint: ev };
    }
    // Optionally load ORT script if provided and not already present
    const ortUrl = params.get('runtime.ort') || (g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('runtime.ort'));
    if (ortUrl && !g.ort) {
      const s = document.createElement('script'); s.src = ortUrl; s.async = true; s.onload = ()=>console.log('[bootstrap] ORT runtime loaded');
      s.onerror = ()=>console.warn('[bootstrap] failed to load ORT from', ortUrl);
      document.head.appendChild(s);
    }

    // Optionally load transformers.js if provided and not already present
    const tfUrl = params.get('transformers');
    if (tfUrl && !g.transformers) {
      const s = document.createElement('script'); s.src = tfUrl; s.async = true; s.type = 'module';
      s.onload = ()=>console.log('[bootstrap] transformers.js loaded');
      s.onerror = ()=>console.warn('[bootstrap] failed to load transformers.js from', tfUrl);
      document.head.appendChild(s);
    }

    // Optionally load kokoro-js if provided and not already present
    const kokoroUrl = params.get('kokoroJs');
    if (kokoroUrl && !g.KokoroTTS) {
      const s = document.createElement('script'); s.src = kokoroUrl; s.async = true;
      s.onload = ()=>console.log('[bootstrap] kokoro-js loaded');
      s.onerror = ()=>console.warn('[bootstrap] failed to load kokoro-js from', kokoroUrl);
      document.head.appendChild(s);
    }

    // Optional: expose legacy pre-refactor modules onto window for compatibility/testing.
    // Enable by adding ?legacy=1 or granular flags (?legacyWhisper=1&legacyKokoro=1&legacyLlama=1)
    const wantLegacy = params.get('legacy') === '1';
    const wantLegacyWhisper = wantLegacy || params.get('legacyWhisper') === '1';
    const wantLegacyKokoro = wantLegacy || params.get('legacyKokoro') === '1';
    const wantLegacyLlama = wantLegacy || params.get('legacyLlama') === '1';
    const needResourceManager = wantLegacyWhisper || wantLegacyKokoro || wantLegacyLlama;
    if (needResourceManager) {
      const mod = document.createElement('script');
      mod.type = 'module';
      // Inline module that imports and re-exports selected classes onto window
      mod.textContent = `
        (async () => {
          try {
            ${needResourceManager ? "const { ResourceManager } = await import('/src/utils/ResourceManager.js'); window.ResourceManager = ResourceManager;" : ''}
            ${wantLegacyWhisper ? "const { WhisperModule } = await import('/src/audio/WhisperModule.js'); window.WhisperModule = WhisperModule;" : ''}
            ${wantLegacyKokoro ? "const { KokoroModule } = await import('/src/audio/KokoroModule.js'); window.KokoroModule = KokoroModule;" : ''}
            ${wantLegacyLlama ? "const { LlamaModule } = await import('/src/ai/LlamaModule.js'); window.LlamaModule = LlamaModule;" : ''}
            console.log('[bootstrap] legacy modules exposed', {
              ResourceManager: !!window.ResourceManager,
              WhisperModule: !!window.WhisperModule,
              KokoroModule: !!window.KokoroModule,
              LlamaModule: !!window.LlamaModule,
            });
          } catch (e) {
            console.warn('[bootstrap] failed exposing legacy modules', e);
          }
        })();
      `;
      document.head.appendChild(mod);
    }
    const summary = {
      'runtime.ort': g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('runtime.ort'),
  transformers: !!g.transformers,
      whisper: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('whisper'),
      sileroVad: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('sileroVad'),
      faceformer: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('faceformer'),
      speecht5: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('speecht5'),
  kokoro: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('kokoro'),
      llama: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('llama'),
      diabloGpt: g.ModelUrlConfig.getModelUrl && g.ModelUrlConfig.getModelUrl('diabloGpt'),
      easyvector: g.ModelUrlConfig.MODELS && g.ModelUrlConfig.MODELS.easyvector && g.ModelUrlConfig.MODELS.easyvector.endpoint,
      legacy: {
        ResourceManager: !!g.ResourceManager,
        WhisperModule: !!g.WhisperModule,
        KokoroModule: !!g.KokoroModule,
        LlamaModule: !!g.LlamaModule,
      }
    };
    console.log('[bootstrap] ModelUrlConfig summary', summary);
  } catch (e) {
    console.warn('[bootstrap] error', e);
  }
})();
