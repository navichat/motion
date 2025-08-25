// LlmEndpointTask: Calls a configurable LLM HTTP endpoint (llama/diablo-gpt) via fetch.
// Fallback: echoes the prompt.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LlmEndpointTask = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class LlmEndpointTask {
    constructor({ id = 'llm', provider = 'llama', prompt = 'Hello', endpoint } = {}) {
      this.id = id; this.provider = provider; this.prompt = prompt; this.endpoint = endpoint;
    }
    _resolveEndpoint() {
      if (this.endpoint) return this.endpoint;
      try {
        const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
        const key = this.provider === 'diablo-gpt' ? 'diabloGpt' : 'llama';
        if (g && g.ModelUrlConfig && typeof g.ModelUrlConfig.getModelUrl === 'function') return g.ModelUrlConfig.getModelUrl(key);
        if (typeof require !== 'undefined') return require('../../../config/models.config.js').getModelUrl(key);
      } catch {}
      return undefined;
    }
    async *run({ prompt } = {}) {
      const ep = this._resolveEndpoint();
      const p = prompt || this.prompt;
      if (ep && typeof fetch === 'function') {
        try {
          const res = await fetch(ep, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: p }) });
          if (res.ok) {
            const data = await res.json().catch(() => ({}));
            const text = data.text || data.output || data.choices?.[0]?.text || '';
            yield { t0: 0, dt: 0, frames: [], metadata: { model: 'llm_endpoint', provider: this.provider, text } };
            return;
          }
        } catch {}
      }
      yield { t0: 0, dt: 0, frames: [], metadata: { model: 'llm_fallback', provider: this.provider, text: p } };
    }
  }
  return { LlmEndpointTask };
});
