// EasyVectorClient: minimal vector DB client using configured endpoint.
// Supports upsert and query; falls back to in-memory store.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.EasyVectorClient = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class EasyVectorClient {
    constructor({ endpoint } = {}) {
      this.endpoint = endpoint || this._resolveEndpoint();
      this.mem = [];
    }
  async initialize() { return true; }
    _resolveEndpoint() {
      try {
        const g = (typeof self !== 'undefined' ? self : (typeof window !== 'undefined' ? window : undefined));
        if (g && g.ModelUrlConfig && g.ModelUrlConfig.MODELS && g.ModelUrlConfig.MODELS.easyvector) return g.ModelUrlConfig.MODELS.easyvector.endpoint;
        if (typeof require !== 'undefined') return require('../../../config/models.config.js').MODELS.easyvector.endpoint;
      } catch {}
      return undefined;
    }
    async upsert(vectors) {
      if (this.endpoint && typeof fetch === 'function') {
        try {
          const res = await fetch(this.endpoint + '/upsert', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ vectors }) });
          if (res.ok) return true;
        } catch {}
      }
      // fallback
      this.mem.push(...vectors);
      return true;
    }
  async query(vector, k = 5) {
      if (this.endpoint && typeof fetch === 'function') {
        try {
      const res = await fetch(this.endpoint + '/query', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ vector, k }) });
          if (res.ok) return await res.json();
        } catch {}
      }
      // fallback cosine over mem
    const sims = this.mem.map((v, idx) => ({ idx, score: this._cosine(v.vector || v, vector) }));
      sims.sort((a, b) => b.score - a.score);
    return sims.slice(0, k).map(m => ({ id: m.idx, score: m.score }));
    }
    _cosine(a, b) {
      let dot = 0, na = 0, nb = 0; for (let i = 0; i < Math.min(a.length, b.length); i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-6);
    }
  }
  return { EasyVectorClient };
});
