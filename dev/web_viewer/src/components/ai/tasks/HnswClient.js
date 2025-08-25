// HnswClient: thin wrapper that uses hnswlib-wasm when available, else falls back to simulated interface.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.HnswClient = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class HnswClient {
    constructor({ dimensions = 512, space = 'l2' } = {}) {
      this.dimensions = dimensions; this.space = space; this.index = null; this.real = false;
    }
    async initialize() {
      try {
        if (typeof window !== 'undefined' && window.HnswlibWasm) {
          const hnswlib = await window.HnswlibWasm.init();
          this.index = new hnswlib.HierarchicalNSW(this.space, this.dimensions);
          this.real = true; return true;
        }
      } catch {}
      // fallback: in-memory vectors
      this.index = { data: [] };
      this.real = false;
      return false;
    }
    async addItem(vector, id) {
      if (!this.index) await this.initialize();
      if (this.real && this.index && this.index.addPoint) {
        this.index.addPoint(vector, id);
      } else {
        this.index.data.push({ id, vector });
      }
    }
    async searchKnn(query, k = 5) {
      if (!this.index) await this.initialize();
      if (this.real && this.index && this.index.searchKnn) return this.index.searchKnn(query, k);
      // fallback cosine
  let sims = this.index.data.map((v, idx) => ({ id: v.id ?? idx, score: this._cosine(v.vector || v, query) }));
      sims.sort((a, b) => b.score - a.score);
  return sims.slice(0, k).map(s => ({ id: s.id, score: s.score }));
    }
    _cosine(a, b) {
      let dot = 0, na = 0, nb = 0; for (let i = 0; i < Math.min(a.length, b.length); i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-6);
    }
  }
  return { HnswClient };
});
