class WorkerPool {
  constructor() {
    this.pools = new Map(); // kind -> { size, busy:Set, idle:Array<WorkerLike> }
  }
  configure(kind, size, factory) {
    const idle = [];
    for (let i = 0; i < size; i++) idle.push(factory());
    this.pools.set(kind, { size, busy: new Set(), idle, factory });
  }
  acquire(kind) {
    const p = this.pools.get(kind);
    if (!p) throw new Error(`No pool for ${kind}`);
    const w = p.idle.pop() || p.factory();
    p.busy.add(w);
    return w;
  }
  release(kind, worker) {
    const p = this.pools.get(kind);
    if (!p) return;
    p.busy.delete(worker);
    p.idle.push(worker);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WorkerPool };
} else {
  window.WorkerPool = WorkerPool;
}
