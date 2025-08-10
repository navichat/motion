/**
 * Minimal Fibonacci Heap for task prioritization.
 * Amortized O(1) insert & decreaseKey, O(log n) extractMin.
 */
class FibNode {
  constructor(key, value) {
    this.key = key;
    this.value = value;
    this.degree = 0;
    this.mark = false;
    this.parent = null;
    this.child = null;
    this.left = this;
    this.right = this;
  }
}

class FibonacciHeap {
  constructor() {
    this.min = null;
    this.n = 0;
  }

  isEmpty() { return this.min === null; }

  insert(key, value) {
    const node = new FibNode(key, value);
    // add to root list
    if (this.min) {
      node.left = this.min;
      node.right = this.min.right;
      this.min.right.left = node;
      this.min.right = node;
      if (key < this.min.key) this.min = node;
    } else {
      this.min = node;
    }
    this.n++;
    return node;
  }

  minNode() { return this.min; }

  extractMin() {
    const z = this.min;
    if (!z) return null;

    // add children to root list
    let x = z.child;
    if (x) {
      const start = x;
      do {
        const next = x.right;
        this._addRoot(x);
        x.parent = null;
        x = next;
      } while (x !== start);
      z.child = null;
    }

    // remove z from root list
    this._removeFromRoot(z);
    if (z === z.right) {
      this.min = null;
    } else {
      this.min = z.right;
      this._consolidate();
    }
    this.n--;
    return z;
  }

  decreaseKey(node, newKey) {
    if (newKey > node.key) throw new Error('new key is greater than current key');
    node.key = newKey;
    const y = node.parent;
    if (y && node.key < y.key) {
      this._cut(node, y);
      this._cascadingCut(y);
    }
    if (this.min && node.key < this.min.key) this.min = node;
  }

  _addRoot(node) {
    // insert node into root list next to min
    node.left = this.min;
    node.right = this.min.right;
    this.min.right.left = node;
    this.min.right = node;
  }

  _removeFromRoot(node) {
    node.left.right = node.right;
    node.right.left = node.left;
  }

  _link(y, x) {
    // remove y from root list
    this._removeFromRoot(y);
    // make y a child of x
    y.left = y.right = y;
    if (!x.child) {
      x.child = y;
    } else {
      y.left = x.child;
      y.right = x.child.right;
      x.child.right.left = y;
      x.child.right = y;
    }
    y.parent = x;
    x.degree++;
    y.mark = false;
  }

  _consolidate() {
    const A = new Array(Math.floor(Math.log2(this.n)) + 5).fill(null);
    const roots = [];
    if (!this.min) return;
    let x = this.min;
    do {
      roots.push(x);
      x = x.right;
    } while (x !== this.min);

    for (const w of roots) {
      let x = w;
      let d = x.degree;
      while (A[d]) {
        let y = A[d];
        if (x.key > y.key) [x, y] = [y, x];
        this._link(y, x);
        A[d] = null;
        d++;
      }
      A[d] = x;
    }

    this.min = null;
    for (const a of A) if (a) {
      if (!this.min) {
        this.min = a;
        a.left = a.right = a;
      } else {
        // add a to root list
        a.left = this.min;
        a.right = this.min.right;
        this.min.right.left = a;
        this.min.right = a;
        if (a.key < this.min.key) this.min = a;
      }
    }
  }

  _cut(x, y) {
    // remove x from child list of y
    if (y.child === x) {
      if (x.right !== x) y.child = x.right; else y.child = null;
    }
    x.left.right = x.right;
    x.right.left = x.left;
    y.degree--;
    // add x to root list
    x.left = this.min;
    x.right = this.min.right;
    this.min.right.left = x;
    this.min.right = x;
    x.parent = null;
    x.mark = false;
  }

  _cascadingCut(y) {
    const z = y.parent;
    if (z) {
      if (!y.mark) y.mark = true; else {
        this._cut(y, z);
        this._cascadingCut(z);
      }
    }
  }
}

// Always expose to window when available
if (typeof window !== 'undefined') {
  window.FibonacciHeap = FibonacciHeap;
}
// Also support CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { FibonacciHeap, FibNode };
}
