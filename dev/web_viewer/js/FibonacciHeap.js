/**
 * Fibonacci Heap Implementation for Task Scheduling
 * Optimized for priority queue operations with O(1) insert and decrease-key
 */

class FibonacciHeapNode {
    constructor(key, value) {
        this.key = key;
        this.value = value;
        this.parent = null;
        this.child = null;
        this.left = this;
        this.right = this;
        this.degree = 0;
        this.marked = false;
    }
}

class FibonacciHeap {
    constructor() {
        this.min = null;
        this.count = 0;
    }

    /**
     * Insert a new node with given key and value
     * Time Complexity: O(1)
     */
    insert(key, value) {
        const node = new FibonacciHeapNode(key, value);
        
        if (this.min === null) {
            this.min = node;
        } else {
            this._addToRootList(node);
            if (node.key < this.min.key) {
                this.min = node;
            }
        }
        
        this.count++;
        return node;
    }

    /**
     * Extract the minimum node
     * Time Complexity: O(log n) amortized
     */
    extractMin() {
        const minNode = this.min;
        
        if (minNode === null) {
            return null;
        }

        // Add all children of min to root list
        if (minNode.child !== null) {
            const children = this._getChildrenList(minNode.child);
            for (const child of children) {
                child.parent = null;
                this._addToRootList(child);
            }
        }

        // Remove min from root list
        this._removeFromRootList(minNode);
        
        if (minNode === minNode.right) {
            // Only node in heap
            this.min = null;
        } else {
            this.min = minNode.right;
            this._consolidate();
        }

        this.count--;
        return { key: minNode.key, value: minNode.value };
    }

    /**
     * Decrease the key of a node
     * Time Complexity: O(1) amortized
     */
    decreaseKey(node, newKey) {
        if (newKey > node.key) {
            throw new Error("New key is greater than current key");
        }

        node.key = newKey;
        const parent = node.parent;

        if (parent !== null && node.key < parent.key) {
            this._cut(node, parent);
            this._cascadingCut(parent);
        }

        if (node.key < this.min.key) {
            this.min = node;
        }
    }

    /**
     * Delete a node
     * Time Complexity: O(log n) amortized
     */
    delete(node) {
        this.decreaseKey(node, -Infinity);
        this.extractMin();
    }

    /**
     * Peek at minimum without extracting
     */
    peek() {
        return this.min ? { key: this.min.key, value: this.min.value } : null;
    }

    /**
     * Check if heap is empty
     */
    isEmpty() {
        return this.count === 0;
    }

    /**
     * Get the size of the heap
     */
    size() {
        return this.count;
    }

    // Private helper methods

    _addToRootList(node) {
        if (this.min === null) {
            this.min = node;
            node.left = node;
            node.right = node;
        } else {
            node.left = this.min;
            node.right = this.min.right;
            this.min.right.left = node;
            this.min.right = node;
        }
    }

    _removeFromRootList(node) {
        if (node.right === node) {
            // Only node in list
            return;
        }
        
        node.left.right = node.right;
        node.right.left = node.left;
    }

    _getChildrenList(child) {
        const children = [];
        let current = child;
        
        do {
            children.push(current);
            current = current.right;
        } while (current !== child);
        
        return children;
    }

    _consolidate() {
        const maxDegree = Math.floor(Math.log2(this.count)) + 1;
        const degreeTable = new Array(maxDegree + 1).fill(null);
        
        // Get all root nodes
        const rootNodes = [];
        let current = this.min;
        
        do {
            rootNodes.push(current);
            current = current.right;
        } while (current !== this.min);

        // Consolidate nodes with same degree
        for (const node of rootNodes) {
            let degree = node.degree;
            let x = node;

            while (degreeTable[degree] !== null) {
                let y = degreeTable[degree];
                
                if (x.key > y.key) {
                    [x, y] = [y, x];
                }

                this._link(y, x);
                degreeTable[degree] = null;
                degree++;
            }

            degreeTable[degree] = x;
        }

        // Rebuild root list and find new minimum
        this.min = null;
        
        for (const node of degreeTable) {
            if (node !== null) {
                if (this.min === null) {
                    this.min = node;
                    node.left = node;
                    node.right = node;
                } else {
                    this._addToRootList(node);
                    if (node.key < this.min.key) {
                        this.min = node;
                    }
                }
            }
        }
    }

    _link(child, parent) {
        // Remove child from root list
        this._removeFromRootList(child);
        
        // Make child a child of parent
        child.parent = parent;
        
        if (parent.child === null) {
            parent.child = child;
            child.left = child;
            child.right = child;
        } else {
            child.left = parent.child;
            child.right = parent.child.right;
            parent.child.right.left = child;
            parent.child.right = child;
        }

        parent.degree++;
        child.marked = false;
    }

    _cut(child, parent) {
        // Remove child from parent's child list
        if (child.right === child) {
            parent.child = null;
        } else {
            if (parent.child === child) {
                parent.child = child.right;
            }
            child.left.right = child.right;
            child.right.left = child.left;
        }

        parent.degree--;
        
        // Add child to root list
        child.parent = null;
        child.marked = false;
        this._addToRootList(child);
    }

    _cascadingCut(node) {
        const parent = node.parent;
        
        if (parent !== null) {
            if (!node.marked) {
                node.marked = true;
            } else {
                this._cut(node, parent);
                this._cascadingCut(parent);
            }
        }
    }

    /**
     * Get heap statistics for debugging
     */
    getStats() {
        return {
            size: this.count,
            minKey: this.min ? this.min.key : null,
            isEmpty: this.isEmpty()
        };
    }

    /**
     * Validate heap properties (for testing)
     */
    validate() {
        if (this.min === null) {
            return this.count === 0;
        }

        // Check if min is actually minimum
        let current = this.min;
        do {
            if (current.key < this.min.key) {
                return false;
            }
            current = current.right;
        } while (current !== this.min);

        return true;
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FibonacciHeap, FibonacciHeapNode };
} else if (typeof window !== 'undefined') {
    window.FibonacciHeap = FibonacciHeap;
    window.FibonacciHeapNode = FibonacciHeapNode;
}
