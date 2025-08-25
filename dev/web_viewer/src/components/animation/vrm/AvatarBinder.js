/**
 * AvatarBinder: minimal VRM adapter that BVHTimelineVRMIntegration can drive.
 * - In "stub" mode (no VRM provided), records updateBone calls for tests.
 * - When provided a VRM (three-vrm) instance, applies pos/euler(rot) to humanoid bones.
 */
(function(factory){
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else {
    const api = factory();
    if (typeof window !== 'undefined') window.AvatarBinder = api.AvatarBinder;
  }
})(function(){
  class AvatarBinder {
    constructor(vrm /* optional */, opts = {}) {
      this.vrm = vrm || null;
      this.stub = !vrm; // no VRM → record-only mode for serverless/unit
      this.applyScale = opts.applyScale ?? 1.0;
      this.records = [];
  this.blendshapeRecords = [];
      this.updateCalls = 0;
      this.errors = 0;
    }

    getBoneNode(name) {
      try {
        return this.vrm?.humanoid?.getBoneNode?.(name) || null;
      } catch (_) { return null; }
    }

    updateBone(boneName, position, rotation) {
      // Always record for diagnostics/tests
      this.records.push({ boneName, position, rotation });

      if (this.stub) return true; // nothing else to do

      const node = this.getBoneNode(boneName);
      if (!node) return false;

      try {
        if (position) {
          const s = this.applyScale;
          if (node.position?.set) node.position.set((position.x||0)*s, (position.y||0)*s, (position.z||0)*s);
        }
        if (rotation) {
          // rotation expected as Euler radians {x,y,z}
          if (node.rotation) {
            node.rotation.x = rotation.x || 0;
            node.rotation.y = rotation.y || 0;
            node.rotation.z = rotation.z || 0;
          } else if (node.setRotationFromEuler) {
            // fallback if provided differently
            node.setRotationFromEuler({ x: rotation.x||0, y: rotation.y||0, z: rotation.z||0 });
          }
        }
        return true;
      } catch (e) {
        this.errors++;
        return false;
      }
    }

    update(time) {
      this.updateCalls++;
      if (this.stub) return;
      try { this.vrm?.update?.(time); } catch (_) { /* ignore */ }
    }

    /**
     * Update a facial expression/blendshape weight.
     * In stub mode, record the update. With a real VRM, try expressionManager APIs.
     */
    updateBlendshape(name, weight) {
      this.blendshapeRecords.push({ name, weight });
      if (this.stub) return true;
      try {
        // VRM 1.0 style
        const em = this.vrm?.expressionManager || this.vrm?.blendShapeProxy || null;
        if (em && typeof em.setValue === 'function') {
          em.setValue(name, weight);
          return true;
        }
        // VRM 0.x fallback
        if (em && typeof em.setValue === 'function') {
          em.setValue(name, weight);
          return true;
        }
      } catch (_) { /* ignore */ }
      return false;
    }

    getStats() {
      return {
        applied: this.records.length,
        updateCalls: this.updateCalls,
        errors: this.errors,
  stub: this.stub,
  expressions: this.blendshapeRecords.length
      };
    }

    clear() { this.records.length = 0; this.errors = 0; }
  }

  return { AvatarBinder };
});
