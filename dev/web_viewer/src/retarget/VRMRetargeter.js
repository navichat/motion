// VRMRetargeter: BVH frame to VRM humanoid transforms (stub)
// UMD export: window.VRMRetargeter or module.exports
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.VRMRetargeter = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class VRMRetargeter {
    constructor(mapping = {}) { this.mapping = mapping; }
    // bvhFrame: { joints: { name: { pos:[x,y,z], rot:[x,y,z,w] } } }
    retarget(bvhFrame, humanoid) {
      const out = {};
      const joints = (bvhFrame && (bvhFrame.joints || bvhFrame.motionData?.joints)) || {};
      for (const [src, dst] of Object.entries(this.mapping)) {
        const j = joints[src];
        if (j) out[dst] = { pos: j.pos || [0,0,0], rot: j.rot || [0,0,0,1] };
      }
      return out; // { boneName: { pos, rot } }
    }
  }
  return VRMRetargeter;
});
