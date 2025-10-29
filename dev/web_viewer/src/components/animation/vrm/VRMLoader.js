/**
 * VRMLoader: tiny wrapper that defers importing three-vrm.
 * For unit/serverless tests, construct without loading to avoid network.
 */
(function(factory){
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else {
    const api = factory();
    if (typeof window !== 'undefined') window.VRMLoaderLite = api.VRMLoaderLite;
  }
})(function(){
  class VRMLoaderLite {
    constructor(opts = {}) {
      this.opts = opts;
    }
    async loadFromArrayBuffer(arrayBuffer, filename = 'model.vrm') {
      // Use the same Three.js version as the main application
      const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.177.0/build/three.module.js');
      const { GLTFLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.177.0/examples/jsm/loaders/GLTFLoader.js');
      const { VRMLoaderPlugin } = await import('https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.6/lib/three-vrm.module.js');
      const loader = new GLTFLoader();
      loader.register(parser => new VRMLoaderPlugin(parser));
      const gltf = await loader.parseAsync(arrayBuffer, filename);
      const vrm = gltf.userData.vrm;
      return { vrm, gltf, THREE };
    }
  }
  return { VRMLoaderLite };
});
