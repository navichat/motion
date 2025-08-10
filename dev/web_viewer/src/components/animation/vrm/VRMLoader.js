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
      // Dynamically import only when actually running in a browser with modules
      const THREE = await import('https://unpkg.com/three@0.159.0/build/three.module.js');
      const { GLTFLoader } = await import('https://unpkg.com/three@0.159.0/examples/jsm/loaders/GLTFLoader.js');
      const { VRMLoaderPlugin } = await import('https://unpkg.com/@pixiv/three-vrm@3.1.0/lib/three-vrm.module.js');
      const loader = new GLTFLoader();
      loader.register(parser => new VRMLoaderPlugin(parser));
      const gltf = await loader.parseAsync(arrayBuffer, filename);
      const vrm = gltf.userData.vrm;
      return { vrm, gltf, THREE };
    }
  }
  return { VRMLoaderLite };
});
