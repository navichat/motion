// SceneBuilder: classroom stage scaffold (no three.js dependency in this stub)
// UMD export: window.SceneBuilder or module.exports
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SceneBuilder = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  class SceneBuilder {
    constructor() {
      this.scene = { objects: [], meta: { type: 'classroom', version: 1 } };
      this.camera = { position: { x: 0, y: 1.6, z: 3 } };
      this.lights = [{ type: 'hemisphere', intensity: 0.8 }];
    }
    addProp(name, data = {}) {
      const prop = { name, data };
      this.scene.objects.push(prop);
      return prop;
    }
    listProps() { return this.scene.objects.map(o => o.name); }
    getProp(name) { return this.scene.objects.find(o => o.name === name) || null; }
    build() { return { scene: this.scene, camera: this.camera, lights: this.lights }; }
  }
  return { SceneBuilder };
});
