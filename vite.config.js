// Vite config focused on browser-only builds for dev/web_viewer
const path = require('path');
module.exports = {
  root: 'dev/web_viewer',
  publicDir: 'public',
  server: {
  port: 5173,
  strictPort: true,
  hmr: { overlay: false },
  watch: { usePolling: true, interval: 300 },
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
  build: {
    outDir: '../../dist-web_viewer',
    emptyOutDir: true,
    target: 'es2019',
    assetsInlineLimit: 0, // keep large assets (wasm/models) external
    rollupOptions: {
      input: {
        ichika: path.resolve(__dirname, 'dev/web_viewer/demos/ichika_voice_conversation_demo.html'),
      },
    },
  },
  define: {
    'process.env': {},
    global: 'window',
  },
  optimizeDeps: {
    // ensure these on-device libs are not prebundled; they run in browser at runtime
    exclude: ['onnxruntime-web', '@xenova/transformers'],
  },
};
