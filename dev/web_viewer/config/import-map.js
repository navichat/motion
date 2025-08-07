/**
 * Import Map for Reorganized Web Viewer Structure
 * 
 * This file provides the new import paths after reorganization.
 * Update your imports according to this mapping.
 */

export const ImportMap = {
  // Core System Modules
  core: {
    'TaskManager': '../../src/core/TaskManager.js',
    'FibonacciHeap': '../../src/core/FibonacciHeap.js',
    'SystemPerformanceAnalyzer': '../../src/core/SystemPerformanceAnalyzer.js'
  },

  // AI Model Modules
  ai: {
    jobs: {
      'AIModelJobs': '../../src/ai/jobs/AIModelJobs.js',
      'RealJobFactory': '../../src/ai/jobs/RealJobFactory.js',
      'RealAIModelJobs': '../../src/ai/jobs/RealAIModelJobs.js',
      'KNNJobs': '../../src/ai/jobs/KNNJobs.js'
    },
    workers: {
      'ai-model-inference-worker': '../../src/ai/workers/ai-model-inference-worker.js',
      'model-loader-webnn': '../../src/ai/workers/model-loader-webnn.js',
      'onnx-runtime-fixer': '../../src/ai/workers/onnx-runtime-fixer.js'
    }
  },

  // Avatar System Modules
  avatar: {
    vrm: {
      'VRMBVHAdapter': '../../src/avatar/vrm/VRMBVHAdapter.js',
      'VRMDiagnostics': '../../src/avatar/vrm/VRMDiagnostics.js',
      'VRMLightingManager': '../../src/avatar/vrm/VRMLightingManager.js',
      'VRMMaterialExtractor': '../../src/avatar/vrm/VRMMaterialExtractor.js',
      'VRMVisibilityFix': '../../src/avatar/vrm/VRMVisibilityFix.js',
      'AdvancedVRMLoader': '../../src/avatar/vrm/AdvancedVRMLoader.js',
      'EnhancedCharacterSystem': '../../src/avatar/vrm/EnhancedCharacterSystem.js',
      'EnhancedTaskManagerTest': '../../src/avatar/vrm/EnhancedTaskManagerTest.js',
      'EnhancedTaskManagerTestDemo': '../../src/avatar/vrm/EnhancedTaskManagerTestDemo.js',
      'EnhancedVRMBVHAdapter': '../../src/avatar/vrm/EnhancedVRMBVHAdapter.js'
    },
    animation: {
      'AnimationBlender': '../../src/avatar/animation/AnimationBlender.js',
      'AnimationSync': '../../src/avatar/animation/AnimationSync.js'
    },
    motion: {
      'BVHTimeline': '../../src/avatar/motion/BVHTimeline.js',
      'BVHTimelineCompositor': '../../src/avatar/motion/BVHTimelineCompositor.js',
      'BVHTimelineExample': '../../src/avatar/motion/BVHTimelineExample.js',
      'BVHTimelineVRMIntegration': '../../src/avatar/motion/BVHTimelineVRMIntegration.js',
      'Audio2GestureBVHConverter': '../../src/avatar/motion/Audio2GestureBVHConverter.js',
      'Audio2GestureTimelineIntegration': '../../src/avatar/motion/Audio2GestureTimelineIntegration.js',
      'DeepMimicBVHConverter': '../../src/avatar/motion/DeepMimicBVHConverter.js',
      'DeepMimicTimelineIntegration': '../../src/avatar/motion/DeepMimicTimelineIntegration.js',
      'FaceFormerBVHConverter': '../../src/avatar/motion/FaceFormerBVHConverter.js',
      'FaceformerBVHConverter': '../../src/avatar/motion/FaceformerBVHConverter.js',
      'FaceFormerTimelineIntegration': '../../src/avatar/motion/FaceFormerTimelineIntegration.js'
    }
  },

  // Audio Processing Modules
  audio: {
    processing: {
      'AudioProcessor': '../../src/audio/processing/AudioProcessor.js',
      'phonemizer': '../../src/audio/processing/phonemizer.js',
      'tts-playback-processor': '../../src/audio/processing/tts-playback-processor.js',
      'vad-processor': '../../src/audio/processing/vad-processor.js'
    },
    tts: {
      'kokoro-js.esm': '../../src/audio/tts/kokoro-js.esm.js',
      'kokoro-tts': '../../src/audio/tts/kokoro-tts.js'
    }
  },

  // Compute Backend Modules
  compute: {
    webgpu: {
      'RealWebGPUJobs': '../../src/compute/webgpu/RealWebGPUJobs.js',
      'real-webgpu-compute': '../../src/compute/webgpu/real-webgpu-compute.js'
    },
    webnn: {
      'RealWebNNJobs': '../../src/compute/webnn/RealWebNNJobs.js',
      'webnn-worker-simple': '../../src/compute/webnn/webnn-worker-simple.js',
      'webnn-worker': '../../src/compute/webnn/webnn-worker.js'
    },
    wasm: {
      'RealWASMJobs': '../../src/compute/wasm/RealWASMJobs.js',
      'wasm-worker-simple-real': '../../src/compute/wasm/wasm-worker-simple-real.js',
      'wasm-worker-simple': '../../src/compute/wasm/wasm-worker-simple.js',
      'real-wasm-modules': '../../src/compute/wasm/real-wasm-modules.js'
    }
  },

  // Test Files (from perspective of demos folder)
  tests: {
    e2e: '../../tests/integration/e2e/',
    unit: '../../tests/unit/',
    performance: '../../tests/performance/'
  },

  // Assets
  assets: {
    animations: '../../assets/animations/',
    models: '../../assets/models/',
    audio: '../../assets/audio/',
    textures: '../../assets/textures/'
  },

  // Tools
  tools: {
    'serve_with_headers': '../../tools/serve_with_headers.py'
  }
};

// Helper function to get import path
export function getImportPath(category, subcategory, moduleName) {
  if (subcategory) {
    return ImportMap[category]?.[subcategory]?.[moduleName];
  }
  return ImportMap[category]?.[moduleName];
}

// Example usage:
// import { TaskManager } from getImportPath('core', null, 'TaskManager');
// import { RealJobFactory } from getImportPath('ai', 'jobs', 'RealJobFactory');
