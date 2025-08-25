#!/usr/bin/env node

/**
 * Import Path Update Tool
 * 
 * Updates import paths in HTML and JavaScript files to reflect the new organized structure.
 * Handles the reorganization from scattered root files to component-based directories.
 */

import fs from 'fs';
import path from 'path';
import globPkg from 'glob';
const { glob } = globPkg;

// Mapping of old paths to new organized paths
const pathMappings = {
  // Core TaskManager and utilities
  './js/TaskManager.js': './src/core/TaskManager.js',
  './js/FibonacciHeap.js': './src/core/FibonacciHeap.js',
  './js/BaseJob.js': './src/core/BaseJob.js',
  './js/main.js': './src/index.js',
  
  // AI Model Components
  './js/AIModelJobs.js': './src/ai/AIModelJobs.js',
  './js/AIModelJobFactory.js': './src/ai/AIModelJobFactory.js',
  'DeepMimicPolicyLoader.js': './src/ai/DeepMimicPolicyLoader.js',
  'onnx-validator.js': './src/ai/onnx-validator.js',
  'onnx-version-manager.js': './src/ai/onnx-version-manager.js',
  
  // Avatar Components
  'Audio2GestureBVHConverter.js': './src/avatar/motion/Audio2GestureBVHConverter.js',
  'RSMTBVHConverter.js': './src/avatar/motion/RSMTBVHConverter.js',
  'DeepMimicVRMBoneMapper.js': './src/avatar/vrm/DeepMimicVRMBoneMapper.js',
  'VRMConversationInterface.js': './src/avatar/vrm/VRMConversationInterface.js',
  'AnimationBackendsTestSuite.js': './src/avatar/animation/AnimationBackendsTestSuite.js',
  'BVHAnimationTestSuite.js': './src/avatar/animation/BVHAnimationTestSuite.js',
  'phase_visualizer.js': './src/avatar/animation/phase_visualizer.js',
  'style_controller.js': './src/avatar/animation/style_controller.js',
  'motion_analyzer.js': './src/avatar/motion/motion_analyzer.js',
  'motion_capture.js': './src/avatar/motion/motion_capture.js',
  'PathfindingBVHPlanner.js': './src/avatar/motion/PathfindingBVHPlanner.js',
  'PathfindingTimelineIntegration.js': './src/avatar/motion/PathfindingTimelineIntegration.js',
  'rsmt_client.js': './src/avatar/motion/rsmt_client.js',
  
  // Audio Components
  'kokoro.web.js': './src/audio/kokoro.web.js',
  'test_phonemizer.js': './src/audio/test_phonemizer.js',
  'voiceChatExample.js': './src/audio/voiceChatExample.js',
  'workerVoiceChatExample.js': './src/audio/workerVoiceChatExample.js',
  
  // Compute Backend Components
  'MockBackends.js': './src/compute/MockBackends.js',
  './js/jobs/WASMJobs.js': './src/compute/wasm/WASMJobs.js',
  './js/jobs/WebGPUJobs.js': './src/compute/webgpu/WebGPUJobs.js',
  './js/jobs/WebNNJobs.js': './src/compute/webnn/WebNNJobs.js',
  './js/RealWASMJobs.js': './src/compute/wasm/RealWASMJobs.js',
  './js/RealWebGPUJobs.js': './src/compute/webgpu/RealWebGPUJobs.js',
  './js/RealWebNNJobs.js': './src/compute/webnn/RealWebNNJobs.js',
  
  // KNN and Vector Search
  './js/KNNJobs.js': './src/ai/KNNJobs.js',
  
  // Testing and Utilities
  './js/MockGPUJobs.js': './src/testing/MockGPUJobs.js',
  './js/RealJobFactory.js': './src/core/RealJobFactory.js',
  './js/TaskManagerTestSuite.js': './src/testing/TaskManagerTestSuite.js',
  './js/EnhancedTaskManagerTest.js': './src/testing/EnhancedTaskManagerTest.js',
  './js/TaskManagerWebGPUTest.js': './src/testing/TaskManagerWebGPUTest.js',
  './js/PerformanceBenchmarkJobs.js': './src/testing/PerformanceBenchmarkJobs.js',
  './js/RealTimePerformanceMonitor.js': './src/utils/RealTimePerformanceMonitor.js',
  './js/SystemPerformanceAnalyzer.js': './src/utils/SystemPerformanceAnalyzer.js',
  
  // Workers
  './js/workers/onnx-runtime-fixer.js': './src/workers/onnx-runtime-fixer.js',
  
  // Third-party libraries
  'three.min.js': './lib/three.min.js',
  './assets/libraries/three.min.js': './lib/three.min.js'
};

// Additional mappings for relative imports within moved files
const relativePathMappings = {
  // From avatar components to core
  './TaskManager.js': '../../core/TaskManager.js',
  './MockGPUJobs.js': '../../testing/MockGPUJobs.js',
  
  // From audio components
  './modules/VoiceChatInterface.js': './VoiceChatInterface.js',
  './modules/WorkerVoiceChatInterface.js': './WorkerVoiceChatInterface.js',
  './js/phonemizer.js': './phonemizer.js',
  
  // From compute workers
  './model-loader-webnn.js': './model-loader-webnn.js',
  './ai-model-inference-worker.js': './ai-model-inference-worker.js',
  './real-wasm-modules.js': './real-wasm-modules.js'
};

function updateFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let updated = false;
    
    // Update import statements and script src attributes
    for (const [oldPath, newPath] of Object.entries(pathMappings)) {
      // Handle various formats: import, require, script src
      const patterns = [
        new RegExp(`(import.*from\\s+['"])${escapeRegex(oldPath)}(['"])`, 'g'),
        new RegExp(`(require\\s*\\(\\s*['"])${escapeRegex(oldPath)}(['"]\\s*\\))`, 'g'),
        new RegExp(`(src\\s*=\\s*['"])${escapeRegex(oldPath)}(['"])`, 'g'),
        new RegExp(`(importScripts\\s*\\(\\s*['"])${escapeRegex(oldPath)}(['"]\\s*\\))`, 'g')
      ];
      
      patterns.forEach(pattern => {
        if (pattern.test(content)) {
          content = content.replace(pattern, `$1${newPath}$2`);
          updated = true;
        }
      });
    }
    
    // Update relative imports for moved files
    for (const [oldPath, newPath] of Object.entries(relativePathMappings)) {
      const patterns = [
        new RegExp(`(import.*from\\s+['"])${escapeRegex(oldPath)}(['"])`, 'g'),
        new RegExp(`(require\\s*\\(\\s*['"])${escapeRegex(oldPath)}(['"]\\s*\\))`, 'g'),
        new RegExp(`(importScripts\\s*\\(\\s*['"])${escapeRegex(oldPath)}(['"]\\s*\\))`, 'g')
      ];
      
      patterns.forEach(pattern => {
        if (pattern.test(content)) {
          content = content.replace(pattern, `$1${newPath}$2`);
          updated = true;
        }
      });
    }
    
    if (updated) {
      fs.writeFileSync(filePath, content);
      console.log(`✅ Updated: ${filePath}`);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`❌ Error updating ${filePath}:`, error.message);
    return false;
  }
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function main() {
  const webViewerDir = '/home/barberb/motion/dev/web_viewer';
  
  console.log('🔧 Starting import path updates...');
  
  // Find all HTML and JS files (excluding node_modules and lib)
  const htmlFiles = glob.sync(`${webViewerDir}/**/*.html`, {
    ignore: [`${webViewerDir}/node_modules/**`, `${webViewerDir}/lib/**`, `${webViewerDir}/kokoro.js/**`]
  });
  
  const jsFiles = glob.sync(`${webViewerDir}/**/*.js`, {
    ignore: [`${webViewerDir}/node_modules/**`, `${webViewerDir}/lib/**`, `${webViewerDir}/kokoro.js/**`]
  });
  
  let updatedCount = 0;
  
  // Update HTML files
  console.log(`📄 Processing ${htmlFiles.length} HTML files...`);
  htmlFiles.forEach(file => {
    if (updateFile(file)) updatedCount++;
  });
  
  // Update JS files
  console.log(`📜 Processing ${jsFiles.length} JavaScript files...`);
  jsFiles.forEach(file => {
    if (updateFile(file)) updatedCount++;
  });
  
  console.log(`\n🎉 Import path update complete!`);
  console.log(`   Updated ${updatedCount} files`);
  console.log(`   Processed ${htmlFiles.length + jsFiles.length} total files`);
}

// Run the main function as async
main().catch(console.error);

export { updateFile, pathMappings };
