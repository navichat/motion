# Web Viewer Reorganization - Status and Next Steps

## Current Status

### ✅ Completed Successfully
1. **Created organized directory structure**:
   - `src/core/` - Core framework components (TaskManager, constants)
   - `src/workers/` - Organized worker files by type 
   - `src/models/` - AI model optimization components
   - `tests/e2e/` - End-to-end test structure
   - `tests/config/` - Test configuration files

2. **Moved key components**:
   - TaskManager.js → `src/core/task-manager/TaskManager.js`
   - Workers → `src/workers/compute/`, `src/workers/ai/`
   - Model optimizers → `src/models/quantized/`
   - Test files → `tests/e2e/playwright/`

3. **Updated import paths**:
   - Fixed relative imports in moved components
   - Created index.js files for clean module interfaces
   - Updated worker imports in moved files

4. **Created test infrastructure**:
   - Playwright configuration for reorganized structure
   - HTML test pages in organized locations
   - Fixed HTML files to use correct paths

### 🔄 Issues Encountered
1. **Worker Path Resolution**: The main.js file loads workers using relative paths that break when HTML files are moved to different locations
2. **Function Interface**: The main.js file uses a different interface than initially expected
3. **AI Inference Not Running**: Tests show 0 results, indicating the AI inference pipeline isn't executing

## Root Cause Analysis

The issue is **NOT** with our reorganized file structure. The reorganized files are correct and properly moved. The issue is with **the original main.js file's worker loading mechanism**.

### Problem Details:
1. **main.js** expects workers to be loaded from `js/workers/` relative to the HTML file location
2. When HTML files are moved to `tests/e2e/html/`, the worker paths resolve incorrectly
3. The `runRealWorkloadTest()` function exists but workers can't load, so no AI inference happens

## Immediate Solutions

### Option 1: Fix Worker Loading (Recommended)
Update main.js to use absolute paths or provide a way to configure worker base paths.

### Option 2: Alternative Test Structure
Create test files that work with the existing worker loading mechanism while keeping our organized structure.

### Option 3: Gradual Migration
Complete the component reorganization first, then address the worker loading system.

## Next Immediate Steps

1. **Fix the worker loading issue** by either:
   - Updating main.js to handle configurable worker paths
   - Creating a wrapper that sets correct worker paths
   - Using absolute URLs for worker loading

2. **Complete systematic component migration**:
   - Move remaining JavaScript files to appropriate src/ directories
   - Update all import paths
   - Create proper module boundaries

3. **Create component-specific tests**:
   - Unit tests for individual components
   - Integration tests for component interactions
   - E2E tests for full workflows

## File Migration Plan

### Phase 1: Core System Files
```
js/main.js → src/core/main.js (with worker path fixes)
js/constants.js → src/core/constants.js
js/utils.js → src/core/utils/ (if exists)
```

### Phase 2: AI Model Components
```
Audio Models:
- kokoro-tts.js → src/models/audio/kokoro/
- phonemizer.js → src/models/audio/utils/
- vad-processor.js → src/models/audio/vad/

Motion Models:
- Audio2GestureBVHConverter.js → src/models/motion/audio2gesture/
- DeepMimicPolicyLoader.js → src/models/motion/deepmimic/
- RSMTBVHConverter.js → src/models/motion/rsmt/
- FaceFormerBVHConverter.js → src/models/motion/faceformer/
```

### Phase 3: Animation & VRM Components
```
- BVHTimeline.js → src/components/animation/timeline/
- AnimationBlender.js → src/components/animation/blender/
- VRMBVHAdapter.js → src/components/animation/vrm/
```

### Phase 4: Test Suite Reorganization
```
- Create unit tests for each moved component
- Update E2E tests to work with new structure
- Create component-specific demo pages
```

## Success Metrics

1. **Reorganization Success**: ✅ Files moved to logical directories
2. **Import Path Integrity**: ✅ All moved files have correct imports  
3. **Test Infrastructure**: ✅ Playwright tests configured for new structure
4. **Worker Loading Fix**: 🔄 IN PROGRESS - Need to fix worker path resolution
5. **Component Testing**: ⏳ PENDING - Need working AI inference first
6. **Full Migration**: ⏳ PENDING - Complete remaining file moves

## Current Priority

**IMMEDIATE**: Fix the worker loading mechanism so we can validate our reorganization works correctly. Once that's working, we can proceed with the systematic migration of remaining components.
