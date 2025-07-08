# THREE.js Migration Guide

## Issue
The current `three.min.js` file contains deprecated code and shows warnings about being removed in THREE.js r160. This affects all HTML viewers in the project.

## Solution
We have multiple options to fix this:

### Option 1: Replace with Latest THREE.js (Quick Fix)
Replace the current `three.min.js` with a modern version:

```bash
# Download latest stable THREE.js
curl -o three.min.js https://cdn.jsdelivr.net/npm/three@0.156.1/build/three.min.js

# Or use wget
wget https://cdn.jsdelivr.net/npm/three@0.156.1/build/three.min.js -O three.min.js
```

### Option 2: Migrate to ES Modules (Recommended)
Modern web development uses ES modules instead of global scripts. This provides better performance, tree-shaking, and no deprecation warnings.

#### Example Migration:
**Old approach:**
```html
<script src="three.min.js"></script>
<script>
    const scene = new THREE.Scene();
</script>
```

**New approach:**
```html
<script type="module">
    import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.156.1/build/three.module.js';
    const scene = new THREE.Scene();
</script>
```

### Option 3: Use CDN (No local files)
Replace local three.min.js references with CDN:

```html
<script src="https://cdn.jsdelivr.net/npm/three@0.156.1/build/three.min.js"></script>
```

## Files Affected
The following files need updating:

### Main Files:
- `rsmt_showcase.html` (main showcase)
- `three_debug.html`
- `three_test_simple.html`
- `simple_motion_test.html`
- `skeleton_test.html`

### Archive Files:
- `archive/animation_test.html`
- `archive/local_motion_viewer.html`
- `archive/known_good_bvh_viewer.html`
- And many others in archive/

## Implementation Status

✅ **Created:** `modern_three_test.html` - Shows ES modules approach
🔄 **In Progress:** Updating main showcase file
📋 **Pending:** Update remaining files

## Testing
After updating, test by opening:
1. `modern_three_test.html` - Should show spinning cube
2. `rsmt_showcase.html` - Should load without deprecation warnings

## Benefits of ES Modules
- No deprecation warnings
- Better performance (tree-shaking)
- Modern development practices
- Future-proof approach
- Better error handling
