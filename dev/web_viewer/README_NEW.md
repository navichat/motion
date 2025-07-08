# RSMT Motion Visualization Showcase

This directory contains web-based visualization tools for the RSMT (Real-time Stylized Motion Transition) project using the 100STYLE dataset.

## ⚠️ Important Note About Transitions

**The current showcase demonstrates BVH motion playback and basic interpolation, NOT the actual RSMT neural network system described in the paper.**

### What This Showcase Actually Does:
- ✅ Visualizes authentic 100STYLE dataset animations
- ✅ Shows different motion styles (emotional, character, energy)
- ✅ Provides interactive 3D skeleton visualization
- ⚠️ Uses **basic linear interpolation** for "transitions" (NOT real RSMT)

### What Real RSMT Does:
The actual RSMT system uses sophisticated neural networks:
- **DeepPhase**: Encodes temporal motion patterns on a 2D phase manifold
- **StyleVAE**: Encodes motion style in latent space  
- **TransitionNet**: Generates natural transitions using learned representations

For details, see: [RSMT_TRANSITION_EXPLANATION.md](RSMT_TRANSITION_EXPLANATION.md)

## Available Viewers

### 1. RSMT Showcase (`rsmt_showcase.html`)
**Primary demonstration viewer with style interpolation**
- 9 professional animations from 100STYLE dataset
- Interactive controls for different motion styles
- Basic style blending (simplified transitions)
- Real-time 3D skeleton visualization

**Features:**
- Emotional styles: Neutral, Elated, Angry, Depressed
- Character styles: Proud, Robot, Elderly
- Energy levels: Rushed, Strutting
- Preset transition sequences
- Manual animation selection

### 2. Motion Viewer (`motion_viewer.html`) 
**Original working motion viewer**
- Single animation playback
- Proven BVH parsing and skeleton rendering
- Reference implementation for 3D visualization

### 3. Index Page (`index.html`)
**Landing page with viewer descriptions and navigation**

## Quick Start

1. **Start HTTP Server:**
   ```bash
   cd /path/to/web_viewer
   python3 -m http.server 8080
   ```

2. **Open in Browser:**
   - Main showcase: `http://localhost:8080/rsmt_showcase.html`
   - Index page: `http://localhost:8080/index.html`

3. **Try the Demo:**
   - Click "Preload All Animations" first
   - Try different animation styles
   - Use preset sequences for style progression

## Animation Catalog

| Animation | Duration | Style | Description |
|-----------|----------|-------|-------------|
| `neutral_reference.bvh` | 92s | Emotional | Natural walking baseline |
| `elated_reference.bvh` | 62s | Emotional | Happy, bouncy movement |
| `angry_reference.bvh` | 58s | Emotional | Aggressive, tense motion |
| `depressed_reference.bvh` | 113s | Emotional | Slow, heavy movements |
| `proud_reference.bvh` | 77s | Character | Confident, upright posture |
| `robot_reference.bvh` | 189s | Character | Mechanical, precise movements |
| `old_reference.bvh` | 187s | Character | Careful, deliberate motion |
| `rushed_reference.bvh` | 43s | Energy | Fast, hurried movements |
| `strutting_reference.bvh` | 105s | Energy | Confident, rhythmic walk |

All animations use forward walking (FW) variants from the 100STYLE dataset for natural motion flow.

## Technical Implementation

### BVH Processing
- **Parser**: Custom JavaScript BVH parser with error handling
- **Skeleton**: Fixed 72-channel mapping to match 100STYLE format
- **Rendering**: Three.js-based 3D visualization
- **Animation**: Real-time frame interpolation and playback

### Style Interpolation (Current)
```javascript
// NOTE: This is simplified interpolation, NOT real RSMT!
function createTransitionFrames(fromBvh, toBvh) {
    // Linear interpolation between frame data
    // Creates "dragging" motion between positions
    // Missing: phase encoding, style encoding, neural networks
}
```

### What's Missing for Real RSMT
1. **Trained Neural Models**: DeepPhase, StyleVAE, TransitionNet
2. **Phase Processing**: Temporal pattern encoding
3. **Style Encoding**: Latent space representations
4. **Physics Constraints**: Foot contact preservation
5. **Real-time Inference**: GPU-accelerated neural network execution

## Files Structure

```
web_viewer/
├── rsmt_showcase.html          # Main demonstration viewer
├── motion_viewer.html          # Original working viewer  
├── index.html                  # Landing page
├── skeleton_test.html          # Skeleton validation tool
├── README.md                   # This documentation
├── RSMT_TRANSITION_EXPLANATION.md  # Technical explanation
├── three.min.js                # Three.js library
├── *_reference.bvh            # 100STYLE animation files
└── archive/                    # Previous iterations
```

## Development Notes

### Recent Improvements
- ✅ Fixed BVH parsing with comprehensive error handling
- ✅ Implemented preloading system for smooth transitions
- ✅ Added proper skeleton scaling and joint positioning
- ✅ Enhanced UI with progress indicators and error messages
- ✅ Added clear disclaimers about transition accuracy

### Known Limitations
- ⚠️ "Transitions" are basic interpolation, not neural network generated
- ⚠️ No phase information or style encoding
- ⚠️ Missing physics constraints
- ⚠️ Different skeleton structure than original RSMT training data

### Future Improvements
- [ ] Integrate actual RSMT neural network models
- [ ] Add physics-based constraints for realistic motion
- [ ] Implement proper phase-aware transitions
- [ ] Add WebGL-based GPU acceleration for inference
- [ ] Create educational mode explaining RSMT components

## Citation

If referencing this visualization work, please cite the original RSMT paper:

```bibtex
@inproceedings{tang2023rsmt,
  title={RSMT: Real-time Stylized Motion Transition for Characters},
  author={Tang, Xiangjun and Wu, Linjun and Wang, He and Hu, Bo and Gong, Xu and Liao, Yuchen and Li, Songnan and Kou, Qilong and Jin, Xiaogang},
  booktitle={SIGGRAPH '23 Conference Proceedings},
  year={2023},
  publisher={ACM}
}
```

## Questions or Issues

For questions about:
- **This visualization**: Check browser console for detailed logs
- **Real RSMT system**: Refer to the main repository documentation
- **100STYLE dataset**: Visit https://www.ianxmason.com/100style/
