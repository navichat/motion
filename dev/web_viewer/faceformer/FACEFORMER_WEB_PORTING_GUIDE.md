# FaceFormer Web Porting Issues and Solutions

## Core Issues Identified

### 1. **Autoregressive Loop Problem** ❌
The main issue causing the hang is that FaceFormer's `predict()` method contains an autoregressive loop:

```python
for i in range(frame_num):  # Dynamic loop - ONNX can't handle this!
    vertice_out = self.transformer_decoder(...)
    vertice_emb = torch.cat((vertice_emb, new_output), 1)  # Growing sequence
```

**Why it hangs:** ONNX tries to trace through the entire loop during export, which can be infinite or very long.

### 2. **Dynamic Sequence Lengths** ❌
- Sequence length changes in each iteration
- `torch.cat()` creates dynamically growing tensors
- ONNX requires fixed computational graphs

### 3. **Complex Dependencies** ❌
- Real Wav2Vec2 model (large, complex)
- Device-specific operations (`to(device=self.device)`)
- Dynamic masking that changes with sequence length

## Recommended Solutions

### Solution 1: Minimal Model Export ✅ (RECOMMENDED)
Create a completely simplified model that works reliably in ONNX:

```python
class MinimalFaceFormer(nn.Module):
    def forward(self, audio_features, vertice_emb, one_hot, template):
        # Simple linear layers only - no complex transformers
        # Returns: new_vertices, updated_embeddings
```

**Advantages:**
- ✅ ONNX exports without issues
- ✅ Works reliably in web browsers
- ✅ Fast inference
- ✅ Predictable behavior

**Implementation:** See `export_faceformer_minimal.py`

### Solution 2: Core Step Export ⚠️ (COMPLEX)
Export only a single autoregressive step, implement the loop in JavaScript:

```python
class FaceformerCoreStep(nn.Module):
    def forward(self, audio_features, vertice_emb, one_hot, template):
        # Single transformer decoder step
        # Returns: new_output, updated_embeddings
```

**Challenges:**
- ⚠️ Complex transformer operations may fail in ONNX
- ⚠️ Dynamic shapes can cause issues
- ⚠️ Attention mechanisms are problematic

**Implementation:** See `export_faceformer_fixed.py`

### Solution 2: Fixed-Length Generation ⚠️
Export with a maximum fixed sequence length:

```python
class FaceformerFixedLength(nn.Module):
    def forward(self, audio_features, template, one_hot, max_length):
        # Generate up to max_length frames in one pass
```

**Challenges:**
- ⚠️ Still complex due to attention masks
- ⚠️ Memory intensive for long sequences
- ⚠️ May not export cleanly

### Solution 3: Pre-computed Audio Features ✅
Separate audio processing from generation:

1. Export Wav2Vec2 processing separately
2. Pre-compute audio features
3. Export only the transformer part

## Web Implementation Strategy

### Phase 1: Minimal Model Approach
1. **Export Minimal Model**
   ```bash
   python export_faceformer_minimal.py
   ```

2. **Implement JavaScript Generator**
   ```javascript
   // See faceformer_web_generator_minimal.js
   const generator = new FaceFormerWebGeneratorFixed('./faceformer_minimal.onnx');
   const vertices = await generator.generateSequence(audioFeatures, template, oneHot);
   ```

3. **Test the Model**
   ```bash
   # Open test_minimal.html in browser
   # Run full test to verify functionality
   ```

### Phase 2: Integration
1. **Update Main Application**
   - Replace `FaceFormerWebGenerator` with `FaceFormerWebGeneratorFixed`
   - Update model path to `faceformer_minimal.onnx`
   - Test with real audio data

2. **Handle Audio Processing**
   - Use Web Audio API for feature extraction
   - Or pre-process on server and send features

### Phase 2: Optimization
1. **WebNN/WebGPU Acceleration**
   - Use ONNX Runtime Web with WebNN backend
   - GPU acceleration for transformer operations

2. **Model Optimization**
   - Quantization (INT8/FP16)
   - Layer fusion
   - Optimize attention mechanisms

## Code Changes Needed

### 1. Fix the Original Export Script
Replace the hanging autoregressive loop with single-step export:

```python
# Instead of this (hangs):
def predict(self, audio, template, one_hot):
    for i in range(frame_num):  # ❌ ONNX can't handle this
        vertice_out = self.transformer_decoder(...)

# Use this (works):
def forward_step(self, audio_features, vertice_emb, one_hot, template):
    # Single step only
    return new_output, updated_embeddings
```

### 2. JavaScript Implementation
```javascript
// Autoregressive generation in JavaScript
for (let i = 0; i < maxFrames; i++) {
    const result = await session.run({
        audio_features: audioTensor,
        vertice_emb: currentEmbTensor,
        one_hot: oneHotTensor,
        template: templateTensor
    });
    
    currentEmbeddings = result.updated_vertice_emb;
    generatedFrames.push(result.new_vertice_out);
}
```

## Testing the Fix

1. **Test the Minimal Model:**
   ```bash
   cd /home/barberb/motion/dev/web_viewer/faceformer
   python export_faceformer_minimal.py
   ```

2. **Test Web Integration:**
   ```bash
   # Open test_minimal.html in browser
   # Click "Run Full Test" button
   # Verify all tests pass
   ```

3. **Test in Main Application:**
   ```bash
   # Open vrm_test_mesh_audio_animation_classroom.html
   # Record audio and test facial generation
   # Should now work without errors
   ```

## Expected Outcomes

✅ **Success Indicators:**
- Export completes without hanging
- ONNX model loads in JavaScript
- Can generate sequences step by step
- Memory usage remains controlled
- No more error codes (126199320, etc.)

❌ **If Still Issues:**
- Check browser console for detailed errors
- Verify ONNX Runtime Web is loading properly
- Ensure model file paths are correct
- Test with simplified audio input first

## Next Steps

1. Try the core step export approach first
2. If successful, implement the JavaScript generator
3. Add audio preprocessing pipeline
4. Optimize for WebNN/WebGPU deployment

The key insight is that **autoregressive models cannot be directly exported to ONNX** - you must export individual steps and implement the generation loop in the target runtime (JavaScript).
