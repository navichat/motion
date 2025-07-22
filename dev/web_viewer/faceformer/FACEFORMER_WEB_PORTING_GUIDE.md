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

### Solution 1: Core Step Export ✅ (RECOMMENDED)
Export only a single autoregressive step, implement the loop in JavaScript:

```python
class FaceformerCoreStep(nn.Module):
    def forward(self, audio_features, vertice_emb, one_hot, template):
        # Single transformer decoder step
        # Returns: new_output, updated_embeddings
```

**Advantages:**
- ✅ ONNX can export single steps
- ✅ Full control over generation in JavaScript
- ✅ Can add stopping conditions
- ✅ Memory efficient

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

### Phase 1: Core Step Approach
1. **Export Core Step Model**
   ```bash
   python export_faceformer_fixed.py
   ```

2. **Implement JavaScript Generator**
   ```javascript
   // See faceformer_web_generator.js
   const generator = new FaceFormerWebGenerator('faceformer_core_step.onnx');
   const vertices = await generator.generateSequence(audioFeatures, template, oneHot);
   ```

3. **Handle Audio Processing**
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

1. **Run the Fixed Export:**
   ```bash
   cd /home/barberb/motion/engine/web_porting_poc
   python export_faceformer_fixed.py
   ```

2. **Test Core Step Model:**
   ```bash
   node faceformer_web_generator.js
   ```

3. **Verify ONNX Model:**
   ```bash
   # Check if model loads without hanging
   python -c "import onnx; model = onnx.load('faceformer_core_step.onnx'); print('Model loaded successfully')"
   ```

## Expected Outcomes

✅ **Success Indicators:**
- Export completes without hanging
- ONNX model loads in JavaScript
- Can generate sequences step by step
- Memory usage remains controlled

❌ **If Still Issues:**
- Simplify further by removing complex attention
- Use teacher-forcing approach instead
- Pre-compute more components offline

## Next Steps

1. Try the core step export approach first
2. If successful, implement the JavaScript generator
3. Add audio preprocessing pipeline
4. Optimize for WebNN/WebGPU deployment

The key insight is that **autoregressive models cannot be directly exported to ONNX** - you must export individual steps and implement the generation loop in the target runtime (JavaScript).
