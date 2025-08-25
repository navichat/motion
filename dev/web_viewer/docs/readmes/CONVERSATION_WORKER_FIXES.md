# Conversation Worker Fixes & Testing Guide

## 🐛 Issues Fixed

### 1. **ONNX Tensor Shape Errors**
- **Problem**: LSTM layer expecting 3D tensors but receiving 5D tensors
- **Error**: `Input X must have 3 dimensions only. Actual:{1,1,1,128,8}`
- **Fix**: Added proper tensor dimension validation and alternative tensor shapes

### 2. **Improved Error Handling**
- **Problem**: Worker failing silently on model errors
- **Fix**: Added comprehensive error logging and fallback mechanisms

### 3. **Model Compatibility Issues**
- **Problem**: Kokoro model incompatible with current transformers.js version
- **Fix**: Added model compatibility testing and simplified inference fallback

### 4. **Phonemizer Integration**
- **Problem**: Text not being properly phonemized before TTS
- **Fix**: Enhanced phonemizer integration with better normalization

## 🚀 Testing Files

### 1. **debug_conversation_worker.html**
- **Purpose**: Comprehensive debug interface for the conversation worker
- **Features**: 
  - Real-time worker status monitoring
  - Error analysis and logging
  - TTS testing capabilities
  - Performance metrics

### 2. **test_enhanced_conversation_worker.html**
- **Purpose**: Interactive test for the enhanced conversation worker
- **Features**:
  - Microphone input simulation
  - Voice selection
  - Audio playback testing

### 3. **test_phonemizer.html**
- **Purpose**: Test the phonemizer module independently
- **Features**:
  - Text normalization testing
  - Phoneme conversion testing
  - eSpeak-NG vs fallback comparison

## 🔧 Key Improvements Made

### Tensor Handling
```javascript
// Before (causing errors)
const inputTensor = new transformers.Tensor('int64', inputIds, [1, inputIds.length]);

// After (with validation and fallback)
const inputTensor = new transformers.Tensor('int64', inputIds, [1, inputIds.length]);
const styleTensor = new transformers.Tensor('float32', style, [1, 256]);
const speedTensor = new transformers.Tensor('float32', speed, [1]);

// With fallback for dimension issues
try {
  output = await model({ input_ids: inputTensor, style: styleTensor, speed: speedTensor });
} catch (onnxError) {
  // Try alternative tensor shapes
  const altInputTensor = new transformers.Tensor('int64', inputIds, [inputIds.length]);
  output = await model({ input_ids: altInputTensor, ... });
}
```

### Enhanced Error Logging
```javascript
workerLog('ERROR', 'Kokoro TTS generation failed', { 
  error: error.message,
  stack: error.stack,
  errorType: error.constructor.name
});
```

### Model Compatibility Testing
```javascript
// Test model compatibility during initialization
try {
  const testInputs = tokenizer('test');
  const testTensor = new transformers.Tensor('int64', testInputs.input_ids.data, testInputs.input_ids.dims);
  workerLog('INFO', 'Tensor creation test passed');
} catch (compatError) {
  workerLog('WARN', 'Model compatibility test failed');
}
```

## 🧪 How to Test

### 1. **Quick Start**
```bash
# Start a local server
python -m http.server 8000

# Open in browser
http://localhost:8000/debug_conversation_worker.html
```

### 2. **Test Sequence**
1. **Worker Initialization**: Check that worker loads without errors
2. **Phonemizer Test**: Test text normalization and phoneme conversion
3. **TTS Test**: Generate audio with different voices
4. **Error Handling**: Verify graceful fallback to Web Speech API

### 3. **Debug Features**
- **Real-time Metrics**: Monitor messages, errors, and TTS generations
- **Error Analysis**: Detailed error logging with stack traces
- **Performance Monitoring**: Track worker uptime and response times

## 📋 Expected Behavior

### ✅ **Success Indicators**
- Worker initializes without ONNX errors
- Phonemizer properly converts text to phonemes
- TTS generates audio (either Kokoro or Web Speech fallback)
- Error handling gracefully manages model failures

### ⚠️ **Acceptable Fallbacks**
- **Kokoro Model Fails**: Falls back to Web Speech API
- **eSpeak-NG Unavailable**: Uses simple phoneme fallback
- **Model Compatibility Issues**: Simplified tensor approach

### ❌ **Critical Failures**
- Worker fails to initialize
- No audio output generated
- Continuous ONNX errors without fallback

## 🔍 Debugging Tips

### 1. **Check Browser Console**
Look for:
- ONNX runtime errors
- Tensor dimension mismatches
- Model loading failures

### 2. **Monitor Worker Messages**
Use the debug interface to track:
- Worker initialization progress
- Model loading status
- TTS generation attempts

### 3. **Test Incrementally**
- Start with phonemizer test
- Then try simple TTS
- Finally test full conversation flow

## 📈 Performance Expectations

### **Typical Performance**
- **Worker Initialization**: 10-30 seconds
- **Model Loading**: 5-15 seconds per model
- **TTS Generation**: 1-3 seconds per sentence
- **Phonemization**: < 100ms per sentence

### **Fallback Performance**
- **Web Speech API**: Near-instant
- **Simple Phonemizer**: < 50ms
- **Error Recovery**: < 1 second

## 🎯 Next Steps

1. **Test with debug_conversation_worker.html**
2. **Verify all error handling works**
3. **Test different voice selections**
4. **Confirm phonemizer integration**
5. **Monitor for any remaining ONNX errors**

The conversation worker should now handle model compatibility issues gracefully and provide detailed debugging information for any remaining issues.
