# RSMT Neural Network Integration

This directory contains a complete implementation framework for integrating actual RSMT neural networks into the web visualization system.

## 🚀 Quick Start

### Option 1: Full Neural Network Setup
If you have trained RSMT models available:

```bash
# Install server dependencies
pip install -r requirements_server.txt

# Start the neural network server
python rsmt_server.py

# In another terminal, start the web server
python -m http.server 8080

# Visit: http://localhost:8080/rsmt_showcase.html
```

### Option 2: Demo Mode (Enhanced Interpolation)
If you don't have trained models:

```bash
# Just start the web server
python -m http.server 8080

# Visit: http://localhost:8080/rsmt_showcase.html
# The system will automatically use enhanced interpolation
```

## 🧠 Architecture Overview

### Components

1. **`rsmt_server.py`** - FastAPI backend for neural network inference
   - Loads trained DeepPhase, StyleVAE, and TransitionNet models
   - Provides REST API for motion encoding and transition generation
   - Falls back to enhanced interpolation when models unavailable

2. **`rsmt_client.js`** - Frontend client for neural network communication
   - Handles server communication and fallback strategies
   - Provides status indicators and quality metrics
   - Gracefully degrades when server unavailable

3. **`rsmt_showcase.html`** - Enhanced web viewer
   - Integrates neural network client
   - Shows real-time connection status
   - Displays quality metrics and processing times

### Neural Network Pipeline

```
Input Motion → DeepPhase → Phase Encoding (2D manifold)
              ↓
Input Motion → StyleVAE → Style Encoding (256D vector)
              ↓
Phase + Style → TransitionNet → Natural Transition
```

## 📊 Status Indicators

The web interface shows real-time status:

- 🔴 **Disconnected**: Using basic interpolation fallback
- 🟢 **Connected (Mock)**: Server running but no trained models
- 🔵 **Neural Active**: Real neural networks loaded and running

## 🔧 Model Loading

The server automatically searches for trained models in:
- `../results/*DeepPhase*/myResults/*/model_*.pth`
- `../results/*StyleVAE*/myResults/*/m_save_model_*`
- `../results/*Transition*/myResults/*/m_save_model_*`

Place your trained model checkpoints in these locations for automatic discovery.

## 📡 API Endpoints

### Health Check
- `GET /` - Server status
- `GET /status` - Model loading status

### Neural Network Operations
- `POST /api/encode_phase` - DeepPhase motion encoding
- `POST /api/encode_style` - StyleVAE style encoding  
- `POST /api/generate_transition` - Full RSMT transition generation

### API Example

```javascript
// Generate transition using neural networks
const result = await fetch('http://localhost:8000/api/generate_transition', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        source_motion: { frames: sourceFrames, frame_time: 0.016667 },
        target_motion: { frames: targetFrames, frame_time: 0.016667 },
        transition_length: 60,
        style_blend_curve: 'smooth'
    })
});

const transition = await result.json();
console.log('Quality:', transition.quality_metrics);
```

## 🎯 Quality Metrics

The system provides real-time quality assessment:

- **Smoothness**: Motion continuity and jerk reduction
- **Style Preservation**: How well the transition maintains source/target styles
- **Foot Skating**: Ground contact violation detection
- **Overall Quality**: Combined quality score (0-1)

## 🔄 Fallback Strategy

The system gracefully handles different scenarios:

1. **Neural Networks Available**: Uses real RSMT pipeline
2. **Server Available (No Models)**: Enhanced interpolation with style awareness
3. **Server Unavailable**: Basic interpolation with quality feedback

## 🛠️ Development

### Adding New Models

1. Place model checkpoints in the expected directories
2. Restart the server to auto-discover new models
3. Check `/status` endpoint to verify loading

### Extending the API

Add new endpoints to `rsmt_server.py`:

```python
@app.post("/api/your_endpoint")
async def your_function(request: YourRequest):
    # Your neural network code here
    return YourResponse(...)
```

### Client Integration

Use the JavaScript client in your own projects:

```javascript
const client = new RSMTClient('http://your-server:8000');
const transition = await client.generateTransition(source, target);
```

## 📈 Performance

### Optimization Tips

1. **GPU Acceleration**: Install PyTorch with CUDA support
2. **Model Quantization**: Use INT8 models for faster inference
3. **Batch Processing**: Process multiple transitions together
4. **Caching**: Cache style encodings for repeated use

### Expected Performance

- **CPU**: ~100-500ms per transition
- **GPU**: ~10-50ms per transition
- **Memory**: ~1-4GB depending on models

## 🐛 Troubleshooting

### Common Issues

1. **Server Won't Start**
   ```bash
   pip install --upgrade fastapi uvicorn torch
   ```

2. **Models Not Loading**
   - Check file paths and permissions
   - Verify PyTorch version compatibility
   - Check server logs for detailed errors

3. **CORS Issues**
   - Server runs on all interfaces (0.0.0.0)
   - CORS middleware allows all origins
   - Use same protocol (http/https) for client and server

4. **Performance Issues**
   - Monitor GPU memory usage
   - Consider model quantization
   - Use smaller transition lengths for faster processing

### Debug Mode

Start server with debug logging:
```bash
python rsmt_server.py --log-level debug
```

## 🔮 Future Enhancements

- **Real-time Phase Visualization**: 2D manifold display
- **Style Space Exploration**: Interactive style interpolation
- **Quality Prediction**: Pre-transition quality estimation
- **Physics Constraints**: Foot contact preservation
- **Style Transfer**: Apply motion style to new animations

## 📚 References

- [Original RSMT Paper](https://arxiv.org/abs/2303.12678)
- [100STYLE Dataset](https://www.ianxmason.com/style/)
- [Implementation Plan](RSMT_IMPLEMENTATION_PLAN.md)
- [Technical Explanation](RSMT_TRANSITION_EXPLANATION.md)

---

This integration framework provides a complete pathway from basic motion interpolation to full neural network-powered RSMT transitions, maintaining compatibility and graceful degradation throughout the upgrade process.
