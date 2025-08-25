#!/bin/bash

# Download Kokoro TTS Model
echo "🎵 Downloading Kokoro TTS Model..."

MODEL_DIR="/home/barberb/motion/dev/models/kokoro-v0_19"
cd "$MODEL_DIR"

# Download model files (PyTorch format)
echo "📥 Downloading kokoro-v1_0.pth..."
wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth

echo "📥 Downloading config.json..."
wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json

echo "📥 Downloading voices info..."
wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/VOICES.md

echo "✅ Kokoro TTS model download complete!"
echo "🎯 Files downloaded to: $MODEL_DIR"
ls -la "$MODEL_DIR"

echo "⚠️  Note: This is a PyTorch model (.pth), not ONNX."
echo "🔄 The worker will use enhanced fallback TTS until ONNX conversion is available."
