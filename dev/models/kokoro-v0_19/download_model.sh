#!/bin/bash

# Download Kokoro TTS Model
echo "🎵 Downloading Kokoro TTS Model..."

MODEL_DIR="/home/barberb/motion/dev/models/kokoro-v0_19"
cd "$MODEL_DIR"

# Download model files
echo "📥 Downloading model.onnx..."
wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/model.onnx

echo "📥 Downloading tokenizer.json..."
wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/tokenizer.json

echo "📥 Downloading config.json..."
wget -q --show-progress https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json

echo "✅ Kokoro TTS model download complete!"
echo "🎯 Files downloaded to: $MODEL_DIR"
ls -la "$MODEL_DIR"

echo "🔄 Restart the conversation worker to use the new model."
