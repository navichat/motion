# Kokoro TTS Model

This directory should contain the Kokoro TTS ONNX model files.

## Required Files:
- `model.onnx` - Main TTS model
- `tokenizer.json` - Tokenizer configuration
- `config.json` - Model configuration

## Download Instructions:
The Kokoro TTS model can be downloaded from:
- HuggingFace: https://huggingface.co/hexgrad/Kokoro-82M
- GitHub: https://github.com/hexgrad/Kokoro

## Alternative Models:
If Kokoro is not available, the system will fall back to:
1. Enhanced synthetic voice generation
2. Web Speech API (if available)
3. Simple tone generation

## Setup Script:
Run the following to download the model:
```bash
cd /home/barberb/motion/dev/models/kokoro-v0_19
wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/model.onnx
wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/tokenizer.json
wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json
```
