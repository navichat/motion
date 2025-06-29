#!/bin/bash

# PyTorch to Mojo/MAX Migration Environment Setup
# This script sets up the development environment for migration

set -e

echo "🚀 Setting up PyTorch to Mojo/MAX Migration Environment"
echo "======================================================="

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the migration_workspace directory"
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {models/{pytorch,onnx,max},data/{preprocessors,samples},validation,deployment/{docker,kubernetes},docs,scripts}

# Check for Python
echo "🐍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check for pip
if ! command -v pip &> /dev/null; then
    echo "❌ pip is required but not installed"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cat > requirements.txt << EOF
torch>=1.12.0
torchvision
onnx>=1.12.0
onnxruntime
numpy>=1.21.0
scipy
matplotlib
tqdm
psutil
fastapi
uvicorn
pydantic
pytest
jupyter
tensorboard
EOF

pip install -r requirements.txt

# Check for Modular installation
echo "🔧 Checking Modular installation..."
if command -v max &> /dev/null; then
    echo "✅ MAX found: $(max --version)"
else
    echo "⚠️  MAX not found. Installing Modular..."
    
    # Install Modular
    if command -v curl &> /dev/null; then
        curl -s https://get.modular.com | sh -
        modular install max
    else
        echo "❌ curl is required to install Modular"
        echo "Please install Modular manually: https://docs.modular.com/max/install"
        exit 1
    fi
fi

if command -v mojo &> /dev/null; then
    echo "✅ Mojo found: $(mojo --version)"
else
    echo "⚠️  Mojo not found. Installing..."
    modular install mojo
fi

# Verify installations
echo "🔍 Verifying installations..."
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch installation failed"
python3 -c "import onnx; print(f'✅ ONNX {onnx.__version__}')" || echo "❌ ONNX installation failed"

if command -v max &> /dev/null; then
    echo "✅ MAX CLI available"
else
    echo "❌ MAX CLI not available"
fi

if command -v mojo &> /dev/null; then
    echo "✅ Mojo compiler available"
else
    echo "❌ Mojo compiler not available"
fi

# Create sample configuration
echo "⚙️  Creating configuration files..."
cat > config.json << EOF
{
    "migration_config": {
        "source_models": {
            "deephase": "../RSMT-Realtime-Stylized-Motion-Transition/output/phase_model/",
            "stylevae": "../RSMT-Realtime-Stylized-Motion-Transition/output/styleVAE/",
            "transitionnet": "../RSMT-Realtime-Stylized-Motion-Transition/output/transitionNet/",
            "deepmimic_actor": "../pytorch_DeepMimic/deepmimic/output/",
            "deepmimic_critic": "../pytorch_DeepMimic/deepmimic/output/"
        },
        "target_paths": {
            "onnx_models": "./models/onnx/",
            "max_models": "./models/max/",
            "validation_data": "./data/samples/"
        },
        "conversion_settings": {
            "onnx_opset_version": 11,
            "max_optimization_level": "O3",
            "batch_size": 1,
            "input_shapes": {
                "deephase": [1, 132],
                "stylevae": [1, 60, 256],
                "transitionnet": [1, 321]
            }
        },
        "validation_settings": {
            "accuracy_threshold": 1e-6,
            "performance_iterations": 1000,
            "test_batch_sizes": [1, 8, 32]
        }
    }
}
EOF

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Models (large files)
models/pytorch/*.pth
models/pytorch/*.pt
models/onnx/*.onnx
models/max/*.maxgraph

# Data
data/samples/*.dat
data/samples/*.bin

# Logs
*.log
logs/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
EOF

# Create initial test data
echo "🧪 Creating test data samples..."
python3 << EOF
import numpy as np
import json
import os

# Create sample motion data for testing
os.makedirs('data/samples', exist_ok=True)

# DeepPhase test data (132-dimensional input)
deephase_sample = np.random.randn(100, 132).astype(np.float32)
np.save('data/samples/deephase_test_input.npy', deephase_sample)

# StyleVAE test data (60 frames, 256 features)
stylevae_sample = np.random.randn(10, 60, 256).astype(np.float32)
np.save('data/samples/stylevae_test_input.npy', stylevae_sample)

# TransitionNet test data
transition_sample = np.random.randn(5, 321).astype(np.float32)
np.save('data/samples/transitionnet_test_input.npy', transition_sample)

# Create metadata
metadata = {
    "deephase": {
        "input_shape": [132],
        "output_shape": [2],
        "description": "Phase encoding model - motion to 2D phase coordinates"
    },
    "stylevae": {
        "input_shape": [60, 256],
        "output_shape": [256],
        "description": "Style VAE model - motion sequence to style vector"
    },
    "transitionnet": {
        "input_shape": [321],
        "output_shape": [63],
        "description": "Transition network - combined input to motion frame"
    }
}

with open('data/samples/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Test data samples created")
EOF

# Create initial documentation
echo "📚 Creating documentation..."
cat > docs/migration_guide.md << EOF
# Step-by-Step Migration Guide

## Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Modular MAX/Mojo installed
- ONNX tools

## Step 1: Model Analysis
\`\`\`bash
python scripts/analyze_models.py
\`\`\`

## Step 2: Export to ONNX
\`\`\`bash
python scripts/migrate_deephase.py --step export
\`\`\`

## Step 3: Convert to MAX
\`\`\`bash
python scripts/migrate_deephase.py --step convert
\`\`\`

## Step 4: Create Mojo Wrapper
\`\`\`bash
python scripts/migrate_deephase.py --step wrap
\`\`\`

## Step 5: Validate
\`\`\`bash
python scripts/validate_migration.py --model deephase
\`\`\`

For detailed information, see the main migration plan.
EOF

cat > docs/troubleshooting.md << EOF
# Troubleshooting Guide

## Common Issues

### ONNX Export Errors
- **Issue**: Model contains unsupported operations
- **Solution**: Use torch.jit.trace instead of torch.onnx.export
- **Alternative**: Implement custom ONNX operators

### MAX Conversion Errors
- **Issue**: ONNX model uses unsupported operations
- **Solution**: Check MAX operation support documentation
- **Workaround**: Split model into supported components

### Performance Issues
- **Issue**: MAX model slower than PyTorch
- **Solution**: Enable MAX optimizations, check batch size
- **Debug**: Use MAX profiling tools

### Accuracy Issues
- **Issue**: Numerical differences between PyTorch and MAX
- **Solution**: Check data types, normalization, random seeds
- **Tolerance**: Adjust accuracy thresholds in validation

## Getting Help
- Check MAX documentation: https://docs.modular.com/max/
- Review ONNX compatibility: https://onnx.ai/
- Contact Modular support for MAX-specific issues
EOF

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run: python scripts/analyze_models.py"
echo "2. Review the analysis output"
echo "3. Start migration with: python scripts/migrate_deephase.py"
echo ""
echo "📖 Documentation:"
echo "- Migration guide: docs/migration_guide.md"
echo "- Troubleshooting: docs/troubleshooting.md"
echo "- Main plan: ../docs/pytorch_to_mojo_migration_plan.md"
echo ""
echo "🎯 Ready to begin PyTorch to Mojo/MAX migration!"
