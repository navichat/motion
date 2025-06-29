# RSMT (Real-time Stylized Motion Transition) Documentation

## Overview

RSMT (Real-time Stylized Motion Transition) is a deep learning system for generating smooth, stylized character motion transitions in real-time. The project implements state-of-the-art techniques for controllable motion synthesis, enabling seamless transitions between different motion styles and types.

## Project Structure

```
RSMT-Realtime-Stylized-Motion-Transition/
├── ReadMe.md                    # Main project documentation
├── LICENSE                      # Project license
├── requirements.txt             # Python dependencies
├── requirements_compat.txt      # Compatibility requirements
├── requirements_updated.txt     # Updated dependencies
├── 
├── # Data Processing Scripts
├── process_dataset.py           # Main dataset processing pipeline
├── preprocess_complete.py       # Complete preprocessing workflow
├── preprocess_steps.py          # Individual preprocessing steps
├── run_preprocess.py           # Preprocessing runner
├── manual_preprocessing.py      # Manual preprocessing tools
├── direct_preprocessing.py      # Direct processing utilities
├── add_phase_to_dataset.py     # Phase vector generation
├── inspect_dataset.py          # Dataset inspection tools
├──
├── # Training Scripts
├── train_deephase.py           # Phase manifold training
├── train_styleVAE.py           # Style VAE training
├── train_sampler.py            # Sampler network training
├──
├── # Debugging and Testing
├── debug.py                    # General debugging tools
├── debug_bvh.py               # BVH debugging utilities
├── advanced_debug.py          # Advanced debugging features
├── hello.py                   # Simple test script
├──
├── # Benchmarking
├── benchmark.py               # Performance benchmarking
├── benchmarkStyle100_withStyle.py  # Style-aware benchmarking
├──
├── # Logs and Outputs
├── debug_output.log           # Debug logging
├── preprocess_log.txt         # Preprocessing logs
├──
├── # Documentation Assets
└── ReadMe.assets/             # Documentation images and diagrams
    ├── phase.png              # Phase manifold visualization
    └── SAFB.png               # Style analysis visualization
```

## Features

### Core Capabilities
- **Real-time Motion Synthesis:** Generate smooth transitions between motion styles
- **Style Preservation:** Maintain character and motion style consistency
- **Phase-aware Processing:** Use motion phase for temporal alignment
- **Multiple Motion Types:** Support for various character animations
- **GPU Acceleration:** Optimized for real-time performance

### Technical Components

#### Phase Manifold
- **Purpose:** Learn temporal structure of motions
- **Architecture:** Deep autoencoder for phase representation
- **Output:** Phase vectors for motion timing alignment
- **Benefits:** Enables smooth temporal transitions

#### Style VAE (Variational Autoencoder)
- **Purpose:** Learn style-aware motion representations
- **Architecture:** Variational autoencoder with style conditioning
- **Capabilities:** Generate motions with specific style characteristics
- **Applications:** Style transfer and motion variation

#### Sampler Network
- **Purpose:** Generate transition sequences between motions
- **Input:** Source and target motion styles, transition length
- **Output:** Smooth transition motion sequences
- **Features:** Real-time generation capabilities

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
matplotlib==3.5.2     # Visualization and plotting
numpy==1.22.3         # Numerical computing
pandas==1.4.3         # Data manipulation
pytorch3d==0.4.0      # 3D deep learning utilities
pytorch_lightning==1.5.10  # PyTorch training framework
scipy==1.9.0          # Scientific computing
torch==1.13.0+cu117   # PyTorch with CUDA support
```

### System Requirements
- **GPU:** CUDA-compatible GPU with 8GB+ VRAM
- **Memory:** 16GB+ RAM recommended
- **Storage:** 50GB+ for datasets and models
- **Python:** 3.8+ recommended

## Dataset Setup

### 100STYLE Dataset
The project uses the 100STYLE dataset for training:

1. **Download:** Get dataset from https://www.ianxmason.com/100style/
2. **Location:** Place in `MotionData/100STYLE/`
3. **Structure:** Should contain BVH motion files and metadata

### Dataset Preprocessing

#### Step 1: Initial Preprocessing
```bash
python process_dataset.py --preprocess
```

This converts BVH files to binary format and performs data augmentation:
- Converts `.bvh` files to binary format
- Creates augmented versions of the dataset
- Generates train/test splits

#### Expected Output Files
```
MotionData/100STYLE/
├── skeleton                    # Skeleton definition
├── test_binary.dat            # Test data (binary)
├── test_binary_agument.dat    # Augmented test data
├── train_binary.dat           # Training data (binary)
└── train_binary_agument.dat   # Augmented training data
```

## Training Pipeline

### Phase 1: Train Phase Manifold

#### Prepare Phase Training Data
```bash
python process_dataset.py --train_phase_model
```

#### Train DeepPhase Model
```bash
python train_deephase.py
```

#### Validate Phase Model
```bash
python train_deephase.py --test --version YOUR_VERSION --epoch YOUR_EPOCH
```

**Output Visualizations:**
- `phase.png` - Phase manifold structure
- `SAFB.png` - Style analysis and feature breakdown

### Phase 2: Generate Phase Vectors

```bash
python process_dataset.py --add_phase_to_dataset --model_path "YOUR_PHASE_MODEL_PATH"
```

This adds phase information to the dataset for improved temporal alignment.

### Phase 3: Train Manifold Model

#### Prepare Manifold Training Data
```bash
python process_dataset.py --train_manifold_model
```

This splits motion sequences into 60-frame windows for manifold training.

#### Train Style VAE
```bash
python train_styleVAE.py
```

#### Validate Manifold Model
```bash
python train_styleVAE.py --test --version YOUR_VERSION --epoch YOUR_EPOCH
```

**Output:**
- Multiple `.bvh` files with `test_net.bvh` showing model-generated motions
- Saved model: `m_save_model_YOUR_EPOCH`

### Phase 4: Train Sampler

#### Prepare Sampler Training Data
```bash
python process_dataset.py --train_sampler_model
```

Note: Sampler uses 120-frame sequences (vs. 60 for manifold).

#### Train Sampler Network
```bash
python train_sampler.py --manifold_model YOUR_MANIFOLD_MODEL
```

Replace `YOUR_MANIFOLD_MODEL` with `m_save_model_YOUR_EPOCH` from previous step.

## Configuration

### Training Parameters

#### Phase Model Settings
- **Epochs:** 100-200 for convergence
- **Learning Rate:** 1e-4 to 1e-3
- **Batch Size:** 32-64 depending on GPU memory
- **Sequence Length:** Variable based on motion type

#### Style VAE Settings
- **Latent Dimensions:** 128-512 for style representation
- **Beta Value:** 0.1-1.0 for KL divergence weighting
- **Window Size:** 60 frames for temporal consistency

#### Sampler Settings
- **Input Length:** 120 frames for transition generation
- **Output Length:** Configurable based on application needs
- **Style Conditioning:** Multiple style inputs supported

### Model Architecture

#### DeepPhase Architecture
```
Input: Motion sequence (joint positions/rotations)
├── Encoder: LSTM + FC layers
├── Phase Manifold: 2D circular representation
└── Decoder: FC + LSTM layers
Output: Reconstructed motion with phase
```

#### Style VAE Architecture
```
Input: Motion window + style label
├── Encoder: Conv1D + FC → μ, σ
├── Sampling: Reparameterization trick
├── Decoder: FC + Conv1D
└── Style Conditioning: Style embedding integration
Output: Style-aware motion representation
```

## Usage Examples

### Real-time Motion Generation

```python
# Load trained models
phase_model = load_phase_model('path/to/phase_model.pth')
manifold_model = load_manifold_model('path/to/manifold_model.pth')
sampler_model = load_sampler_model('path/to/sampler_model.pth')

# Generate transition
source_motion = load_motion('source.bvh')
target_style = 'walking'
transition_length = 60

transition = generate_transition(
    source_motion=source_motion,
    target_style=target_style,
    length=transition_length,
    models=(phase_model, manifold_model, sampler_model)
)
```

### Batch Processing

```python
# Process multiple motions
motion_files = ['walk.bvh', 'run.bvh', 'jump.bvh']
styles = ['casual', 'energetic', 'athletic']

for motion_file, style in zip(motion_files, styles):
    processed_motion = process_motion_with_style(
        motion_file, style, models
    )
    save_motion(processed_motion, f'output_{style}.bvh')
```

## Debugging and Analysis

### Debug Tools

#### General Debugging
```bash
python debug.py --motion_file path/to/motion.bvh --visualize
```

#### BVH-specific Debugging
```bash
python debug_bvh.py --input_file motion.bvh --check_skeleton
```

#### Advanced Analysis
```bash
python advanced_debug.py --model_path model.pth --analyze_latent_space
```

### Performance Monitoring

#### Benchmarking
```bash
# General performance
python benchmark.py --model_path model.pth --test_data path/to/test/

# Style-specific benchmarking
python benchmarkStyle100_withStyle.py --style_list casual,energetic,athletic
```

### Inspection Tools

```bash
# Dataset inspection
python inspect_dataset.py --data_path MotionData/100STYLE/

# Model analysis
python inspect_model.py --model_path model.pth --layer_analysis
```

## Integration with Other Projects

### With BvhToDeepMimic
1. Use converted motions as input to RSMT
2. Apply style transfers to DeepMimic reference motions
3. Generate varied training data for RL

### With pytorch_DeepMimic
1. Use RSMT for generating training motions
2. Apply real-time transitions to trained policies
3. Enhance RL agents with style awareness

### With Chat Interface
1. Provide real-time motion transitions for characters
2. Enable style-based character animation
3. Support interactive motion generation

## Performance Optimization

### Training Optimization
- **Mixed Precision:** Use automatic mixed precision for faster training
- **Gradient Clipping:** Prevent exploding gradients
- **Learning Rate Scheduling:** Adaptive learning rate adjustment
- **Data Loading:** Optimize data pipeline for GPU utilization

### Inference Optimization
- **Model Quantization:** Reduce model size for deployment
- **Batch Processing:** Process multiple requests simultaneously
- **Caching:** Cache frequently used style representations
- **GPU Memory Management:** Optimize memory usage for real-time performance

## Quality Metrics

### Motion Quality
- **Smoothness:** Measure transition smoothness via acceleration analysis
- **Style Consistency:** Evaluate style preservation through motion descriptors
- **Temporal Coherence:** Assess phase alignment and timing consistency
- **Realism:** Compare with ground truth motion data

### Performance Metrics
- **Generation Speed:** Frames per second for real-time applications
- **Memory Usage:** RAM and VRAM consumption
- **Model Size:** Storage requirements for deployment
- **Latency:** End-to-end processing time

## Troubleshooting

### Common Issues

#### Training Problems
1. **Convergence Issues:**
   - Reduce learning rate
   - Increase batch size
   - Check data preprocessing

2. **Memory Errors:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

3. **Style Transfer Quality:**
   - Verify phase alignment
   - Adjust style conditioning weights
   - Increase model capacity

#### Runtime Issues
1. **Slow Inference:**
   - Optimize model architecture
   - Use model quantization
   - Enable GPU acceleration

2. **Poor Motion Quality:**
   - Check input data quality
   - Verify model convergence
   - Adjust sampling parameters

### Debug Logs
Check log files for detailed error information:
- `debug_output.log` - Runtime debugging
- `preprocess_log.txt` - Data preprocessing logs
- Training logs in model output directories

## Advanced Features

### Custom Style Definition
```python
# Define custom motion styles
custom_style = {
    'name': 'aggressive',
    'features': {
        'velocity_scale': 1.5,
        'joint_stiffness': 0.8,
        'smoothness': 0.3
    }
}
```

### Multi-character Support
- Different skeleton structures
- Character-specific style adaptations
- Cross-character motion transfer

### Interactive Control
- Real-time style adjustment
- Interactive transition control
- User-guided motion generation

## Research Applications

### Motion Synthesis Research
- Novel transition algorithms
- Style transfer techniques
- Temporal motion modeling

### Animation Industry
- Artist-friendly tools
- Automated motion generation
- Style-consistent character animation

### Gaming Applications
- Real-time character control
- Procedural animation systems
- Adaptive motion responses

## Future Developments

### Planned Features
- Multi-style blending
- Hierarchical motion models
- Real-time interaction support
- Mobile deployment optimization

### Research Directions
- Improved phase representations
- Enhanced style control
- Cross-modal motion generation
- Uncertainty quantification

## Contributing

### Development Setup
```bash
git clone [repository]
cd RSMT-Realtime-Stylized-Motion-Transition
pip install -r requirements.txt
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include comprehensive docstrings
- Add unit tests for new features

### Testing
```bash
# Run test suite
python -m pytest tests/
python test_models.py
python test_preprocessing.py
```

## License

See LICENSE file for project licensing terms.

## Citation

If using this work in research, please cite the appropriate papers:
- "RSMT: Real-time Stylized Motion Transition for Characters"
- "Real-time Controllable Motion Transition for Characters"

## Support

- GitHub Issues for bug reports
- Documentation for usage questions
- Research forums for academic discussions
