# PyTorch DeepMimic Project Documentation

## Overview

PyTorch DeepMimic is a complete reimplementation of the original DeepMimic reinforcement learning system, translated from TensorFlow 1.15 to PyTorch 1.12. This project enables training of physics-based character controllers that can imitate a wide variety of reference motions.

## Project Structure

```
pytorch_DeepMimic/
├── setup.py                    # Package installation
├── README.md                   # Project documentation
├── LICENSE                     # License file
└── deepmimic/                  # Main package directory
    ├── __init__.py
    ├── DeepMimic_Optimizer.py  # Main training script
    ├── testrl.py               # Inference and testing
    ├── _pybullet_data/         # Physics simulation assets
    │   ├── ball.vtk, block.urdf, bunny.obj
    │   ├── cartpole.urdf, cloth_z_up.*
    │   ├── checker_*.png/jpg/gif
    │   └── [various simulation assets]
    ├── _pybullet_env/          # Environment definitions
    ├── _pybullet_utils/        # Utility functions
    ├── envs/                   # Environment configurations
    ├── keep/                   # Model checkpoints and saves
    ├── output/                 # Training outputs and logs
    └── test/                   # Test configurations
```

## Features

### Core Capabilities
- **Motion Imitation:** Learn to imitate reference motions through RL
- **Physics Simulation:** PyBullet-based physics for realistic character movement
- **Multiple Characters:** Support for humanoid and other character types
- **Policy Training:** Actor-critic networks with PPO algorithm
- **Real-time Inference:** Fast policy execution for interactive applications

### Technical Features
- **PyTorch Backend:** Modern deep learning framework with GPU acceleration
- **PPO Algorithm:** Proximal Policy Optimization for stable training
- **Multi-threading:** MPI support for distributed training
- **Flexible Architecture:** Configurable network architectures and hyperparameters

## Installation

### Prerequisites
```bash
# Required Python packages
pip install gym==0.23.1
pip install pybullet==3.2.5
pip install mpi4py==3.1.4
pip install pytorch==1.12.1
pip install numpy==1.25.2
```

### Installation Steps
```bash
git clone https://github.com/myiKim/pytorch_DeepMimic.git
cd pytorch_DeepMimic
pip install -e .
```

### Dependencies
- **gym:** 0.23.1 - OpenAI Gym environment interface
- **pybullet:** 3.2.5 - Physics simulation engine
- **mpi4py:** 3.1.4 - Message Passing Interface for parallel computing
- **pytorch:** 1.12.1 - Deep learning framework
- **numpy:** 1.25.2 - Numerical computing library

## Usage

### Training

#### Basic Training
```bash
cd deepmimic
python DeepMimic_Optimizer.py --arg_file train_humanoid3d_walk_args.txt
```

#### Training Configuration Files
- `train_humanoid3d_walk_args.txt` - Walking motion training
- `train_humanoid3d_run_args.txt` - Running motion training
- `train_humanoid3d_jump_args.txt` - Jumping motion training

### Inference

#### Running Trained Models
```bash
cd deepmimic
python testrl.py --arg_file run_humanoid3d_walk_args.txt
```

#### Model Files
After training, models are saved in the `output/` directory:
- `agent0_model_anet.pth` - Actor network (policy)
- `agent0_model_cnet.pth` - Critic network (value function)

## Architecture

### Network Components

#### Actor Network
- **Purpose:** Policy network that outputs actions
- **Input:** Character state observations
- **Output:** Action distributions (joint torques/angles)
- **Architecture:** Multi-layer perceptron with tanh activations

#### Critic Network
- **Purpose:** Value function estimation
- **Input:** Character state observations
- **Output:** Estimated state value
- **Architecture:** Similar to actor but single output

### Training Algorithm

#### Proximal Policy Optimization (PPO)
- **Objective:** Stable policy gradient updates
- **Clipping:** Prevents large policy changes
- **Value Function:** Baseline for variance reduction
- **Advantage Estimation:** GAE (Generalized Advantage Estimation)

## Configuration

### Argument Files
Training and inference behavior is controlled by argument files:

```txt
# Example training arguments
--env_type humanoid3d
--char_file chars/humanoid3d.txt
--motion_file motions/humanoid3d_walk.txt
--num_workers 4
--int_output_iters 100
--int_save_iters 100
```

### Environment Configuration
- **Character Files:** Define character morphology and constraints
- **Motion Files:** Reference motions for imitation
- **Environment Settings:** Simulation parameters and reward functions

### Hyperparameters
- **Learning Rate:** Typically 1e-4 to 1e-3
- **Batch Size:** 256-1024 samples
- **Discount Factor:** 0.95-0.99
- **GAE Lambda:** 0.95

## Character Types

### Humanoid3D
- **DOF:** 34 degrees of freedom
- **Joints:** Full body articulation
- **Capabilities:** Walking, running, jumping, dancing
- **Physics:** Realistic mass distribution and constraints

### Customization
- Modify character files for different morphologies
- Adjust joint limits and masses
- Create new motion files for custom behaviors

## Motion Files

### Format
DeepMimic motion files contain:
- Frame timing information
- Joint angle trajectories
- Root motion data
- Loop points (if applicable)

### Integration with BvhToDeepMimic
1. Convert BVH files using BvhToDeepMimic tool
2. Use output files as reference motions
3. Configure training with new motion files

## Training Process

### Phases
1. **Initialization:** Load character and motion files
2. **Environment Setup:** Configure physics simulation
3. **Network Initialization:** Create actor and critic networks
4. **Training Loop:** 
   - Collect experience rollouts
   - Compute advantages and returns
   - Update networks with PPO
   - Save checkpoints periodically

### Monitoring
- **Reward Tracking:** Monitor imitation quality
- **Policy Updates:** Track learning progress
- **Model Checkpoints:** Regular saves for resuming training

## Performance Optimization

### Hardware Recommendations
- **CPU:** Multi-core processor for parallel simulation
- **GPU:** CUDA-compatible GPU for neural network training
- **Memory:** 16GB+ RAM for large batch sizes
- **Storage:** SSD for faster data loading

### Training Tips
- **Start Simple:** Begin with basic motions before complex sequences
- **Tune Rewards:** Adjust reward weights for desired behavior
- **Monitor Stability:** Watch for policy collapse or instability
- **Use Checkpoints:** Save frequently to recover from crashes

## Evaluation and Testing

### Metrics
- **Imitation Quality:** How closely the policy follows reference motion
- **Stability:** Ability to maintain balance and recover from perturbations
- **Generalization:** Performance on unseen variations

### Validation
```bash
# Test trained model
python testrl.py --arg_file run_humanoid3d_walk_args.txt --model_file output/agent0_model_anet.pth
```

## Integration with Other Projects

### With BvhToDeepMimic
1. Convert motion capture data to DeepMimic format
2. Train policies on converted motions
3. Deploy trained models for animation

### With RSMT
1. Use trained policies as base controllers
2. Enhance with real-time transition capabilities
3. Combine for comprehensive animation systems

### With Chat Interface
1. Deploy trained models for character animation
2. Provide real-time motion for interactive characters
3. Support multiple character types and behaviors

## Troubleshooting

### Common Issues

#### Training Instability
- **Symptoms:** Reward collapse, NaN values
- **Solutions:** Reduce learning rate, adjust reward scaling
- **Prevention:** Monitor training curves, use stable hyperparameters

#### Physics Simulation Problems
- **Symptoms:** Character falls through ground, unrealistic behavior
- **Solutions:** Check character file constraints, adjust simulation timestep
- **Debug:** Enable physics visualization

#### Memory Issues
- **Symptoms:** Out of memory errors during training
- **Solutions:** Reduce batch size, number of workers
- **Optimization:** Use gradient accumulation

### Debug Tools
- **Visualization:** Enable PyBullet GUI for physics debugging
- **Logging:** Increase verbosity for detailed training information
- **Profiling:** Use PyTorch profiler for performance analysis

## Advanced Features

### Custom Rewards
Modify reward functions for specific behaviors:
- Imitation rewards for motion following
- Style rewards for aesthetic quality
- Task rewards for goal completion
- Stability rewards for robust control

### Multi-Skill Training
Train policies on multiple motions:
- Sequential training on different skills
- Simultaneous multi-task learning
- Skill switching and blending

### Transfer Learning
- Pre-train on simpler motions
- Fine-tune for specific characters
- Adapt to new environments

## Research Applications

### Motion Synthesis
- Generate new motions through policy interpolation
- Create variations of reference motions
- Develop style transfer capabilities

### Control Theory
- Study optimal control strategies
- Analyze policy representations
- Investigate learning dynamics

### Animation Tools
- Provide artist-friendly interfaces
- Enable interactive character control
- Support real-time applications

## Contributing

### Development Guidelines
- Follow PyTorch best practices
- Maintain compatibility with original DeepMimic
- Add comprehensive tests for new features
- Document all configuration options

### Testing
```bash
# Run test suite
cd deepmimic/test
python test_environments.py
python test_networks.py
```

## Performance Benchmarks

### Training Speed
- **CPU Only:** 2-4 hours for basic walking motion
- **GPU Accelerated:** 30-60 minutes for basic walking motion
- **Distributed:** Scales with number of workers

### Model Size
- **Actor Network:** ~2-5MB depending on architecture
- **Critic Network:** ~2-5MB depending on architecture
- **Total Storage:** ~10-20MB per trained character

## License

See LICENSE file for details.

## References

- Original DeepMimic paper and implementation
- PyTorch documentation and tutorials
- PyBullet physics simulation documentation
- PPO algorithm papers and implementations

## Support

- GitHub Issues for bug reports
- Documentation for implementation details
- Community forums for research discussions
