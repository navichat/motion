# BvhToDeepMimic Project Documentation

## Overview

BvhToDeepMimic is a Python library that converts BVH (Biovision Hierarchy) motion capture files into DeepMimic-compatible motion files. This enables the use of custom reference motions for training DeepMimic reinforcement learning agents.

## Project Structure

```
BvhToDeepMimic/
├── bvhtomimic.py              # Main conversion script
├── example_script.py          # Example usage script
├── setup.py                   # Package installation configuration
├── README.md                  # Project documentation
├── LICENSE.md                 # MIT License
├── Assets/                    # Example GIFs and media
│   ├── SpeedVault_example.gif
│   └── walking_example.gif
├── bvhtodeepmimic/           # Main package directory
│   ├── __init__.py
│   ├── bvh_extended.py       # Extended BVH parsing
│   ├── bvh_joint_handler.py  # Joint processing logic
│   ├── bvh_joint.py         # Joint data structures
│   ├── joint_info.py        # Joint information management
│   ├── test/                 # Test utilities
│   └── tests/               # Unit tests and test data
│       ├── 0005_Walking001.bvh    # Sample BVH file
│       ├── 0005_Walking001.json   # Sample output
│       ├── test_bvh_joint_handler.py
│       └── test_bvhjoint.py
└── Settings/
    └── settings.json         # Joint mapping configuration
```

## Features

### Core Functionality
- **BVH Parsing:** Reads and parses BVH motion capture files
- **Joint Mapping:** Maps BVH skeleton to DeepMimic humanoid model
- **Scale Transformation:** Adjusts motion scale to match target character
- **Root Motion Handling:** Processes root joint rotation and translation
- **Motion Looping:** Optional loop point detection and processing

### Supported Formats
- **Input:** BVH (Biovision Hierarchy) files
- **Output:** DeepMimic motion text files
- **Compatibility:** SFU Motion Capture Database and standard BVH formats

## Installation

### Via PyPI
```bash
pip install bvhtodeepmimic
```

### From Source
```bash
git clone [repository_url]
cd BvhToDeepMimic
pip install -e .
```

### Dependencies
- Python 3.6 or 3.7
- pyquaternion
- numpy
- bvh
- tqdm

## Usage

### Basic Usage

```python
from bvhtomimic import BvhConverter

# Create converter with settings
converter = BvhConverter("./Settings/settings.json")

# Convert BVH file to DeepMimic format
converter.convertBvhFile("path/to/file.bvh", loop=False)

# Save directly to file
converter.writeDeepMimicFile("input.bvh", "output.txt")
```

### Example Script Usage

```bash
# Place BVH files in InputBvh/ directory
python example_script.py
# Output files will be in OutputMimic/ directory
```

### Settings Configuration

The `Settings/settings.json` file contains:

```json
{
  "scale": 0.01,
  "rootJoints": ["hip", "spine"],
  "jointMapping": {
    "hip": "root",
    "leftHip": "left_hip",
    "rightHip": "right_hip",
    // ... additional joint mappings
  }
}
```

## Configuration

### Joint Mapping
- Map BVH bone names to DeepMimic joint names
- Critical for proper motion transfer
- Must cover all required joints for humanoid model

### Scale Settings
- Adjust motion scale to match target character size
- Typical values: 0.01 - 1.0 depending on source data units

### Root Joint Configuration
- Define which joints determine root orientation
- Usually includes hip and spine joints
- Affects character grounding and movement

## Testing

### Running Tests
```bash
cd bvhtodeepmimic
python -m pytest tests/
```

### Test Files
- Sample BVH files for validation
- Expected output comparisons
- Joint handler unit tests

## Examples and Results

### Supported Motions
- Walking and running gaits
- Athletic movements (vault, jump)
- Dance and expressive motions
- Martial arts sequences

### Output Quality
- Smooth motion transitions
- Preserved timing and style
- Compatible with DeepMimic training pipeline

## Troubleshooting

### Common Issues

1. **Joint Mapping Errors**
   - Verify BVH bone names match settings.json
   - Check for typos in joint names
   - Ensure all required joints are mapped

2. **Scale Problems**
   - Adjust scale factor in settings
   - Check source data units (meters vs. centimeters)
   - Verify character proportions

3. **Root Motion Issues**
   - Review root joint configuration
   - Check for proper grounding
   - Validate rotation handling

### Debug Tips
- Use verbose output for detailed conversion logs
- Validate BVH file structure before conversion
- Test with known working examples first

## Integration with DeepMimic

### Workflow
1. Convert BVH files using this tool
2. Use output files as reference motions in DeepMimic
3. Train imitation policies with converted data
4. Deploy trained models for character animation

### Compatibility
- Works with original DeepMimic (TensorFlow)
- Compatible with pytorch_DeepMimic implementation
- Supports various character configurations

## Performance

### Processing Speed
- Fast conversion for typical motion files
- Batch processing supported
- Progress tracking with tqdm

### Memory Usage
- Efficient memory management for large files
- Streaming processing for very long sequences
- Configurable chunk sizes

## Related Projects

- [DeepMimic](https://github.com/xbpeng/DeepMimic) - Original implementation
- [pytorch_DeepMimic](../pytorch_DeepMimic) - PyTorch version
- [RSMT](../RSMT-Realtime-Stylized-Motion-Transition) - Real-time transitions

## Contributing

### Development Setup
```bash
git clone [repository]
cd BvhToDeepMimic
pip install -e .[dev]
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Include docstrings for all functions

### Testing Requirements
- Add tests for new features
- Maintain high code coverage
- Test with various BVH formats

## License

MIT License - see LICENSE.md for details

## Support

- GitHub Issues for bug reports
- Documentation for usage questions
- Community forums for general discussion
