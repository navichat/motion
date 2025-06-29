# Motion Viewer - Development Environment

## Overview

This is the unified 3D avatar viewer created by migrating the Chat Interface's 3D capabilities to a new development environment. Phase 1 focuses on establishing the foundation, while Phase 2 will integrate RSMT's advanced neural network features.

## Phase 1: Foundation Migration ✅

### Completed Features

- **3D Avatar Viewer**: Extracted and modernized from chat interface
- **VRM Avatar Support**: Loading and display of VRM character models
- **Animation Playback**: JSON format animations from chat interface
- **Environment System**: Classroom, stage, studio, and outdoor environments
- **Interactive Controls**: Play/pause, speed control, timeline scrubbing
- **REST API**: Endpoints for avatars, animations, and environment management

### Architecture

```
dev/
├── viewer/                     # Frontend application
│   ├── src/
│   │   ├── Player.js          # Core 3D player (Three.js)
│   │   ├── AvatarLoader.js    # VRM avatar loading
│   │   ├── AnimationController.js # Animation playback
│   │   ├── SceneManager.js    # Environment management
│   │   └── app.js             # Main application
│   ├── components/
│   │   └── MotionViewer.jsx   # React component
│   ├── styles/
│   │   └── viewer.scss        # Styling
│   └── templates/
│       └── index.html         # Main page
├── server/
│   ├── enhanced_motion_server.py # FastAPI server
│   └── start_dev.sh           # Development startup script
└── assets/
    ├── avatars/               # Character models
    ├── animations/            # Motion data
    └── scenes/                # Environment assets
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (optional, for frontend building)

### Running the Development Server

```bash
cd /home/barberb/motion/dev/server
./start_dev.sh
```

This will:
1. Set up Python virtual environment
2. Install required dependencies  
3. Build the frontend (if Node.js available)
4. Start the development server at http://localhost:8081

### API Endpoints

- `GET /` - Main viewer interface
- `GET /api/avatars` - List available avatars
- `GET /api/animations` - List available animations
- `GET /api/environments` - List available environments
- `GET /api/stats` - System statistics
- `GET /docs` - Interactive API documentation

## Usage Examples

### Basic Integration

```html
<!DOCTYPE html>
<html>
<head>
    <title>Motion Viewer Example</title>
</head>
<body>
    <div id="motion-viewer-container"></div>
    <script type="module" src="/viewer/dist/app.bundle.js"></script>
</body>
</html>
```

### Programmatic Control

```javascript
// Initialize viewer
const viewer = new MotionViewer({
    width: 800,
    height: 600,
    environment: 'classroom',
    showControls: true
});

// Load avatar
await viewer.loadAvatar({
    name: 'Teacher Avatar',
    url: '/assets/avatars/teacher.vrm'
});

// Play animation
await viewer.playAnimation({
    name: 'Explain Gesture',
    file: '/assets/animations/explain_gesture.json',
    format: 'json'
});
```

## Development Guidelines

### Adding New Avatars

1. Place VRM files in `assets/avatars/`
2. Update `assets/avatars/index.json`
3. Include preview images and configuration

```json
{
    "id": "new_avatar",
    "name": "New Avatar",
    "file": "new_avatar.vrm",
    "format": "vrm",
    "description": "Description of the avatar",
    "config": {
        "scale": 1.0,
        "position": [0, 0, 0]
    }
}
```

### Adding New Animations

1. Place JSON animation files in `assets/animations/`
2. Update `assets/animations/index.json`
3. Specify format, duration, and metadata

```json
{
    "id": "new_animation",
    "name": "New Animation",
    "file": "new_animation.json",
    "format": "json",
    "duration": 5.0,
    "description": "Description of the animation",
    "tags": ["gesture", "teaching"]
}
```

### Creating Custom Environments

Extend the `SceneManager` class in `viewer/src/SceneManager.js`:

```javascript
createCustomEnvironment(options = {}) {
    const environment = {
        type: 'custom',
        objects: [],
        lights: [],
        camera: {
            position: new THREE.Vector3(0, 2, 5),
            target: new THREE.Vector3(0, 1, 0)
        }
    };
    
    // Add custom objects, lighting, etc.
    
    return environment;
}
```

## Phase 2: RSMT Integration (Planned)

### Upcoming Features

- **Neural Network Integration**: DeepPhase, StyleVAE, TransitionNet models
- **100STYLE Dataset**: Access to 100+ motion styles
- **Style Transfer**: Real-time motion style transformation
- **Transition Generation**: AI-powered motion transitions
- **WebSocket Streaming**: Real-time motion updates
- **Advanced Controls**: Style interpolation, motion blending

### Migration Roadmap

1. **Neural Model Integration**: Load RSMT PyTorch models
2. **BVH Format Support**: Parse and convert BVH motion data
3. **Style API Endpoints**: REST endpoints for style operations
4. **Real-time Pipeline**: WebSocket for live motion generation
5. **Advanced UI**: Style controllers and transition tools

### Phase 2 Architecture

```
server/
├── enhanced_motion_server.py    # Current FastAPI server
├── rsmt_server_progressive.py   # RSMT neural server (Phase 2)
├── neural_models.py             # Model loading and inference
└── motion_processor.py          # BVH processing and style transfer

viewer/
├── src/
│   ├── MotionConverter.js       # BVH ↔ JSON conversion
│   ├── MotionStreamingClient.js # WebSocket client
│   └── StyleController.js       # Neural style controls
└── components/
    ├── StyleSelector.jsx        # Style selection UI
    └── TransitionDemo.jsx       # Transition demonstration
```

## Technology Stack

### Frontend
- **Framework**: React + Three.js
- **3D Engine**: Three.js R177+
- **Avatar Format**: VRM (via @pixiv/three-vrm)
- **Animation**: JSON keyframes + BVH (Phase 2)
- **Build**: esbuild for fast bundling

### Backend  
- **Server**: FastAPI with async support
- **ML Framework**: PyTorch (Phase 2)
- **Motion Processing**: NumPy, SciPy
- **WebSocket**: For real-time updates (Phase 2)

## Troubleshooting

### Common Issues

1. **Server won't start**: Check Python version and dependencies
2. **Avatars not loading**: Verify VRM files and index.json
3. **Animations not playing**: Check animation format and mixer setup
4. **Performance issues**: Monitor memory usage, reduce model complexity

### Debug Mode

Set environment variable for detailed logging:
```bash
export LOG_LEVEL=DEBUG
./start_dev.sh
```

### Browser Console

Check browser console for client-side errors. Common issues:
- WebGL not supported
- Asset loading failures
- JavaScript errors in Three.js

## Contributing

### Code Style
- Use ES6+ features
- Follow React best practices
- Document functions and classes
- Add type hints in Python

### Testing
- Test with multiple avatar formats
- Verify animation playback
- Check environment switching
- Test on different devices/browsers

### Pull Requests
- Include feature description
- Add relevant tests
- Update documentation
- Follow migration plan phases

## Resources

### Documentation
- [Three.js Documentation](https://threejs.org/docs/)
- [VRM Specification](https://vrm.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Related Projects
- Chat Interface: `/home/barberb/motion/chat/`
- RSMT System: `/home/barberb/motion/RSMT-Realtime-Stylized-Motion-Transition/`
- Migration Plan: `/home/barberb/motion/docs/migration_plan.md`

## License

Same as parent Motion workspace. See individual component licenses for details.

---

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🔄

For questions or issues, check the migration plan documentation or server logs.
