# Motion Platform - Integrated Development Environment

This unified development environment integrates chat functionality, 3D avatar viewing, and RSMT (Realtime Stylized Motion Transition) capabilities into a single cohesive platform.

## Features Integrated

### ✅ From Chat Folder
- **Complete Asset Library**: All animations, avatars, and scenes
- **Chat Interface**: Full real-time chat with avatar interaction
- **WebSocket Server**: Real-time communication infrastructure  
- **Psyche Module**: AI conversation and emotion handling
- **Avatar Management**: Dynamic avatar loading and animation

### ✅ From Dev Folder
- **3D Viewer**: Modern Three.js-based motion viewer
- **RSMT Integration**: Python-based neural motion processing
- **React Components**: Modern UI components for motion visualization
- **Development Tools**: Testing, linting, and build pipeline

## Architecture

```
dev/
├── assets/                 # Shared assets (copied from chat)
│   ├── animations/         # Motion animation files
│   ├── avatars/           # Avatar configurations  
│   └── scenes/            # 3D scene definitions
├── psyche/                # AI conversation engine
├── server/                # Dual server setup
│   ├── *.js              # Node.js chat server
│   ├── *.py              # Python RSMT server
│   └── engine-stub.js    # Local engine compatibility
├── webapp/               # Chat interface webapp
├── viewer/               # 3D motion viewer
└── package.json          # Unified dependencies
```

## Quick Start

### Prerequisites
- Node.js 16+
- Python 3.8+
- Git

### Setup and Run

1. **Setup Assets and Dependencies**:
   ```bash
   cd dev
   node setup-assets.js
   npm run install:all
   ```

2. **Start Development Environment**:
   ```bash
   ./start-dev-platform.sh
   ```

   This starts all services:
   - Chat Interface: http://localhost:3000
   - 3D Viewer: http://localhost:3001  
   - Node.js Server: http://localhost:8080
   - Python Server: http://localhost:8081

### Individual Service Commands

- **Development Mode**: `npm run dev` (starts all services)
- **Production Mode**: `npm run start` 
- **Build All**: `npm run build`
- **Test Suite**: `npm run test`

## Integration Points

### Asset Sharing
- All components share the same `assets/` directory
- Symlinks ensure consistent asset access
- Generated manifest provides asset discovery

### Server Communication
- Node.js server handles chat and real-time features
- Python server processes motion data and RSMT
- WebSocket bridge for real-time motion updates

### Frontend Integration  
- Chat interface can embed 3D viewer components
- Viewer can receive motion data from chat interactions
- Shared styling and UI components

## Development Workflow

### Adding New Features

1. **New Chat Features**: Add to `webapp/` and `psyche/`
2. **New Motion Features**: Add to `viewer/` and `server/*.py`
3. **New Assets**: Add to `assets/` and run `node setup-assets.js`

### Testing

- **Unit Tests**: `npm run test`
- **Integration Tests**: `./start-dev-platform.sh` then manual testing
- **Asset Verification**: `node setup-assets.js`

### Configuration

Edit `config.json` to modify:
- Service ports
- Asset paths  
- Feature toggles
- Integration settings

## Troubleshooting

### Missing Dependencies
```bash
npm run install:all
```

### Asset Issues
```bash
node setup-assets.js
```

### Server Conflicts
```bash
# Kill any existing processes
pkill -f "node.*server"
pkill -f "python.*server"
./start-dev-platform.sh
```

### Log Files
Check `logs/` directory for service-specific logs:
- `logs/node-server.log`
- `logs/python-server.log`
- `logs/webapp.log`
- `logs/viewer.log`

## Next Steps

### Phase 2 Integration
- Connect RSMT neural networks to chat interface
- Real-time motion generation from conversation
- Advanced avatar emotion and gesture mapping

### Performance Optimization
- Asset caching and lazy loading
- WebSocket connection pooling
- Python/Node.js process optimization

---

**Note**: This integration maintains all original functionality from both the `chat` and `dev` folders while providing a unified development experience.
