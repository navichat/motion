# CI/CD Testing Framework for Motion Viewer

## Overview

This framework provides automated testing for the 3D avatar viewer with visual regression testing, screenshot analysis, and continuous integration capabilities.

## Components

### 1. Visual Testing Suite
- **Playwright**: Browser automation and screenshot capture
- **Puppeteer**: Headless Chrome testing for 3D rendering
- **Canvas Testing**: WebGL rendering validation
- **Image Comparison**: Visual regression detection

### 2. 3D Environment Testing
- **Scene Rendering**: Verify all environments load correctly
- **Avatar Loading**: Test VRM file loading and display
- **Animation Playback**: Validate motion data rendering
- **Performance Monitoring**: FPS and memory usage tracking

### 3. Automated Analysis
- **Screenshot Comparison**: Detect visual regressions
- **Performance Metrics**: Rendering benchmarks
- **Error Detection**: JavaScript and WebGL errors
- **Asset Validation**: Verify all resources load

### 4. CI/CD Pipeline
- **GitHub Actions**: Automated testing on commits
- **Docker**: Consistent testing environment
- **Artifact Storage**: Screenshot and test results
- **Notification**: Slack/email alerts for failures

## Architecture

```
tests/
├── e2e/                    # End-to-end tests
│   ├── visual/            # Visual regression tests
│   ├── performance/       # Performance benchmarks
│   └── integration/       # API integration tests
├── unit/                  # Unit tests
│   ├── viewer/           # 3D viewer components
│   ├── server/           # API server tests
│   └── utils/            # Utility functions
├── fixtures/             # Test data and expectations
│   ├── screenshots/      # Reference screenshots
│   ├── avatars/         # Test avatar files
│   └── animations/      # Test animation data
├── tools/               # Testing utilities
│   ├── screenshot-compare.js
│   ├── performance-analyzer.js
│   └── report-generator.js
└── ci/                  # CI/CD configuration
    ├── docker/
    ├── github-actions/
    └── scripts/
```

## Quick Start

```bash
# Install testing dependencies
npm install

# Run all tests
npm test

# Run visual tests only
npm run test:visual

# Run performance tests
npm run test:performance

# Generate test report
npm run test:report
```
