/**
 * Setup Fixtures - Motion Viewer Testing Framework
 * Creates necessary test fixtures and baseline images
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class FixtureSetup {
    constructor() {
        this.fixturesDir = path.join(__dirname, '../fixtures');
        this.baselinesDir = path.join(__dirname, '../baselines');
        this.reportsDir = path.join(__dirname, '../reports');
    }

    async setup() {
        console.log('🔧 Setting up test fixtures...');
        
        // Create directories
        this.createDirectories();
        
        // Create sample data files
        this.createSampleAvatars();
        this.createSampleAnimations();
        this.createSampleEnvironments();
        this.createMockData();
        
        console.log('✅ Test fixtures setup complete!');
    }

    createDirectories() {
        const dirs = [
            this.fixturesDir,
            path.join(this.fixturesDir, 'avatars'),
            path.join(this.fixturesDir, 'animations'),
            path.join(this.fixturesDir, 'environments'),
            path.join(this.fixturesDir, 'mock-data'),
            this.baselinesDir,
            path.join(this.baselinesDir, 'classroom'),
            path.join(this.baselinesDir, 'stage'),
            path.join(this.baselinesDir, 'studio'),
            path.join(this.baselinesDir, 'outdoor'),
            this.reportsDir
        ];

        dirs.forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`  📁 Created: ${path.relative(process.cwd(), dir)}`);
            }
        });
    }

    createSampleAvatars() {
        const avatarData = {
            "test_avatar_1": {
                "id": "test_avatar_1",
                "name": "Test Avatar",
                "format": "vrm",
                "url": "/assets/avatars/test_avatar.vrm",
                "metadata": {
                    "version": "1.0",
                    "author": "Test Author",
                    "created": "2025-06-28"
                }
            }
        };

        const avatarFile = path.join(this.fixturesDir, 'avatars', 'test-avatars.json');
        fs.writeFileSync(avatarFile, JSON.stringify(avatarData, null, 2));
        console.log('  🎭 Created sample avatar data');
    }

    createSampleAnimations() {
        const animationData = {
            "test_animation_1": {
                "id": "test_animation_1",
                "name": "Test Walk Animation",
                "duration": 2000,
                "fps": 30,
                "keyframes": [
                    {
                        "time": 0,
                        "joints": {
                            "root": { "position": [0, 0, 0], "rotation": [0, 0, 0, 1] },
                            "spine": { "rotation": [0, 0, 0, 1] }
                        }
                    },
                    {
                        "time": 1000,
                        "joints": {
                            "root": { "position": [0, 0, 1], "rotation": [0, 0, 0, 1] },
                            "spine": { "rotation": [0, 0.1, 0, 1] }
                        }
                    }
                ]
            }
        };

        const animationFile = path.join(this.fixturesDir, 'animations', 'test-animations.json');
        fs.writeFileSync(animationFile, JSON.stringify(animationData, null, 2));
        console.log('  🚶 Created sample animation data');
    }

    createSampleEnvironments() {
        const environments = [
            { id: 'classroom', name: 'Classroom', lighting: 'bright' },
            { id: 'stage', name: 'Stage', lighting: 'dramatic' },
            { id: 'studio', name: 'Studio', lighting: 'neutral' },
            { id: 'outdoor', name: 'Outdoor', lighting: 'natural' }
        ];

        const envFile = path.join(this.fixturesDir, 'environments', 'test-environments.json');
        fs.writeFileSync(envFile, JSON.stringify(environments, null, 2));
        console.log('  🏢 Created sample environment data');
    }

    createMockData() {
        // API responses mock data
        const apiResponses = {
            "GET /api/avatars": {
                "status": 200,
                "data": [
                    { "id": "avatar1", "name": "Avatar 1", "status": "ready" },
                    { "id": "avatar2", "name": "Avatar 2", "status": "ready" }
                ]
            },
            "GET /api/animations": {
                "status": 200,
                "data": [
                    { "id": "walk", "name": "Walking", "duration": 2000 },
                    { "id": "idle", "name": "Idle", "duration": 5000 }
                ]
            },
            "POST /api/upload": {
                "status": 200,
                "data": { "id": "uploaded123", "message": "Upload successful" }
            }
        };

        const mockFile = path.join(this.fixturesDir, 'mock-data', 'api-responses.json');
        fs.writeFileSync(mockFile, JSON.stringify(apiResponses, null, 2));
        console.log('  📡 Created API mock data');

        // Create placeholder baseline images (empty files for now)
        const environments = ['classroom', 'stage', 'studio', 'outdoor'];
        environments.forEach(env => {
            const baselineFile = path.join(this.baselinesDir, env, 'baseline.png');
            if (!fs.existsSync(baselineFile)) {
                fs.writeFileSync(baselineFile, ''); // Placeholder
                console.log(`  🖼️ Created baseline placeholder: ${env}`);
            }
        });
    }
}

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    const setup = new FixtureSetup();
    setup.setup().catch(console.error);
}

export default FixtureSetup;
