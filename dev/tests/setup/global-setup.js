/**
 * Global Test Setup and Teardown
 * 
 * Configures test environment, fixtures, and cleanup for Motion Viewer testing
 */

import { execSync, spawn } from 'child_process';
import { promises as fs } from 'fs';
import fsSync from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class TestEnvironment {
  constructor() {
    this.projectRoot = path.resolve(__dirname, '../../..');
    this.serverProcess = null;
    this.serverUrl = process.env.TEST_SERVER_URL || 'http://localhost:8081';
    this.isCI = process.env.CI === 'true';
  }

  /**
   * Global setup - runs once before all tests
   */
  async globalSetup() {
    console.log('🚀 Setting up Motion Viewer test environment...');

    try {
      // Create necessary directories
      await this.createTestDirectories();

      // Generate test fixtures
      await this.generateTestFixtures();

      // Start test server if not in CI
      if (!this.isCI) {
        await this.startTestServer();
      }

      // Wait for server to be ready
      await this.waitForServer();

      // Set up test database
      await this.setupTestDatabase();

      // Generate reference screenshots if needed
      await this.setupReferenceScreenshots();

      console.log('✅ Test environment setup complete');
    } catch (error) {
      console.error('❌ Test environment setup failed:', error);
      throw error;
    }
  }

  /**
   * Global teardown - runs once after all tests
   */
  async globalTeardown() {
    console.log('🧹 Cleaning up test environment...');

    try {
      // Stop test server
      if (this.serverProcess) {
        this.serverProcess.kill();
        this.serverProcess = null;
      }

      // Clean up test database
      await this.cleanupTestDatabase();

      // Clean up temporary files
      await this.cleanupTempFiles();

      console.log('✅ Test environment cleanup complete');
    } catch (error) {
      console.error('❌ Test environment cleanup failed:', error);
    }
  }

  /**
   * Create necessary test directories
   */
  async createTestDirectories() {
    const dirs = [
      'tests/screenshots/baseline',
      'tests/screenshots/current',
      'tests/screenshots/diff',
      'tests/screenshots/thumbnails',
      'tests/reports/visual',
      'tests/reports/performance',
      'tests/reports/lighthouse',
      'tests/fixtures/animations',
      'tests/fixtures/avatars',
      'tests/fixtures/environments',
      'tests/coverage',
      'tests/temp'
    ];

    for (const dir of dirs) {
      const fullPath = path.join(this.projectRoot, 'dev', dir);
      await fs.mkdir(fullPath, { recursive: true });
    }
  }

  /**
   * Generate test fixtures and mock data
   */
  async generateTestFixtures() {
    await Promise.all([
      this.generateAnimationFixtures(),
      this.generateAvatarFixtures(),
      this.generateEnvironmentFixtures()
    ]);
  }

  /**
   * Generate animation test fixtures
   */
  async generateAnimationFixtures() {
    const animationsDir = path.join(this.projectRoot, 'dev/tests/fixtures/animations');

    // Generate simple walking animation
    const walkingAnimation = {
      format: 'json',
      name: 'test-walking',
      duration: 2.0,
      tracks: [
        {
          name: 'Hips.position',
          type: 'vector',
          times: [0, 0.5, 1.0, 1.5, 2.0],
          values: [
            0, 1, 0,      // Frame 0
            0.1, 1.1, 0.1, // Frame 1
            0, 1, 0.2,    // Frame 2
            -0.1, 1.1, 0.3, // Frame 3
            0, 1, 0.4     // Frame 4
          ]
        },
        {
          name: 'LeftLeg.rotation',
          type: 'quaternion',
          times: [0, 0.5, 1.0, 1.5, 2.0],
          values: [
            0, 0, 0, 1,      // Rest
            0.1, 0, 0, 0.995, // Forward
            0, 0, 0, 1,      // Rest
            -0.1, 0, 0, 0.995, // Back
            0, 0, 0, 1       // Rest
          ]
        }
      ]
    };

    await fs.writeFile(
      path.join(animationsDir, 'walking.json'),
      JSON.stringify(walkingAnimation, null, 2)
    );

    // Generate BVH test file
    const testBVH = `HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.00 5.36 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.00 5.36 0.00
        }
    }
}
MOTION
Frames: 10
Frame Time: 0.033333
0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
0.10 1.10 0.10 5.00 0.00 0.00 2.00 0.00 0.00
0.00 1.00 0.20 0.00 0.00 0.00 0.00 0.00 0.00
-0.10 1.10 0.30 -5.00 0.00 0.00 -2.00 0.00 0.00
0.00 1.00 0.40 0.00 0.00 0.00 0.00 0.00 0.00
0.10 1.10 0.50 5.00 0.00 0.00 2.00 0.00 0.00
0.00 1.00 0.60 0.00 0.00 0.00 0.00 0.00 0.00
-0.10 1.10 0.70 -5.00 0.00 0.00 -2.00 0.00 0.00
0.00 1.00 0.80 0.00 0.00 0.00 0.00 0.00 0.00
0.00 1.00 0.90 0.00 0.00 0.00 0.00 0.00 0.00
`;

    await fs.writeFile(path.join(animationsDir, 'test-walking.bvh'), testBVH);
  }

  /**
   * Generate avatar test fixtures
   */
  async generateAvatarFixtures() {
    const avatarsDir = path.join(this.projectRoot, 'dev/tests/fixtures/avatars');

    const avatarConfig = {
      name: 'Test Avatar',
      url: '/fixtures/avatars/test-avatar.vrm',
      scale: 1.0,
      position: [0, 0, 0],
      rotation: [0, 0, 0]
    };

    await fs.writeFile(
      path.join(avatarsDir, 'test-avatar.json'),
      JSON.stringify(avatarConfig, null, 2)
    );

    // Create avatar index
    const avatarIndex = [
      {
        id: 'test-avatar-1',
        name: 'Test Avatar 1',
        url: '/fixtures/avatars/test-avatar-1.vrm',
        thumbnail: '/fixtures/avatars/test-avatar-1-thumb.jpg',
        scale: 1.0
      },
      {
        id: 'test-avatar-2',
        name: 'Test Avatar 2',
        url: '/fixtures/avatars/test-avatar-2.vrm',
        thumbnail: '/fixtures/avatars/test-avatar-2-thumb.jpg',
        scale: 0.9
      }
    ];

    await fs.writeFile(
      path.join(avatarsDir, 'index.json'),
      JSON.stringify(avatarIndex, null, 2)
    );
  }

  /**
   * Generate environment test fixtures
   */
  async generateEnvironmentFixtures() {
    const environmentsDir = path.join(this.projectRoot, 'dev/tests/fixtures/environments');

    const environments = {
      classroom: {
        type: 'classroom',
        lighting: {
          ambient: [0.3, 0.3, 0.3],
          directional: {
            color: [1, 1, 1],
            position: [10, 10, 5],
            intensity: 1.0
          }
        },
        objects: [
          { type: 'desk', position: [0, 0, 2] },
          { type: 'chair', position: [0, 0, 0] },
          { type: 'whiteboard', position: [0, 2, 4] }
        ]
      },
      stage: {
        type: 'stage',
        lighting: {
          ambient: [0.2, 0.2, 0.2],
          directional: {
            color: [1, 0.9, 0.8],
            position: [0, 15, 0],
            intensity: 1.5
          },
          spotlights: [
            { position: [5, 8, 5], target: [0, 0, 0], intensity: 2.0 },
            { position: [-5, 8, 5], target: [0, 0, 0], intensity: 2.0 }
          ]
        },
        objects: [
          { type: 'stage-platform', position: [0, -0.5, 0] },
          { type: 'backdrop', position: [0, 3, -5] }
        ]
      }
    };

    for (const [name, config] of Object.entries(environments)) {
      await fs.writeFile(
        path.join(environmentsDir, `${name}.json`),
        JSON.stringify(config, null, 2)
      );
    }
  }

  /**
   * Start test server
   */
  async startTestServer() {
    return new Promise((resolve, reject) => {
      console.log('🔄 Starting test server...');

      try {
        const serverPath = path.join(this.projectRoot, 'dev/server');
        
        // Check if Python server exists
        const serverFile = path.join(serverPath, 'enhanced_motion_server.py');
        if (!fsSync.existsSync(serverFile)) {
          throw new Error('Python server file not found');
        }

        // Start Python server in background
        this.serverProcess = spawn('python3', ['enhanced_motion_server.py'], {
          cwd: serverPath,
          detached: true,
          stdio: ['ignore', 'pipe', 'pipe']
        });

        this.serverProcess.unref();

        // Give server time to start
        setTimeout(resolve, 5000);
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Wait for server to be ready
   */
  async waitForServer(maxAttempts = 30) {
    console.log(`⏳ Waiting for server at ${this.serverUrl}...`);

    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await fetch(`${this.serverUrl}/health`);
        if (response.ok) {
          console.log('✅ Test server is ready');
          return;
        }
      } catch {
        // Server not ready yet
      }

      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    throw new Error('Test server failed to start within timeout');
  }

  /**
   * Set up test database
   */
  async setupTestDatabase() {
    // Mock database setup for testing
    console.log('🗄️ Setting up test database...');
    
    // In a real implementation, you would:
    // - Create test database
    // - Run migrations
    // - Seed test data
  }

  /**
   * Set up reference screenshots
   */
  async setupReferenceScreenshots() {
    const baselineDir = path.join(this.projectRoot, 'dev/tests/screenshots/baseline');
    const files = await fs.readdir(baselineDir).catch(() => []);

    if (files.length === 0) {
      console.log('📸 No baseline screenshots found - will be generated on first run');
    } else {
      console.log(`📸 Found ${files.length} baseline screenshots`);
    }
  }

  /**
   * Clean up test database
   */
  async cleanupTestDatabase() {
    console.log('🗄️ Cleaning up test database...');
    // Cleanup test database
  }

  /**
   * Clean up temporary files
   */
  async cleanupTempFiles() {
    const tempDir = path.join(this.projectRoot, 'dev/tests/temp');
    try {
      const files = await fs.readdir(tempDir);
      for (const file of files) {
        await fs.unlink(path.join(tempDir, file));
      }
    } catch {
      // Directory might not exist
    }
  }
}

export default TestEnvironment;
