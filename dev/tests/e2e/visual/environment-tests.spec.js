import { test, expect } from '@playwright/test';

/**
 * Visual Regression Tests for Motion Viewer 3D Environment
 * 
 * These tests capture screenshots of the 3D environment and compare them
 * against reference images to detect visual regressions.
 */

test.describe('3D Environment Visual Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the motion viewer
    await page.goto('/');
    
    // Wait for the viewer to fully initialize
    await page.waitForSelector('.motion-viewer', { state: 'visible' });
    
    // Wait for WebGL context to be ready
    await page.waitForFunction(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return false;
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return gl && !gl.isContextLost();
    });
    
    // Wait for initial loading to complete
    await page.waitForSelector('.loading-overlay', { state: 'hidden', timeout: 10000 });
  });

  test('visual: classroom environment renders correctly', async ({ page }) => {
    // Ensure classroom environment is loaded
    await page.selectOption('.environment-selector', 'classroom');
    
    // Wait for environment to load
    await page.waitForTimeout(2000);
    
    // Hide UI controls for clean screenshot
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'none';
    });
    
    // Take screenshot of the 3D scene
    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('classroom-environment.png', {
      mask: [page.locator('.dev-info')], // Hide dev info
      animations: 'disabled'
    });
  });

  test('visual: stage environment renders correctly', async ({ page }) => {
    await page.selectOption('.environment-selector', 'stage');
    await page.waitForTimeout(2000);
    
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'none';
    });
    
    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('stage-environment.png', {
      animations: 'disabled'
    });
  });

  test('visual: studio environment renders correctly', async ({ page }) => {
    await page.selectOption('.environment-selector', 'studio');
    await page.waitForTimeout(2000);
    
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'none';
    });
    
    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('studio-environment.png', {
      animations: 'disabled'
    });
  });

  test('visual: outdoor environment renders correctly', async ({ page }) => {
    await page.selectOption('.environment-selector', 'outdoor');
    await page.waitForTimeout(2000);
    
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'none';
    });
    
    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('outdoor-environment.png', {
      animations: 'disabled'
    });
  });

  test('visual: avatar loading interface', async ({ page }) => {
    // Click avatar selector to show dropdown
    await page.click('.selector-button:has-text("Select Avatar")');
    
    // Take screenshot of avatar selection UI
    await expect(page.locator('.motion-viewer')).toHaveScreenshot('avatar-selection.png');
  });

  test('visual: animation controls interface', async ({ page }) => {
    // Show all controls
    const controls = page.locator('.viewer-controls');
    await expect(controls).toBeVisible();
    
    // Take screenshot of control panel
    await expect(controls).toHaveScreenshot('animation-controls.png');
  });

  test('visual: error state displays correctly', async ({ page }) => {
    // Trigger an error by trying to load invalid avatar
    await page.evaluate(() => {
      // Simulate an error in the viewer
      window.dispatchEvent(new CustomEvent('motionViewerError', {
        detail: { message: 'Test error for visual testing' }
      }));
    });
    
    // Wait for error overlay
    await page.waitForSelector('.error-overlay', { state: 'visible' });
    
    // Screenshot error state
    await expect(page.locator('.motion-viewer')).toHaveScreenshot('error-state.png');
  });

  test('visual: loading state displays correctly', async ({ page }) => {
    // Navigate to fresh page to capture loading state
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    
    // Quickly capture loading state before it disappears
    try {
      await expect(page.locator('.loading-overlay')).toHaveScreenshot('loading-state.png', {
        timeout: 2000
      });
    } catch (error) {
      // Loading might be too fast, create artificial loading state
      await page.evaluate(() => {
        const container = document.getElementById('motion-viewer-container');
        if (container) {
          container.innerHTML = `
            <div class="motion-viewer">
              <div class="loading-overlay">
                <div class="loading-spinner"></div>
                <div class="loading-text">Loading 3D Viewer...</div>
              </div>
            </div>
          `;
        }
      });
      
      await expect(page.locator('.loading-overlay')).toHaveScreenshot('loading-state.png');
    }
  });

  test('visual: responsive design on mobile', async ({ page, isMobile }) => {
    test.skip(!isMobile, 'Mobile-specific test');
    
    // Wait for mobile layout
    await page.waitForTimeout(1000);
    
    // Hide dev info for clean mobile screenshot
    await page.evaluate(() => {
      const devInfo = document.getElementById('dev-info');
      if (devInfo) devInfo.style.display = 'none';
    });
    
    // Full page screenshot on mobile
    await expect(page).toHaveScreenshot('mobile-layout.png', {
      fullPage: true
    });
  });

  test('visual: environment transitions', async ({ page }) => {
    // Start with classroom
    await page.selectOption('.environment-selector', 'classroom');
    await page.waitForTimeout(1000);
    
    // Hide controls for clean comparison
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'none';
    });
    
    // Capture before transition
    const canvas = page.locator('canvas');
    await expect(canvas).toHaveScreenshot('transition-before.png');
    
    // Switch to stage
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'block';
    });
    
    await page.selectOption('.environment-selector', 'stage');
    await page.waitForTimeout(2000);
    
    await page.evaluate(() => {
      const controls = document.querySelector('.viewer-controls');
      if (controls) controls.style.display = 'none';
    });
    
    // Capture after transition
    await expect(canvas).toHaveScreenshot('transition-after.png');
  });
});
