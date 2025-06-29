import { test, expect } from '@playwright/test';

/**
 * Integration Tests for Motion Viewer API
 * 
 * These tests verify that the backend API endpoints work correctly
 * and integrate properly with the frontend 3D environment.
 */

test.describe('API Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('api: health check endpoint responds correctly', async ({ request }) => {
    const response = await request.get('/health');
    expect(response.ok()).toBeTruthy();
    
    const data = await response.json();
    expect(data.status).toBe('healthy');
    expect(data.phase).toBe('1');
    expect(data.features).toContain('3D avatar loading');
  });

  test('api: avatars endpoint returns valid data', async ({ request }) => {
    const response = await request.get('/api/avatars');
    expect(response.ok()).toBeTruthy();
    
    const avatars = await response.json();
    expect(Array.isArray(avatars)).toBeTruthy();
    expect(avatars.length).toBeGreaterThan(0);
    
    // Validate avatar structure
    const firstAvatar = avatars[0];
    expect(firstAvatar).toHaveProperty('id');
    expect(firstAvatar).toHaveProperty('name');
    expect(firstAvatar).toHaveProperty('file');
    expect(firstAvatar).toHaveProperty('format');
    expect(firstAvatar.format).toBe('vrm');
  });

  test('api: animations endpoint returns valid data', async ({ request }) => {
    const response = await request.get('/api/animations');
    expect(response.ok()).toBeTruthy();
    
    const animations = await response.json();
    expect(Array.isArray(animations)).toBeTruthy();
    expect(animations.length).toBeGreaterThan(0);
    
    // Validate animation structure
    const firstAnimation = animations[0];
    expect(firstAnimation).toHaveProperty('id');
    expect(firstAnimation).toHaveProperty('name');
    expect(firstAnimation).toHaveProperty('file');
    expect(firstAnimation).toHaveProperty('format');
    expect(firstAnimation.format).toBe('json');
    expect(firstAnimation.source).toBe('chat');
  });

  test('api: environments endpoint returns valid data', async ({ request }) => {
    const response = await request.get('/api/environments');
    expect(response.ok()).toBeTruthy();
    
    const environments = await response.json();
    expect(Array.isArray(environments)).toBeTruthy();
    expect(environments.length).toBe(4); // classroom, stage, studio, outdoor
    
    // Check for required environments
    const envIds = environments.map(env => env.id);
    expect(envIds).toContain('classroom');
    expect(envIds).toContain('stage');
    expect(envIds).toContain('studio');
    expect(envIds).toContain('outdoor');
  });

  test('api: stats endpoint provides system information', async ({ request }) => {
    const response = await request.get('/api/stats');
    expect(response.ok()).toBeTruthy();
    
    const stats = await response.json();
    expect(stats.phase).toBe('1');
    expect(stats.counts).toHaveProperty('avatars');
    expect(stats.counts).toHaveProperty('animations');
    expect(stats.counts).toHaveProperty('environments');
    expect(stats.features['3d_rendering']).toBe(true);
    expect(stats.features['neural_networks']).toBe(false); // Phase 2
  });

  test('api: individual avatar endpoint works', async ({ request }) => {
    // First get list of avatars
    const avatarsResponse = await request.get('/api/avatars');
    const avatars = await avatarsResponse.json();
    
    if (avatars.length > 0) {
      const avatarId = avatars[0].id;
      
      // Get specific avatar
      const response = await request.get(`/api/avatar/${avatarId}`);
      expect(response.ok()).toBeTruthy();
      
      const avatar = await response.json();
      expect(avatar.id).toBe(avatarId);
      expect(avatar).toHaveProperty('name');
      expect(avatar).toHaveProperty('file');
    }
  });

  test('api: individual animation endpoint works', async ({ request }) => {
    // First get list of animations
    const animationsResponse = await request.get('/api/animations');
    const animations = await animationsResponse.json();
    
    if (animations.length > 0) {
      const animationId = animations[0].id;
      
      // Get specific animation
      const response = await request.get(`/api/animation/${animationId}`);
      expect(response.ok()).toBeTruthy();
      
      const animation = await response.json();
      expect(animation.id).toBe(animationId);
      expect(animation).toHaveProperty('name');
      expect(animation).toHaveProperty('file');
    }
  });

  test('api: 404 handling for non-existent resources', async ({ request }) => {
    const response = await request.get('/api/avatar/non-existent');
    expect(response.status()).toBe(404);
    
    const error = await response.json();
    expect(error.detail).toBe('Avatar not found');
  });

  test('api: Phase 2 endpoints return 501 (not implemented)', async ({ request }) => {
    const styleResponse = await request.post('/api/style-transfer');
    expect(styleResponse.status()).toBe(501);
    
    const transitionResponse = await request.post('/api/generate-transition');
    expect(transitionResponse.status()).toBe(501);
    
    const stylesResponse = await request.get('/api/styles');
    expect(stylesResponse.ok()).toBeTruthy();
    
    const styles = await stylesResponse.json();
    expect(styles.message).toContain('Phase 2');
  });

  test('integration: frontend loads avatar list from API', async ({ page }) => {
    // Wait for the viewer to load
    await page.waitForSelector('.motion-viewer', { state: 'visible' });
    await page.waitForSelector('.loading-overlay', { state: 'hidden' });
    
    // Click avatar selector
    await page.click('.selector-button:has-text("Select Avatar")');
    
    // Check that dropdown is populated
    const avatarItems = page.locator('.dropdown-item');
    const count = await avatarItems.count();
    expect(count).toBeGreaterThan(0);
    
    // Verify that avatar names match API response
    const apiResponse = await page.request.get('/api/avatars');
    const apiAvatars = await apiResponse.json();
    
    const uiAvatarTexts = await avatarItems.allTextContents();
    
    // Each API avatar should appear in the UI
    for (const avatar of apiAvatars) {
      const found = uiAvatarTexts.some(text => text.includes(avatar.name));
      expect(found).toBeTruthy();
    }
  });

  test('integration: frontend loads animation list from API', async ({ page }) => {
    await page.waitForSelector('.motion-viewer', { state: 'visible' });
    await page.waitForSelector('.loading-overlay', { state: 'hidden' });
    
    // Click animation selector
    await page.click('.selector-button:has-text("Select Animation")');
    
    // Check that dropdown is populated
    const animationItems = page.locator('.dropdown-item');
    const count = await animationItems.count();
    expect(count).toBeGreaterThan(0);
    
    // Verify that animation names match API response
    const apiResponse = await page.request.get('/api/animations');
    const apiAnimations = await apiResponse.json();
    
    const uiAnimationTexts = await animationItems.allTextContents();
    
    // Each API animation should appear in the UI (at least some of them)
    const matches = apiAnimations.filter(animation => 
      uiAnimationTexts.some(text => text.includes(animation.name))
    );
    expect(matches.length).toBeGreaterThan(0);
  });

  test('integration: environment switching calls API correctly', async ({ page }) => {
    await page.waitForSelector('.motion-viewer', { state: 'visible' });
    await page.waitForSelector('.loading-overlay', { state: 'hidden' });
    
    // Monitor network requests
    const environmentRequests = [];
    page.on('request', request => {
      if (request.url().includes('/api/environments')) {
        environmentRequests.push(request.method());
      }
    });
    
    // Switch environments
    await page.selectOption('.environment-selector', 'stage');
    await page.waitForTimeout(1000);
    
    await page.selectOption('.environment-selector', 'studio');
    await page.waitForTimeout(1000);
    
    // Should have made API calls (or at least initial one)
    expect(environmentRequests.length).toBeGreaterThan(0);
    expect(environmentRequests[0]).toBe('GET');
  });

  test('integration: error handling for API failures', async ({ page, request }) => {
    // Temporarily break the API by stopping the server or mocking failures
    // For this test, we'll simulate by intercepting requests
    
    await page.route('/api/avatars', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal server error' })
      });
    });
    
    await page.goto('/');
    await page.waitForSelector('.motion-viewer', { state: 'visible' });
    
    // Try to open avatar selector - should handle error gracefully
    await page.click('.selector-button:has-text("Select Avatar")');
    
    // Should either show error message or fallback content
    const hasError = await page.locator('.error-message, .dropdown-item:has-text("Default")').count() > 0;
    expect(hasError).toBeTruthy();
  });

  test('integration: CORS headers are present', async ({ request }) => {
    const response = await request.get('/api/avatars');
    const headers = response.headers();
    
    expect(headers['access-control-allow-origin']).toBeDefined();
    expect(headers['access-control-allow-methods']).toBeDefined();
  });

  test('integration: API documentation is accessible', async ({ request }) => {
    const response = await request.get('/docs');
    expect(response.ok()).toBeTruthy();
    
    const contentType = response.headers()['content-type'];
    expect(contentType).toContain('text/html');
  });
});
