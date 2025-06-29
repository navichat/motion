/**
 * Storage Utility Functions
 * 
 * Helper functions for local storage and caching
 */

/**
 * Save user preferences to localStorage
 */
export function savePreferences(preferences) {
  try {
    localStorage.setItem('viewer-preferences', JSON.stringify(preferences));
  } catch (error) {
    console.warn('Failed to save preferences:', error);
  }
}

/**
 * Load user preferences from localStorage
 */
export function loadPreferences() {
  try {
    const stored = localStorage.getItem('viewer-preferences');
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.warn('Failed to load preferences:', error);
    return {};
  }
}

/**
 * Cache animation data
 */
export function cacheAnimation(key, animation) {
  try {
    const cacheKey = `animation-cache-${key}`;
    localStorage.setItem(cacheKey, JSON.stringify(animation));
  } catch (error) {
    console.warn('Failed to cache animation:', error);
  }
}

/**
 * Get cached animation data
 */
export function getCachedAnimation(key) {
  try {
    const cacheKey = `animation-cache-${key}`;
    const cached = localStorage.getItem(cacheKey);
    return cached ? JSON.parse(cached) : null;
  } catch (error) {
    console.warn('Failed to get cached animation:', error);
    return null;
  }
}

/**
 * Clear animation cache
 */
export function clearAnimationCache() {
  try {
    const keys = Object.keys(localStorage);
    for (const key of keys) {
      if (key.startsWith('animation-cache-')) {
        localStorage.removeItem(key);
      }
    }
  } catch (error) {
    console.warn('Failed to clear animation cache:', error);
  }
}

/**
 * Get storage usage information
 */
export function getStorageUsage() {
  if (!localStorage) {
    return { used: 0, available: 0 };
  }
  
  let used = 0;
  for (const key in localStorage) {
    if (localStorage.hasOwnProperty(key)) {
      used += localStorage.getItem(key).length;
    }
  }
  
  // Estimate available space (most browsers allow ~5-10MB)
  const estimated = 5 * 1024 * 1024; // 5MB estimate
  
  return {
    used: used,
    available: Math.max(0, estimated - used),
    usedMB: (used / (1024 * 1024)).toFixed(2),
    availableMB: ((estimated - used) / (1024 * 1024)).toFixed(2)
  };
}
