/**
 * Validation Utility Functions
 * 
 * Helper functions for validating data structures and configurations
 */

/**
 * Validate avatar configuration
 */
export function validateAvatarConfig(config) {
  if (!config || typeof config !== 'object') {
    return false;
  }
  
  // Required fields
  if (!config.name || typeof config.name !== 'string') {
    return false;
  }
  
  if (!config.url || typeof config.url !== 'string') {
    return false;
  }
  
  // Optional fields with type checking
  if (config.scale !== undefined && typeof config.scale !== 'number') {
    return false;
  }
  
  if (config.position !== undefined) {
    if (!Array.isArray(config.position) || config.position.length !== 3) {
      return false;
    }
    if (!config.position.every(v => typeof v === 'number')) {
      return false;
    }
  }
  
  return true;
}

/**
 * Validate environment configuration
 */
export function validateEnvironmentConfig(config) {
  if (!config || typeof config !== 'object') {
    return false;
  }
  
  // Valid environment types
  const validTypes = ['classroom', 'stage', 'studio', 'outdoor'];
  if (!config.type || !validTypes.includes(config.type)) {
    return false;
  }
  
  // Validate lighting if present
  if (config.lighting) {
    if (config.lighting.ambient && !validateColor(config.lighting.ambient)) {
      return false;
    }
    
    if (config.lighting.directional) {
      const dir = config.lighting.directional;
      if (dir.color && !validateColor(dir.color)) {
        return false;
      }
      if (dir.position && !validatePosition(dir.position)) {
        return false;
      }
    }
  }
  
  return true;
}

/**
 * Validate color array [r, g, b] with values 0-1
 */
export function validateColor(color) {
  if (!Array.isArray(color) || color.length !== 3) {
    return false;
  }
  
  return color.every(component => 
    typeof component === 'number' && component >= 0 && component <= 1
  );
}

/**
 * Validate 3D position array [x, y, z]
 */
export function validatePosition(position) {
  if (!Array.isArray(position) || position.length !== 3) {
    return false;
  }
  
  return position.every(component => typeof component === 'number');
}

/**
 * Validate animation configuration
 */
export function validateAnimationConfig(config) {
  if (!config || typeof config !== 'object') {
    return false;
  }
  
  if (!config.name || typeof config.name !== 'string') {
    return false;
  }
  
  if (!config.format || typeof config.format !== 'string') {
    return false;
  }
  
  const validFormats = ['json', 'bvh', 'fbx'];
  if (!validFormats.includes(config.format)) {
    return false;
  }
  
  return true;
}

/**
 * Validate URL format
 */
export function validateURL(url) {
  if (!url || typeof url !== 'string') {
    return false;
  }
  
  try {
    new URL(url);
    return true;
  } catch {
    // Check if it's a relative path
    return url.startsWith('/') || url.startsWith('./') || url.startsWith('../');
  }
}
