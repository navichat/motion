/**
 * Animation Utility Functions
 * 
 * Helper functions for animation validation, conversion, and processing
 */

/**
 * Validate JSON animation format
 */
export function validateAnimationFormat(animation) {
  if (!animation || typeof animation !== 'object') {
    return false;
  }
  
  if (!animation.format || animation.format !== 'json') {
    return false;
  }
  
  if (!animation.name || typeof animation.name !== 'string') {
    return false;
  }
  
  if (!animation.tracks || !Array.isArray(animation.tracks)) {
    return false;
  }
  
  return true;
}

/**
 * Convert BVH data to JSON animation format
 */
export function convertBVHToJSON(bvhData) {
  const lines = bvhData.split('\n');
  const animation = {
    format: 'json',
    name: 'converted-animation',
    tracks: []
  };
  
  let inMotionSection = false;
  let frameCount = 0;
  let frameTime = 0.033333; // Default 30fps
  
  for (const line of lines) {
    const trimmed = line.trim();
    
    if (trimmed.startsWith('MOTION')) {
      inMotionSection = true;
      continue;
    }
    
    if (trimmed.startsWith('Frames:')) {
      frameCount = parseInt(trimmed.split(':')[1].trim());
      continue;
    }
    
    if (trimmed.startsWith('Frame Time:')) {
      frameTime = parseFloat(trimmed.split(':')[1].trim());
      continue;
    }
    
    if (inMotionSection && trimmed.length > 0 && !isNaN(parseFloat(trimmed.split(' ')[0]))) {
      // This is frame data
      const values = trimmed.split(/\s+/).map(v => parseFloat(v));
      
      // Create basic tracks from frame data
      if (animation.tracks.length === 0) {
        for (let i = 0; i < values.length; i++) {
          animation.tracks.push({
            name: `channel_${i}`,
            type: 'number',
            times: [],
            values: []
          });
        }
      }
      
      const currentTime = animation.tracks[0].times.length * frameTime;
      values.forEach((value, index) => {
        if (animation.tracks[index]) {
          animation.tracks[index].times.push(currentTime);
          animation.tracks[index].values.push(value);
        }
      });
    }
  }
  
  return animation;
}

/**
 * Calculate animation duration from tracks
 */
export function calculateAnimationDuration(animation) {
  if (!animation.tracks || animation.tracks.length === 0) {
    return 0;
  }
  
  let maxTime = 0;
  for (const track of animation.tracks) {
    if (track.times && track.times.length > 0) {
      const trackMax = Math.max(...track.times);
      maxTime = Math.max(maxTime, trackMax);
    }
  }
  
  return maxTime;
}

/**
 * Interpolate values between keyframes
 */
export function interpolateValues(values, times, targetTime, componentCount = 1) {
  if (!values || !times || times.length === 0) {
    return new Array(componentCount).fill(0);
  }
  
  // Find the two keyframes to interpolate between
  let beforeIndex = 0;
  let afterIndex = 0;
  
  for (let i = 0; i < times.length - 1; i++) {
    if (targetTime >= times[i] && targetTime <= times[i + 1]) {
      beforeIndex = i;
      afterIndex = i + 1;
      break;
    }
  }
  
  if (beforeIndex === afterIndex) {
    // No interpolation needed
    const startIndex = beforeIndex * componentCount;
    return values.slice(startIndex, startIndex + componentCount);
  }
  
  // Linear interpolation
  const t = (targetTime - times[beforeIndex]) / (times[afterIndex] - times[beforeIndex]);
  const result = [];
  
  for (let i = 0; i < componentCount; i++) {
    const beforeValue = values[beforeIndex * componentCount + i];
    const afterValue = values[afterIndex * componentCount + i];
    result.push(beforeValue + (afterValue - beforeValue) * t);
  }
  
  return result;
}
