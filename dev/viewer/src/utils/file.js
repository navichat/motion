/**
 * File Utility Functions
 * 
 * Helper functions for file type detection, validation, and processing
 */

/**
 * Get file type from file extension
 */
export function getFileType(filename) {
  if (!filename || typeof filename !== 'string') {
    return 'unknown';
  }
  
  const extension = filename.toLowerCase().split('.').pop();
  
  const typeMap = {
    'vrm': 'vrm',
    'gltf': 'gltf',
    'glb': 'gltf',
    'bvh': 'bvh',
    'json': 'json',
    'fbx': 'fbx',
    'obj': 'obj'
  };
  
  return typeMap[extension] || 'unknown';
}

/**
 * Validate file size against limit
 */
export function validateFileSize(file, maxSizeBytes) {
  if (!file || typeof file.size !== 'number') {
    return false;
  }
  
  return file.size <= maxSizeBytes;
}

/**
 * Read file as text
 */
export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = () => {
      resolve(reader.result);
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsText(file);
  });
}

/**
 * Read file as array buffer
 */
export function readFileAsArrayBuffer(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = () => {
      resolve(reader.result);
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
