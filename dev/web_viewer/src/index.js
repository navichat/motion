/**
 * Main Source Module Index
 * Provides clean imports for the entire Avatar AI system
 */

// Core Framework
export * from './core/index.js';

// AI Models
export * from './models/index.js';

// Motion Models (explicit exports for better accessibility)
export { Audio2GestureBVHConverter, RSMTBVHConverter } from './models/motion/index.js';

// Workers (Note: Web Workers need special instantiation)
export * from './workers/index.js';

// Animation and VRM Components
export * from './components/index.js';

// Testing Framework
export * from './testing/index.js';

// Utilities
export * from './utils/index.js';
