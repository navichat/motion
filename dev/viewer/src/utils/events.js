/**
 * Event Utility Functions
 * 
 * Event emitter and event handling utilities
 */

/**
 * Simple event emitter implementation
 */
export class EventEmitter {
  constructor() {
    this.events = {};
  }
  
  /**
   * Add event listener
   */
  on(event, listener) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(listener);
  }
  
  /**
   * Add one-time event listener
   */
  once(event, listener) {
    const onceWrapper = (...args) => {
      listener(...args);
      this.off(event, onceWrapper);
    };
    
    this.on(event, onceWrapper);
  }
  
  /**
   * Remove event listener
   */
  off(event, listener) {
    if (!this.events[event]) {
      return;
    }
    
    const index = this.events[event].indexOf(listener);
    if (index > -1) {
      this.events[event].splice(index, 1);
    }
  }
  
  /**
   * Emit event to all listeners
   */
  emit(event, ...args) {
    if (!this.events[event]) {
      return;
    }
    
    this.events[event].forEach(listener => {
      try {
        listener(...args);
      } catch (error) {
        console.error('Error in event listener:', error);
      }
    });
  }
  
  /**
   * Remove all listeners for an event
   */
  removeAllListeners(event) {
    if (event) {
      delete this.events[event];
    } else {
      this.events = {};
    }
  }
  
  /**
   * Get listener count for an event
   */
  listenerCount(event) {
    return this.events[event] ? this.events[event].length : 0;
  }
}

/**
 * Debounce function calls
 */
export function debounce(func, wait, immediate = false) {
  let timeout;
  
  return function executedFunction(...args) {
    const later = () => {
      timeout = null;
      if (!immediate) func(...args);
    };
    
    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    
    if (callNow) func(...args);
  };
}

/**
 * Throttle function calls
 */
export function throttle(func, limit) {
  let inThrottle;
  
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}
