// Auto-initialize for screenshots
setTimeout(() => {
  if (typeof initializeSystem === 'function') {
    initializeSystem();
    console.log('System initialized');
    
    setTimeout(() => {
      if (typeof loadAssets === 'function') {
        loadAssets(); 
        console.log('Assets loading started');
      }
    }, 2000);
    
    setTimeout(() => {
      if (typeof startWalkingDemo === 'function') {
        startWalkingDemo();
        console.log('Walking demo started');
      }
    }, 7000);
  }
}, 1000);
