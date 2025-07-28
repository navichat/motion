const { test, expect } = require('@playwright/test');

test.describe('Apple-to-Apple KNN Benchmark Tests', () => {
  test('should run comprehensive KNN benchmark with 512D vectors and 4096 dataset', async ({ page }) => {
    console.log('🏁 Starting Apple-to-Apple KNN Benchmark');
    console.log('📊 Test Specifications:');
    console.log('   - Vector Dimensions: 512');
    console.log('   - Total Vectors: 4,096');
    console.log('   - Query K: 8 (top 8 results)');
    console.log('   - Distance Metric: Cosine');
    console.log('   - Test Iterations: 5');
    console.log('   - Warmup Runs: 2');
    
    // Navigate to the benchmark page
    await page.goto('http://localhost:8000/dev/web_viewer/knn-apple-to-apple-benchmark.html');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    await expect(page.locator('h1')).toContainText('Apple-to-Apple KNN Benchmark');
    
    console.log('✅ Benchmark page loaded successfully');
    
    // Capture console output from the page
    const benchmarkLogs = [];
    page.on('console', msg => {
      benchmarkLogs.push(msg.text());
    });
    
    // Run the full benchmark
    console.log('🚀 Starting full benchmark execution...');
    await page.click('button:has-text("Run Full Apple-to-Apple Benchmark")');
    
    // Wait for benchmark to complete (increased timeout for large dataset)
    await expect(page.locator('.status.success')).toContainText('Benchmark completed successfully!', { 
      timeout: 120000 // 2 minutes timeout for 4096 vectors
    });
    
    console.log('✅ Benchmark completed successfully');
    
    // Wait a bit more for results to be fully processed
    await page.waitForTimeout(3000);
    
    // Extract benchmark results
    const benchmarkResults = await page.evaluate(() => {
      return window.benchmarkResults;
    });
    
    // Validate that we have results
    expect(benchmarkResults).toBeDefined();
    expect(benchmarkResults.config).toBeDefined();
    expect(benchmarkResults.searchResults).toBeDefined();
    
    console.log('\n📊 APPLE-TO-APPLE BENCHMARK RESULTS:');
    console.log('=' * 60);
    
    // Display configuration
    console.log('🔧 Configuration:');
    console.log(`   Dimensions: ${benchmarkResults.config.dimensions}`);
    console.log(`   Total Vectors: ${benchmarkResults.config.totalVectors}`);
    console.log(`   K (top results): ${benchmarkResults.config.k}`);
    console.log(`   Iterations: ${benchmarkResults.config.iterations}`);
    console.log(`   Distance Metric: ${benchmarkResults.config.distanceMetric}`);
    
    // Analyze and compare results
    const implementations = Object.keys(benchmarkResults.searchResults);
    console.log(`\n🏁 Implementations Tested: ${implementations.join(', ')}`);
    
    // Create performance comparison
    const performanceComparison = {};
    let fastestImpl = null;
    let mostAccurateImpl = null;
    let fastestTime = Infinity;
    let highestAccuracy = 0;
    
    Object.entries(benchmarkResults.searchResults).forEach(([impl, results]) => {
      performanceComparison[impl] = {
        avgTime: results.avgSearchTime,
        accuracy: results.avgAccuracy,
        stdDev: results.stdDevSearchTime,
        errors: results.errors,
        consistency: results.stdDevSearchTime / results.avgSearchTime // Lower is more consistent
      };
      
      if (results.avgSearchTime < fastestTime) {
        fastestTime = results.avgSearchTime;
        fastestImpl = impl;
      }
      
      if (results.avgAccuracy > highestAccuracy) {
        highestAccuracy = results.avgAccuracy;
        mostAccurateImpl = impl;
      }
    });
    
    console.log('\n📈 PERFORMANCE COMPARISON:');
    Object.entries(performanceComparison).forEach(([impl, metrics]) => {
      console.log(`\n🔍 ${impl.toUpperCase()}:`);
      console.log(`   Average Search Time: ${metrics.avgTime.toFixed(3)}ms`);
      console.log(`   Standard Deviation: ±${metrics.stdDev.toFixed(3)}ms`);
      console.log(`   Consistency Score: ${(metrics.consistency * 100).toFixed(1)}% (lower is better)`);
      console.log(`   Average Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
      console.log(`   Error Count: ${metrics.errors}`);
      
      // Performance indicators
      if (impl === fastestImpl) {
        console.log(`   🏆 FASTEST IMPLEMENTATION`);
      }
      if (impl === mostAccurateImpl) {
        console.log(`   🎯 MOST ACCURATE IMPLEMENTATION`);
      }
    });
    
    console.log('\n⚖️ APPLE-TO-APPLE COMPARISON ANALYSIS:');
    
    // Speed comparison
    const speedRatios = {};
    Object.entries(performanceComparison).forEach(([impl, metrics]) => {
      speedRatios[impl] = fastestTime / metrics.avgTime;
    });
    
    console.log('🚀 Speed Performance (relative to fastest):');
    Object.entries(speedRatios).forEach(([impl, ratio]) => {
      const percentage = (ratio * 100).toFixed(1);
      if (ratio === 1.0) {
        console.log(`   ${impl}: 100.0% (baseline - fastest)`);
      } else {
        console.log(`   ${impl}: ${percentage}% (${(1/ratio).toFixed(2)}x slower)`);
      }
    });
    
    // Accuracy comparison
    console.log('\n🎯 Accuracy Performance:');
    Object.entries(performanceComparison).forEach(([impl, metrics]) => {
      const accuracyPercent = (metrics.accuracy * 100).toFixed(2);
      console.log(`   ${impl}: ${accuracyPercent}%${impl === mostAccurateImpl ? ' (highest)' : ''}`);
    });
    
    // Consistency comparison
    console.log('\n📊 Consistency Analysis (lower standard deviation is better):');
    const consistencyRanking = Object.entries(performanceComparison)
      .sort(([,a], [,b]) => a.consistency - b.consistency);
    
    consistencyRanking.forEach(([impl, metrics], index) => {
      const rank = index + 1;
      console.log(`   #${rank} ${impl}: ${(metrics.consistency * 100).toFixed(1)}% coefficient of variation`);
    });
    
    // Overall recommendation
    console.log('\n🎖️ OVERALL ASSESSMENT:');
    
    // Calculate composite score (speed * accuracy / consistency)
    const compositeScores = {};
    Object.entries(performanceComparison).forEach(([impl, metrics]) => {
      // Normalize metrics (higher is better)
      const speedScore = fastestTime / metrics.avgTime; // Higher is better
      const accuracyScore = metrics.accuracy; // Higher is better  
      const consistencyScore = 1 / (1 + metrics.consistency); // Higher is better (less variation)
      
      compositeScores[impl] = (speedScore * 0.4 + accuracyScore * 0.4 + consistencyScore * 0.2);
    });
    
    const bestOverall = Object.entries(compositeScores)
      .sort(([,a], [,b]) => b - a)[0];
    
    console.log(`🏆 Best Overall Performance: ${bestOverall[0].toUpperCase()}`);
    console.log(`   Composite Score: ${bestOverall[1].toFixed(3)}`);
    
    // Specific use case recommendations
    console.log('\n💡 USE CASE RECOMMENDATIONS:');
    console.log(`🚀 For Speed-Critical Applications: ${fastestImpl.toUpperCase()}`);
    console.log(`🎯 For Accuracy-Critical Applications: ${mostAccurateImpl.toUpperCase()}`);
    console.log(`📊 For Consistent Performance: ${consistencyRanking[0][0].toUpperCase()}`);
    
    // Test assertions
    console.log('\n✅ VALIDATION CHECKS:');
    
    // Ensure all implementations completed without excessive errors
    Object.entries(benchmarkResults.searchResults).forEach(([impl, results]) => {
      const errorRate = results.errors / (benchmarkResults.config.iterations * 10); // 10 queries per iteration
      expect(errorRate).toBeLessThan(0.1); // Less than 10% error rate
      console.log(`   ${impl}: Error rate ${(errorRate * 100).toFixed(1)}% ✅`);
    });
    
    // Ensure reasonable performance (all should complete within reasonable time)
    Object.entries(benchmarkResults.searchResults).forEach(([impl, results]) => {
      expect(results.avgSearchTime).toBeLessThan(1000); // Less than 1 second average
      console.log(`   ${impl}: Average time ${results.avgSearchTime.toFixed(2)}ms ✅`);
    });
    
    // Ensure reasonable accuracy (should find some correct matches)
    Object.entries(benchmarkResults.searchResults).forEach(([impl, results]) => {
      expect(results.avgAccuracy).toBeGreaterThan(0.1); // At least 10% accuracy
      console.log(`   ${impl}: Accuracy ${(results.avgAccuracy * 100).toFixed(1)}% ✅`);
    });
    
    console.log('\n🎉 Apple-to-Apple KNN Benchmark completed successfully!');
    console.log(`📊 Tested ${implementations.length} implementations with ${benchmarkResults.config.totalVectors} vectors`);
    console.log(`🎯 Each implementation processed ${benchmarkResults.config.iterations * 10} queries`);
    console.log(`⚡ Performance range: ${fastestTime.toFixed(2)}ms - ${Math.max(...Object.values(performanceComparison).map(m => m.avgTime)).toFixed(2)}ms`);
    
    // Store results for potential further analysis
    await page.evaluate((results) => {
      window.finalBenchmarkResults = results;
    }, benchmarkResults);
    
    // Final validation that the benchmark ran with correct parameters
    expect(benchmarkResults.config.dimensions).toBe(512);
    expect(benchmarkResults.config.totalVectors).toBe(4096);
    expect(benchmarkResults.config.k).toBe(8);
    expect(benchmarkResults.config.distanceMetric).toBe('cosine');
    
    console.log('\n✅ All validation checks passed - Apple-to-Apple benchmark successful!');
  });
});
