/**
 * Screenshot Analysis and Comparison Tools
 * 
 * Automated visual regression detection and analysis for the Motion Viewer
 */

import { promises as fs } from 'fs';
import path from 'path';
import sharp from 'sharp';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';

export class VisualAnalyzer {
  constructor(options = {}) {
    this.threshold = options.threshold || 0.1;
    this.includeAA = options.includeAA || false;
    this.outputDir = options.outputDir || './reports/visual';
    this.screenshotDir = options.screenshotDir || './screenshots';
    this.baselineDir = path.join(this.screenshotDir, 'baseline');
    this.currentDir = path.join(this.screenshotDir, 'current');
    this.diffDir = path.join(this.screenshotDir, 'diff');
  }

  /**
   * Initialize analysis directories
   */
  async initialize() {
    const dirs = [this.outputDir, this.baselineDir, this.currentDir, this.diffDir];
    
    for (const dir of dirs) {
      await fs.mkdir(dir, { recursive: true });
    }
  }

  /**
   * Compare two screenshots and generate diff
   */
  async compareScreenshots(baselinePath, currentPath, diffPath) {
    try {
      // Read images
      const baseline = PNG.sync.read(await fs.readFile(baselinePath));
      const current = PNG.sync.read(await fs.readFile(currentPath));

      // Ensure images have same dimensions
      if (baseline.width !== current.width || baseline.height !== current.height) {
        throw new Error(`Image dimensions don't match: ${baseline.width}x${baseline.height} vs ${current.width}x${current.height}`);
      }

      // Create diff image
      const diff = new PNG({ width: baseline.width, height: baseline.height });

      // Compare pixels
      const pixelDiff = pixelmatch(
        baseline.data,
        current.data,
        diff.data,
        baseline.width,
        baseline.height,
        {
          threshold: this.threshold,
          includeAA: this.includeAA
        }
      );

      // Save diff image
      await fs.writeFile(diffPath, PNG.sync.write(diff));

      // Calculate difference percentage
      const totalPixels = baseline.width * baseline.height;
      const diffPercentage = (pixelDiff / totalPixels) * 100;

      return {
        pixelDiff,
        totalPixels,
        diffPercentage,
        hasDifferences: pixelDiff > 0,
        baselineSize: { width: baseline.width, height: baseline.height },
        currentSize: { width: current.width, height: current.height }
      };
    } catch (error) {
      console.error(`Error comparing screenshots: ${error.message}`);
      return {
        error: error.message,
        hasDifferences: true,
        diffPercentage: 100
      };
    }
  }

  /**
   * Analyze all screenshots in the current directory
   */
  async analyzeAll() {
    const results = {
      timestamp: new Date().toISOString(),
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        new: 0,
        removed: 0
      },
      tests: []
    };

    try {
      // Get all current screenshots
      const currentFiles = await fs.readdir(this.currentDir);
      const baselineFiles = await fs.readdir(this.baselineDir).catch(() => []);

      results.summary.total = currentFiles.length;

      // Compare existing screenshots
      for (const filename of currentFiles) {
        if (!filename.endsWith('.png')) continue;

        const baselinePath = path.join(this.baselineDir, filename);
        const currentPath = path.join(this.currentDir, filename);
        const diffPath = path.join(this.diffDir, filename);

        let comparison;
        let status;

        if (baselineFiles.includes(filename)) {
          // Compare with baseline
          comparison = await this.compareScreenshots(baselinePath, currentPath, diffPath);
          
          if (comparison.error) {
            status = 'error';
            results.summary.failed++;
          } else if (comparison.hasDifferences && comparison.diffPercentage > 1) {
            status = 'failed';
            results.summary.failed++;
          } else {
            status = 'passed';
            results.summary.passed++;
          }
        } else {
          // New screenshot
          status = 'new';
          results.summary.new++;
          comparison = { diffPercentage: 0, hasDifferences: false };
        }

        results.tests.push({
          filename,
          status,
          comparison,
          baselineExists: baselineFiles.includes(filename)
        });
      }

      // Check for removed screenshots
      for (const filename of baselineFiles) {
        if (!currentFiles.includes(filename)) {
          results.summary.removed++;
          results.tests.push({
            filename,
            status: 'removed',
            comparison: null,
            baselineExists: true
          });
        }
      }

      // Generate detailed report
      await this.generateReport(results);

      return results;
    } catch (error) {
      console.error(`Error during analysis: ${error.message}`);
      throw error;
    }
  }

  /**
   * Generate HTML report
   */
  async generateReport(results) {
    const reportPath = path.join(this.outputDir, 'visual-report.html');
    
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Regression Report - Motion Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .summary-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; }
        .summary-card h3 { margin: 0 0 10px 0; font-size: 24px; }
        .summary-card p { margin: 0; color: #666; }
        .passed { border-left: 4px solid #4CAF50; }
        .failed { border-left: 4px solid #f44336; }
        .new { border-left: 4px solid #2196F3; }
        .removed { border-left: 4px solid #FF9800; }
        .error { border-left: 4px solid #9C27B0; }
        .test-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .test-card { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
        .test-card img { width: 100%; height: 200px; object-fit: cover; }
        .test-info { padding: 15px; }
        .test-info h4 { margin: 0 0 10px 0; }
        .diff-percentage { font-weight: bold; }
        .diff-percentage.high { color: #f44336; }
        .diff-percentage.medium { color: #FF9800; }
        .diff-percentage.low { color: #4CAF50; }
        .image-tabs { margin-top: 10px; }
        .image-tabs button { padding: 5px 10px; margin-right: 5px; border: 1px solid #ddd; background: white; cursor: pointer; }
        .image-tabs button.active { background: #2196F3; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Visual Regression Report</h1>
        <p>Generated: ${results.timestamp}</p>
        <p>Motion Viewer 3D Environment Testing</p>
    </div>

    <div class="summary">
        <div class="summary-card passed">
            <h3>${results.summary.passed}</h3>
            <p>Passed</p>
        </div>
        <div class="summary-card failed">
            <h3>${results.summary.failed}</h3>
            <p>Failed</p>
        </div>
        <div class="summary-card new">
            <h3>${results.summary.new}</h3>
            <p>New</p>
        </div>
        <div class="summary-card removed">
            <h3>${results.summary.removed}</h3>
            <p>Removed</p>
        </div>
    </div>

    <div class="test-grid">
        ${results.tests.map(test => this.generateTestCard(test)).join('')}
    </div>

    <script>
        function showImage(testId, type) {
            const img = document.getElementById('img-' + testId);
            const buttons = document.querySelectorAll('[data-test="' + testId + '"]');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            document.querySelector('[data-test="' + testId + '"][data-type="' + type + '"]').classList.add('active');
            
            img.src = '../screenshots/' + type + '/' + testId + '.png';
        }
    </script>
</body>
</html>`;

    await fs.writeFile(reportPath, html);
    
    // Also save JSON summary
    await fs.writeFile(
      path.join(this.outputDir, 'summary.json'),
      JSON.stringify(results, null, 2)
    );

    console.log(`📊 Visual regression report generated: ${reportPath}`);
  }

  /**
   * Generate HTML for test card
   */
  generateTestCard(test) {
    const statusClass = test.status;
    const diffClass = test.comparison?.diffPercentage > 5 ? 'high' : 
                     test.comparison?.diffPercentage > 1 ? 'medium' : 'low';

    return `
    <div class="test-card ${statusClass}">
        <img id="img-${test.filename}" src="../screenshots/current/${test.filename}" alt="${test.filename}">
        <div class="test-info">
            <h4>${test.filename}</h4>
            <p><strong>Status:</strong> ${test.status}</p>
            ${test.comparison ? `
                <p class="diff-percentage ${diffClass}">
                    <strong>Difference:</strong> ${test.comparison.diffPercentage?.toFixed(2) || 0}%
                </p>
            ` : ''}
            ${test.baselineExists ? `
                <div class="image-tabs">
                    <button data-test="${test.filename}" data-type="current" class="active" 
                            onclick="showImage('${test.filename}', 'current')">Current</button>
                    <button data-test="${test.filename}" data-type="baseline" 
                            onclick="showImage('${test.filename}', 'baseline')">Baseline</button>
                    ${test.status === 'failed' ? `
                        <button data-test="${test.filename}" data-type="diff" 
                                onclick="showImage('${test.filename}', 'diff')">Diff</button>
                    ` : ''}
                </div>
            ` : ''}
        </div>
    </div>`;
  }

  /**
   * Update baseline screenshots with current ones
   */
  async updateBaseline() {
    const currentFiles = await fs.readdir(this.currentDir);
    
    for (const filename of currentFiles) {
      if (!filename.endsWith('.png')) continue;

      const currentPath = path.join(this.currentDir, filename);
      const baselinePath = path.join(this.baselineDir, filename);

      await fs.copyFile(currentPath, baselinePath);
    }

    console.log(`✅ Updated ${currentFiles.length} baseline screenshots`);
  }

  /**
   * Resize image to standard dimensions
   */
  async resizeImage(inputPath, outputPath, width = 1920, height = 1080) {
    await sharp(inputPath)
      .resize(width, height, { fit: 'contain', background: '#000000' })
      .png()
      .toFile(outputPath);
  }

  /**
   * Generate image thumbnails for reports
   */
  async generateThumbnails() {
    const thumbnailDir = path.join(this.screenshotDir, 'thumbnails');
    await fs.mkdir(thumbnailDir, { recursive: true });

    const currentFiles = await fs.readdir(this.currentDir);
    
    for (const filename of currentFiles) {
      if (!filename.endsWith('.png')) continue;

      const inputPath = path.join(this.currentDir, filename);
      const outputPath = path.join(thumbnailDir, filename);

      await sharp(inputPath)
        .resize(400, 300, { fit: 'cover' })
        .png()
        .toFile(outputPath);
    }

    console.log(`📸 Generated ${currentFiles.length} thumbnails`);
  }

  /**
   * Clean up old screenshots
   */
  async cleanup(daysToKeep = 30) {
    const cutoffDate = new Date(Date.now() - (daysToKeep * 24 * 60 * 60 * 1000));
    
    const dirs = [this.currentDir, this.diffDir];
    
    for (const dir of dirs) {
      const files = await fs.readdir(dir);
      
      for (const filename of files) {
        const filePath = path.join(dir, filename);
        const stats = await fs.stat(filePath);
        
        if (stats.mtime < cutoffDate) {
          await fs.unlink(filePath);
          console.log(`🗑️ Removed old screenshot: ${filename}`);
        }
      }
    }
  }
}

export default VisualAnalyzer;
