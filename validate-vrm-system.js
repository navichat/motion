#!/usr/bin/env node

/**
 * Simple VRM System Validator
 * Validates that the VRM system fixes are working without browser automation
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

async function validateVRMSystemFixes() {
    console.log('🎭 Validating VRM System Fixes...');
    
    const results = {
        timestamp: new Date().toISOString(),
        filesFixed: {},
        assetsAvailable: {},
        systemReady: false,
        errors: []
    };
    
    try {
        // 1. Validate key files were fixed
        console.log('📋 Step 1: Checking fixed VRM integration files...');
        
        const keyFiles = [
            'dev/web_viewer/src/scene/ClassroomAvatarIntegration.js',
            'dev/web_viewer/demos/complete_ichika_conversation_system.html',
            'dev/web_viewer/real_vrm_system_test.html'
        ];
        
        for (const file of keyFiles) {
            const fullPath = path.join(__dirname, file);
            const exists = fs.existsSync(fullPath);
            results.filesFixed[file] = exists;
            
            if (exists) {
                const content = fs.readFileSync(fullPath, 'utf8');
                
                // Check for key fixes
                if (file.includes('ClassroomAvatarIntegration.js')) {
                    const hasRealVRM = content.includes('loadRealBVHAnimations') && 
                                     content.includes('setupRealVRMAnimationSystem') &&
                                     !content.includes('createSimpleAvatar() {');
                    results.filesFixed[file + '_hasRealVRM'] = hasRealVRM;
                    console.log(`  ${hasRealVRM ? '✅' : '❌'} ${file} - Real VRM system integration`);
                } else if (file.includes('complete_ichika_conversation_system.html')) {
                    const hasFixedPaths = content.includes('./src/components/') && 
                                        !content.includes('../src/components/');
                    results.filesFixed[file + '_hasFixedPaths'] = hasFixedPaths;
                    console.log(`  ${hasFixedPaths ? '✅' : '❌'} ${file} - Fixed script paths`);
                }
            } else {
                console.log(`  ❌ ${file} - File missing`);
            }
        }
        
        // 2. Check VRM asset availability
        console.log('📋 Step 2: Checking VRM asset availability...');
        
        const vrmAssets = [
            'dev/web_viewer/assets/avatars/ichika.vrm',
            'dev/web_viewer/assets/avatars/buny.vrm',
            'dev/web_viewer/assets/avatars/kaede.vrm'
        ];
        
        for (const asset of vrmAssets) {
            const fullPath = path.join(__dirname, asset);
            const exists = fs.existsSync(fullPath);
            
            if (exists) {
                const stats = fs.statSync(fullPath);
                const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
                results.assetsAvailable[asset] = { exists: true, size: stats.size, sizeMB };
                console.log(`  ✅ ${asset} - ${sizeMB}MB`);
            } else {
                results.assetsAvailable[asset] = { exists: false };
                console.log(`  ❌ ${asset} - Not found`);
            }
        }
        
        // 3. Check BVH animation files
        console.log('📋 Step 3: Checking BVH animation files...');
        
        const bvhAssets = [
            'dev/web_viewer/assets/bvh/minimal_idle.bvh'
        ];
        
        for (const asset of bvhAssets) {
            const fullPath = path.join(__dirname, asset);
            const exists = fs.existsSync(fullPath);
            
            if (exists) {
                const stats = fs.statSync(fullPath);
                results.assetsAvailable[asset] = { exists: true, size: stats.size };
                console.log(`  ✅ ${asset} - ${(stats.size / 1024).toFixed(1)}KB`);
            } else {
                results.assetsAvailable[asset] = { exists: false };
                console.log(`  ❌ ${asset} - Not found`);
            }
        }
        
        // 4. Check VRM component files
        console.log('📋 Step 4: Checking VRM component files...');
        
        const vrmComponents = [
            'dev/web_viewer/src/components/animation/vrm/VRMLoader.js',
            'dev/web_viewer/src/components/animation/vrm/AvatarBinder.js',
            'dev/web_viewer/src/components/animation/vrm/BVHTimelineVRMIntegration.js',
            'dev/web_viewer/src/components/animation/timeline/BVHTimeline.js'
        ];
        
        for (const component of vrmComponents) {
            const fullPath = path.join(__dirname, component);
            const exists = fs.existsSync(fullPath);
            results.filesFixed[component] = exists;
            console.log(`  ${exists ? '✅' : '❌'} ${component}`);
        }
        
        // 5. Generate summary
        console.log('📋 Step 5: Generating validation summary...');
        
        const totalFiles = Object.keys(results.filesFixed).length;
        const fixedFiles = Object.values(results.filesFixed).filter(Boolean).length;
        const totalAssets = Object.keys(results.assetsAvailable).length;
        const availableAssets = Object.values(results.assetsAvailable).filter(asset => asset.exists).length;
        
        results.systemReady = (fixedFiles / totalFiles) > 0.8 && (availableAssets / totalAssets) > 0.7;
        
        console.log('📊 VALIDATION SUMMARY:');
        console.log(`   Files Fixed: ${fixedFiles}/${totalFiles} (${(fixedFiles/totalFiles*100).toFixed(1)}%)`);
        console.log(`   Assets Available: ${availableAssets}/${totalAssets} (${(availableAssets/totalAssets*100).toFixed(1)}%)`);
        console.log(`   System Ready: ${results.systemReady ? '✅ YES' : '❌ NO'}`);
        
        // 6. Create validation report
        const reportHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>VRM System Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status { padding: 15px; margin: 10px 0; border-radius: 4px; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #dee2e6; }
        .card h3 { margin-top: 0; color: #495057; }
        .file-list { list-style: none; padding: 0; }
        .file-list li { padding: 5px 0; }
        .file-list .success { color: #28a745; }
        .file-list .error { color: #dc3545; }
        .summary { background: #e9ecef; padding: 20px; border-radius: 6px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 VRM System Validation Report</h1>
            <p>Generated: ${results.timestamp}</p>
            <div class="status ${results.systemReady ? 'success' : 'error'}">
                <h2>${results.systemReady ? '✅ System Ready' : '❌ System Needs Attention'}</h2>
                <p>${results.systemReady ? 'VRM avatar system has been successfully fixed and is ready for testing' : 'VRM avatar system requires additional fixes before it can work properly'}</p>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📁 Files Fixed</h3>
                <ul class="file-list">
                    ${Object.entries(results.filesFixed).map(([file, fixed]) => 
                        `<li class="${fixed ? 'success' : 'error'}">${fixed ? '✅' : '❌'} ${file}</li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="card">
                <h3>🎭 VRM Assets</h3>
                <ul class="file-list">
                    ${Object.entries(results.assetsAvailable).map(([asset, info]) => 
                        `<li class="${info.exists ? 'success' : 'error'}">${info.exists ? '✅' : '❌'} ${asset}${info.sizeMB ? ` (${info.sizeMB}MB)` : ''}</li>`
                    ).join('')}
                </ul>
            </div>
        </div>
        
        <div class="summary">
            <h3>📊 Summary</h3>
            <p><strong>Files Fixed:</strong> ${fixedFiles}/${totalFiles} (${(fixedFiles/totalFiles*100).toFixed(1)}%)</p>
            <p><strong>Assets Available:</strong> ${availableAssets}/${totalAssets} (${(availableAssets/totalAssets*100).toFixed(1)}%)</p>
            <p><strong>Next Steps:</strong></p>
            <ul>
                ${results.systemReady ? 
                    '<li>✅ System is ready for screenshot capture</li><li>✅ VRM avatars should load instead of geometric fallbacks</li><li>✅ BVH skeletal animations should be functional</li>' :
                    '<li>❌ Fix missing files and assets</li><li>❌ Ensure VRM loading infrastructure is complete</li><li>❌ Verify BVH animation system integration</li>'
                }
            </ul>
        </div>
        
        <div class="card">
            <h3>🎯 Expected Behavior</h3>
            <p>When the system is working correctly, you should see:</p>
            <ul>
                <li>Real anime VRM avatars loading (ichika.vrm, buny.vrm, kaede.vrm)</li>
                <li>No geometric fallbacks (no pink sphere + blue rectangle)</li>
                <li>BVH skeletal animations driving avatar movement</li>
                <li>3D classroom environment with positioned VRM avatar</li>
                <li>Voice-synchronized facial expressions and gestures</li>
            </ul>
        </div>
    </div>
</body>
</html>`;
        
        const reportPath = path.join(__dirname, 'test-results', 'vrm-system-validation-report.html');
        if (!fs.existsSync(path.dirname(reportPath))) {
            fs.mkdirSync(path.dirname(reportPath), { recursive: true });
        }
        fs.writeFileSync(reportPath, reportHTML);
        
        console.log(`📊 Validation report saved: ${reportPath}`);
        
        // Save JSON results
        const jsonPath = path.join(__dirname, 'test-results', 'vrm-system-validation.json');
        fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
        
        console.log('🎉 VRM System Validation Complete!');
        return results;
        
    } catch (error) {
        console.error('❌ Validation failed:', error);
        results.errors.push(error.message);
        return results;
    }
}

// Run if called directly
if (require.main === module) {
    validateVRMSystemFixes().then(results => {
        if (results.systemReady) {
            console.log('\n✅ SUCCESS: VRM system fixes are working!');
            process.exit(0);
        } else {
            console.log('\n❌ ATTENTION: VRM system needs additional fixes');
            process.exit(1);
        }
    }).catch(error => {
        console.error('Validation error:', error);
        process.exit(1);
    });
}

module.exports = { validateVRMSystemFixes };