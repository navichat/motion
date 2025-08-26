#!/usr/bin/env node

/**
 * Simple Browser Test for VRM System
 * Uses a simple HTTP request to test if the demo pages load properly
 */

const http = require('http');
const { URL } = require('url');

async function testVRMSystemWithBrowser() {
    console.log('🌐 Testing VRM system with browser-like requests...');
    
    const testResults = {
        timestamp: new Date().toISOString(),
        tests: {},
        serverRunning: false,
        pagesAccessible: 0,
        totalPages: 0
    };
    
    try {
        // Test server availability
        console.log('📋 Step 1: Testing server availability...');
        const serverTest = await testHTTP('http://localhost:8000/');
        testResults.serverRunning = serverTest.success;
        
        if (!serverTest.success) {
            console.log('❌ Server not running at localhost:8000');
            console.log('   Please run: cd dev/web_viewer && python3 -m http.server 8000');
            return testResults;
        }
        
        console.log('✅ Server running at localhost:8000');
        
        // Test key demo pages
        console.log('📋 Step 2: Testing demo page accessibility...');
        
        const testPages = [
            '/demos/complete_ichika_conversation_system.html',
            '/real_vrm_system_test.html',
            '/vrm_debug_test.html'
        ];
        
        testResults.totalPages = testPages.length;
        
        for (const page of testPages) {
            const url = `http://localhost:8000${page}`;
            const result = await testHTTP(url);
            testResults.tests[page] = result;
            
            if (result.success) {
                testResults.pagesAccessible++;
                console.log(`  ✅ ${page} - Accessible (${result.size} bytes)`);
            } else {
                console.log(`  ❌ ${page} - ${result.error}`);
            }
        }
        
        // Test VRM assets
        console.log('📋 Step 3: Testing VRM asset accessibility...');
        
        const vrmAssets = [
            '/assets/avatars/ichika.vrm',
            '/assets/avatars/buny.vrm', 
            '/assets/avatars/kaede.vrm'
        ];
        
        for (const asset of vrmAssets) {
            const url = `http://localhost:8000${asset}`;
            const result = await testHTTP(url, 'HEAD');
            testResults.tests[asset] = result;
            
            if (result.success) {
                const sizeMB = result.contentLength ? (result.contentLength / 1024 / 1024).toFixed(1) + 'MB' : 'Unknown size';
                console.log(`  ✅ ${asset} - Available (${sizeMB})`);
            } else {
                console.log(`  ❌ ${asset} - ${result.error}`);
            }
        }
        
        // Test VRM component scripts
        console.log('📋 Step 4: Testing VRM component script accessibility...');
        
        const vrmScripts = [
            '/src/components/animation/vrm/VRMLoader.js',
            '/src/components/animation/vrm/AvatarBinder.js',
            '/src/components/animation/vrm/BVHTimelineVRMIntegration.js',
            '/src/components/animation/timeline/BVHTimeline.js',
            '/src/scene/ClassroomAvatarIntegration.js'
        ];
        
        for (const script of vrmScripts) {
            const url = `http://localhost:8000${script}`;
            const result = await testHTTP(url);
            testResults.tests[script] = result;
            
            if (result.success) {
                console.log(`  ✅ ${script} - Accessible (${result.size} bytes)`);
            } else {
                console.log(`  ❌ ${script} - ${result.error}`);
            }
        }
        
        // Generate summary
        const totalTests = Object.keys(testResults.tests).length;
        const successfulTests = Object.values(testResults.tests).filter(test => test.success).length;
        const successRate = (successfulTests / totalTests * 100).toFixed(1);
        
        console.log('📊 TEST SUMMARY:');
        console.log(`   Server Running: ${testResults.serverRunning ? '✅ YES' : '❌ NO'}`);
        console.log(`   Pages Accessible: ${testResults.pagesAccessible}/${testResults.totalPages}`);
        console.log(`   Overall Success: ${successfulTests}/${totalTests} (${successRate}%)`);
        
        testResults.overallSuccess = successRate >= 80;
        
        if (testResults.overallSuccess) {
            console.log('🎉 VRM system is ready for browser testing!');
            console.log('   You can now visit:');
            console.log('   - http://localhost:8000/real_vrm_system_test.html');
            console.log('   - http://localhost:8000/demos/complete_ichika_conversation_system.html');
        } else {
            console.log('⚠️  VRM system has some accessibility issues');
        }
        
        return testResults;
        
    } catch (error) {
        console.error('❌ Browser test failed:', error);
        testResults.error = error.message;
        return testResults;
    }
}

// Helper function to test HTTP endpoints
function testHTTP(url, method = 'GET') {
    return new Promise((resolve) => {
        const urlObj = new URL(url);
        
        const options = {
            hostname: urlObj.hostname,
            port: urlObj.port,
            path: urlObj.pathname,
            method: method
        };
        
        const req = http.request(options, (res) => {
            let data = '';
            
            res.on('data', (chunk) => {
                if (method === 'GET') {
                    data += chunk;
                }
            });
            
            res.on('end', () => {
                resolve({
                    success: res.statusCode >= 200 && res.statusCode < 300,
                    statusCode: res.statusCode,
                    size: data.length,
                    contentLength: res.headers['content-length'],
                    contentType: res.headers['content-type'],
                    data: method === 'GET' ? data.substring(0, 200) + '...' : null
                });
            });
        });
        
        req.on('error', (error) => {
            resolve({
                success: false,
                error: error.message
            });
        });
        
        req.setTimeout(5000, () => {
            req.abort();
            resolve({
                success: false,
                error: 'Request timeout'
            });
        });
        
        req.end();
    });
}

// Run if called directly
if (require.main === module) {
    testVRMSystemWithBrowser().then(results => {
        if (results.overallSuccess) {
            process.exit(0);
        } else {
            process.exit(1);
        }
    }).catch(error => {
        console.error('Test error:', error);
        process.exit(1);
    });
}

module.exports = { testVRMSystemWithBrowser };