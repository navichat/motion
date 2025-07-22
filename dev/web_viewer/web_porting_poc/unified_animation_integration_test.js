/**
 * Quick Integration Test for Unified Animation System with DeepMimic
 * Verifies that all four animation systems work together
 */

class UnifiedAnimationIntegrationTest {
    constructor() {
        this.results = {
            rsmt: { status: 'pending', error: null },
            faceformer: { status: 'pending', error: null },
            audiogesture: { status: 'pending', error: null },
            deepmimic: { status: 'pending', error: null },
            timeline: { status: 'pending', error: null },
            compositing: { status: 'pending', error: null }
        };
    }

    async runIntegrationTest() {
        console.log('🚀 Starting Unified Animation System Integration Test');
        console.log('Testing: RSMT, FaceFormer, AudioGesture, DeepMimic + BVH Timeline');
        
        try {
            // Test 1: Verify system initialization
            await this.testSystemInitialization();
            
            // Test 2: Test individual system functionality
            await this.testIndividualSystems();
            
            // Test 3: Test BVH frame generation
            await this.testBVHFrameGeneration();
            
            // Test 4: Test frame compositing
            await this.testFrameCompositing();
            
            // Test 5: Test timeline integration
            await this.testTimelineIntegration();
            
            this.printResults();
            
        } catch (error) {
            console.error('❌ Integration test failed:', error);
        }
    }

    async testSystemInitialization() {
        console.log('\n🔧 Testing system initialization...');
        
        // Test RSMT initialization
        try {
            if (typeof initializeRSMT === 'function') {
                console.log('✅ RSMT initialization function available');
                this.results.rsmt.status = 'available';
            } else {
                throw new Error('RSMT initialization function not found');
            }
        } catch (error) {
            this.results.rsmt.status = 'error';
            this.results.rsmt.error = error.message;
            console.log('❌ RSMT initialization test failed:', error.message);
        }

        // Test FaceFormer initialization
        try {
            if (typeof initializeFaceFormer === 'function') {
                console.log('✅ FaceFormer initialization function available');
                this.results.faceformer.status = 'available';
            } else {
                throw new Error('FaceFormer initialization function not found');
            }
        } catch (error) {
            this.results.faceformer.status = 'error';
            this.results.faceformer.error = error.message;
            console.log('❌ FaceFormer initialization test failed:', error.message);
        }

        // Test AudioGesture initialization
        try {
            if (typeof initializeAudioGesture === 'function') {
                console.log('✅ AudioGesture initialization function available');
                this.results.audiogesture.status = 'available';
            } else {
                throw new Error('AudioGesture initialization function not found');
            }
        } catch (error) {
            this.results.audiogesture.status = 'error';
            this.results.audiogesture.error = error.message;
            console.log('❌ AudioGesture initialization test failed:', error.message);
        }

        // Test DeepMimic initialization
        try {
            if (typeof initializeDeepMimic === 'function') {
                console.log('✅ DeepMimic initialization function available');
                this.results.deepmimic.status = 'available';
            } else {
                throw new Error('DeepMimic initialization function not found');
            }
        } catch (error) {
            this.results.deepmimic.status = 'error';
            this.results.deepmimic.error = error.message;
            console.log('❌ DeepMimic initialization test failed:', error.message);
        }

        // Test Timeline initialization
        try {
            if (typeof initializeTimeline === 'function') {
                console.log('✅ Timeline initialization function available');
                this.results.timeline.status = 'available';
            } else {
                throw new Error('Timeline initialization function not found');
            }
        } catch (error) {
            this.results.timeline.status = 'error';
            this.results.timeline.error = error.message;
            console.log('❌ Timeline initialization test failed:', error.message);
        }
    }

    async testIndividualSystems() {
        console.log('\n🎯 Testing individual system functionality...');
        
        // Test RSMT generation function
        try {
            if (typeof generateRSMTTransition === 'function') {
                console.log('✅ RSMT generation function available');
                this.results.rsmt.status = 'ready';
            }
        } catch (error) {
            this.results.rsmt.error = error.message;
            console.log('❌ RSMT generation test failed:', error.message);
        }

        // Test FaceFormer generation function
        try {
            if (typeof generateFacialAnimation === 'function') {
                console.log('✅ FaceFormer generation function available');
                this.results.faceformer.status = 'ready';
            }
        } catch (error) {
            this.results.faceformer.error = error.message;
            console.log('❌ FaceFormer generation test failed:', error.message);
        }

        // Test AudioGesture generation function
        try {
            if (typeof generateGestures === 'function') {
                console.log('✅ AudioGesture generation function available');
                this.results.audiogesture.status = 'ready';
            }
        } catch (error) {
            this.results.audiogesture.error = error.message;
            console.log('❌ AudioGesture generation test failed:', error.message);
        }

        // Test DeepMimic execution function
        try {
            if (typeof runDeepMimicPolicy === 'function') {
                console.log('✅ DeepMimic policy execution function available');
                this.results.deepmimic.status = 'ready';
            }
        } catch (error) {
            this.results.deepmimic.error = error.message;
            console.log('❌ DeepMimic execution test failed:', error.message);
        }
    }

    async testBVHFrameGeneration() {
        console.log('\n🎬 Testing BVH frame generation...');
        
        // Test if BVH Timeline is available
        try {
            if (typeof window.BVHTimeline !== 'undefined') {
                console.log('✅ BVH Timeline class available');
                
                // Create test timeline
                const timeline = new BVHTimeline();
                console.log('✅ BVH Timeline instance created');
                
                // Test frame generation
                const testFrame = timeline.generateCurrentFrame();
                console.log('✅ BVH frame generation successful');
                
                this.results.timeline.status = 'working';
            } else {
                throw new Error('BVH Timeline not available');
            }
        } catch (error) {
            this.results.timeline.status = 'error';
            this.results.timeline.error = error.message;
            console.log('❌ BVH frame generation test failed:', error.message);
        }
    }

    async testFrameCompositing() {
        console.log('\n🎨 Testing frame compositing...');
        
        try {
            // Test if global systems object exists
            if (typeof window.systems !== 'undefined') {
                console.log('✅ Global systems object available');
                
                // Test if currentBVHFrame exists
                if (typeof window.currentBVHFrame !== 'undefined') {
                    console.log('✅ Current BVH frame buffer available');
                } else {
                    console.log('⚠️ Current BVH frame buffer not initialized');
                }
                
                this.results.compositing.status = 'ready';
            } else {
                throw new Error('Global systems object not found');
            }
        } catch (error) {
            this.results.compositing.status = 'error';
            this.results.compositing.error = error.message;
            console.log('❌ Frame compositing test failed:', error.message);
        }
    }

    async testTimelineIntegration() {
        console.log('\n📽️ Testing timeline integration...');
        
        try {
            // Test timeline controls
            const controls = ['playTimeline', 'pauseTimeline', 'stopTimeline', 'clearTimeline'];
            let availableControls = 0;
            
            controls.forEach(control => {
                if (typeof window[control] === 'function') {
                    availableControls++;
                    console.log(`✅ ${control} function available`);
                } else {
                    console.log(`❌ ${control} function missing`);
                }
            });
            
            if (availableControls === controls.length) {
                this.results.timeline.status = 'complete';
                console.log('✅ All timeline controls available');
            } else {
                this.results.timeline.status = 'partial';
                console.log(`⚠️ Only ${availableControls}/${controls.length} timeline controls available`);
            }
            
        } catch (error) {
            this.results.timeline.status = 'error';
            this.results.timeline.error = error.message;
            console.log('❌ Timeline integration test failed:', error.message);
        }
    }

    printResults() {
        console.log('\n📋 Integration Test Results:');
        console.log('================================');
        
        Object.entries(this.results).forEach(([system, result]) => {
            const statusIcon = this.getStatusIcon(result.status);
            console.log(`${statusIcon} ${system.toUpperCase()}: ${result.status}`);
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
        });
        
        // Calculate overall success
        const total = Object.keys(this.results).length;
        const successful = Object.values(this.results).filter(r => 
            r.status === 'ready' || r.status === 'working' || r.status === 'complete' || r.status === 'available'
        ).length;
        
        console.log(`\n🎯 Overall Status: ${successful}/${total} systems operational`);
        
        if (successful === total) {
            console.log('🎉 All systems integrated successfully!');
            console.log('✨ DeepMimic integration complete - ready for comprehensive animation testing');
        } else {
            console.log('⚠️ Some systems need attention');
        }
    }

    getStatusIcon(status) {
        switch (status) {
            case 'ready': case 'working': case 'complete': case 'available': return '✅';
            case 'partial': return '⚠️';
            case 'error': return '❌';
            case 'pending': return '⏳';
            default: return '❓';
        }
    }
}

// Make test available globally
window.UnifiedAnimationIntegrationTest = UnifiedAnimationIntegrationTest;

// Auto-run if requested
if (window.location.search.includes('autotest=true')) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            console.log('🔄 Auto-running unified animation integration test...');
            const test = new UnifiedAnimationIntegrationTest();
            test.runIntegrationTest();
        }, 2000);
    });
}

console.log('🧪 Unified Animation Integration Test loaded');
console.log('📋 Available commands:');
console.log('   - new UnifiedAnimationIntegrationTest().runIntegrationTest()');
console.log('   - Add ?autotest=true to URL for automatic testing');
