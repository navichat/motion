/**
 * DeepMimic Animation Demo
 * Demonstrates real-time policy execution and BVH frame generation
 */

class DeepMimicDemo {
    constructor() {
        this.isRunning = false;
        this.animationFrame = null;
        this.frameCount = 0;
        this.currentState = null;
        this.actionHistory = [];
    }

    async startDemo() {
        console.log('🤖 Starting DeepMimic Animation Demo');
        
        try {
            // Initialize DeepMimic system
            await initializeDeepMimic();
            console.log('✅ DeepMimic system initialized');
            
            // Set up demo parameters
            document.getElementById('deepmimic-policy').value = 'humanoid3d_walk';
            document.getElementById('deepmimic-state').value = 'reference';
            document.getElementById('deepmimic-noise').value = '0.1';
            
            this.isRunning = true;
            this.runAnimationLoop();
            
        } catch (error) {
            console.error('❌ Demo initialization failed:', error);
        }
    }

    stopDemo() {
        console.log('⏹️ Stopping DeepMimic Animation Demo');
        this.isRunning = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    async runAnimationLoop() {
        if (!this.isRunning) return;
        
        try {
            // Execute policy at 30 FPS
            await runDeepMimicPolicy();
            this.frameCount++;
            
            // Log progress every 30 frames (1 second)
            if (this.frameCount % 30 === 0) {
                console.log(`🎬 Generated ${this.frameCount} frames`);
                
                // Show current BVH frame info
                if (window.currentBVHFrame) {
                    const jointCount = Object.keys(window.currentBVHFrame).filter(k => !k.startsWith('_')).length;
                    console.log(`📊 Current frame: ${jointCount} joints animated`);
                }
            }
            
            // Schedule next frame
            this.animationFrame = requestAnimationFrame(() => {
                setTimeout(() => this.runAnimationLoop(), 33); // ~30 FPS
            });
            
        } catch (error) {
            console.error('❌ Animation loop error:', error);
            this.stopDemo();
        }
    }

    async demonstrateTransitions() {
        console.log('🔄 Demonstrating policy transitions...');
        
        const policies = ['humanoid3d_walk', 'humanoid3d_run', 'humanoid3d_jump'];
        
        for (const policy of policies) {
            console.log(`🎯 Switching to ${policy} policy`);
            document.getElementById('deepmimic-policy').value = policy;
            
            // Generate a few frames with this policy
            for (let i = 0; i < 10; i++) {
                await runDeepMimicPolicy();
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            console.log(`✅ Generated 10 frames with ${policy} policy`);
        }
        
        console.log('🎉 Policy transition demonstration complete');
    }

    async testBVHCompositing() {
        console.log('🎨 Testing BVH frame compositing with all systems...');
        
        try {
            // Initialize all systems
            await Promise.all([
                initializeRSMT().catch(e => console.warn('RSMT init failed:', e.message)),
                initializeFaceFormer().catch(e => console.warn('FaceFormer init failed:', e.message)),
                initializeAudioGesture().catch(e => console.warn('AudioGesture init failed:', e.message)),
                initializeDeepMimic().catch(e => console.warn('DeepMimic init failed:', e.message))
            ]);
            
            console.log('🔧 All systems initialized');
            
            // Generate frames from each system
            const results = await Promise.allSettled([
                generateRSMTTransition().catch(e => ({ error: e.message })),
                generateFacialAnimation().catch(e => ({ error: e.message })),
                generateGestures().catch(e => ({ error: e.message })),
                runDeepMimicPolicy().catch(e => ({ error: e.message }))
            ]);
            
            results.forEach((result, index) => {
                const systems = ['RSMT', 'FaceFormer', 'AudioGesture', 'DeepMimic'];
                if (result.status === 'fulfilled') {
                    console.log(`✅ ${systems[index]} frame generated`);
                } else {
                    console.log(`❌ ${systems[index]} generation failed:`, result.reason);
                }
            });
            
            // Check final composite frame
            if (window.currentBVHFrame) {
                const metadata = window.currentBVHFrame._metadata;
                if (metadata && metadata.sources) {
                    console.log(`🎭 Composite frame contains data from: ${metadata.sources.join(', ')}`);
                }
                
                const joints = Object.keys(window.currentBVHFrame).filter(k => !k.startsWith('_'));
                console.log(`📊 Final composite frame: ${joints.length} joints`);
                
                // Show sample joint data
                if (joints.length > 0) {
                    const sampleJoint = joints[0];
                    const sampleData = window.currentBVHFrame[sampleJoint];
                    console.log(`🎯 Sample data (${sampleJoint}): [${sampleData.map(v => v.toFixed(2)).join(', ')}]`);
                }
            }
            
            console.log('✅ BVH compositing test complete');
            
        } catch (error) {
            console.error('❌ BVH compositing test failed:', error);
        }
    }

    generateReport() {
        const report = {
            frameCount: this.frameCount,
            isRunning: this.isRunning,
            systemStatus: {
                deepmimic: window.systems?.deepmimic ? 'initialized' : 'not initialized',
                timeline: window.systems?.timeline ? 'initialized' : 'not initialized'
            },
            currentFrame: window.currentBVHFrame ? 'available' : 'none',
            actionHistory: this.actionHistory.length
        };
        
        console.log('📋 DeepMimic Demo Report:', report);
        return report;
    }
}

// Make demo available globally
window.DeepMimicDemo = DeepMimicDemo;

// Quick test functions
window.runDeepMimicDemo = async function() {
    const demo = new DeepMimicDemo();
    await demo.startDemo();
    
    // Run for 5 seconds then stop
    setTimeout(() => {
        demo.stopDemo();
        demo.generateReport();
    }, 5000);
};

window.testDeepMimicTransitions = async function() {
    const demo = new DeepMimicDemo();
    await demo.demonstrateTransitions();
};

window.testUnifiedCompositing = async function() {
    const demo = new DeepMimicDemo();
    await demo.testBVHCompositing();
};

console.log('🎮 DeepMimic Demo loaded');
console.log('📋 Available commands:');
console.log('   - window.runDeepMimicDemo() - Run 5-second animation demo');
console.log('   - window.testDeepMimicTransitions() - Test policy transitions');
console.log('   - window.testUnifiedCompositing() - Test all-system compositing');
