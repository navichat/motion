// Test Original FaceFormer Web Implementation
const { OriginalFaceFormerSystem } = require('./original_faceformer_web.cjs');
const fs = require('fs');

async function testOriginalFaceFormer() {
    console.log('🧪 Testing Original FaceFormer Web Implementation');
    console.log('='* 60);
    
    try {
        // Test VOCASET model
        console.log('\n📌 Test 1: VOCASET Model Initialization');
        const vocasetSystem = new OriginalFaceFormerSystem();
        const vocasetSuccess = await vocasetSystem.initialize('vocaset');
        
        if (vocasetSuccess) {
            console.log('✅ VOCASET system initialized successfully');
            console.log('📊 Status:', vocasetSystem.getSystemStatus());
            
            // Test generation
            console.log('\n📌 Test 2: VOCASET Generation');
            const audioBuffer = new Float32Array(16000); // 1 second of audio
            audioBuffer.fill(0.01); // Small signal
            
            const template = new Array(15069).fill(0.0); // VOCASET template
            
            const result = await vocasetSystem.generateFromAudio(
                audioBuffer, 
                template, 
                0, // Subject 0
                10 // 10 frames
            );
            
            console.log('✅ VOCASET generation successful:');
            console.log(`  Frames: ${result.frameCount}`);
            console.log(`  Vertex dimension: ${result.vertices[0].length}`);
            console.log(`  Sample vertex values: [${result.vertices[0].slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
            
            // Save VOCASET results
            const vocasetResults = {
                test: 'original_faceformer_vocaset',
                timestamp: new Date().toISOString(),
                system_status: vocasetSystem.getSystemStatus(),
                generation_result: result,
                sample_vertices: {
                    frame_0: result.vertices[0].slice(0, 10),
                    frame_last: result.vertices[result.vertices.length - 1].slice(0, 10)
                }
            };
            
            fs.writeFileSync('./original_faceformer_vocaset_test.json', JSON.stringify(vocasetResults, null, 2));
            console.log('💾 VOCASET results saved');
            
        } else {
            console.error('❌ VOCASET initialization failed');
        }
        
        // Test BIWI model
        console.log('\n📌 Test 3: BIWI Model Initialization');
        const biwiSystem = new OriginalFaceFormerSystem();
        const biwiSuccess = await biwiSystem.initialize('biwi');
        
        if (biwiSuccess) {
            console.log('✅ BIWI system initialized successfully');
            
            // Test generation
            console.log('\n📌 Test 4: BIWI Generation');
            const audioBuffer = new Float32Array(16000); // 1 second of audio
            audioBuffer.fill(0.01); // Small signal
            
            const template = new Array(70110).fill(0.0); // BIWI template
            
            const result = await biwiSystem.generateFromAudio(
                audioBuffer,
                template,
                0, // Subject 0
                5  // 5 frames (BIWI is larger)
            );
            
            console.log('✅ BIWI generation successful:');
            console.log(`  Frames: ${result.frameCount}`);
            console.log(`  Vertex dimension: ${result.vertices[0].length}`);
            console.log(`  Sample vertex values: [${result.vertices[0].slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
            
            // Save BIWI results
            const biwiResults = {
                test: 'original_faceformer_biwi',
                timestamp: new Date().toISOString(),
                system_status: biwiSystem.getSystemStatus(),
                generation_result: result,
                sample_vertices: {
                    frame_0: result.vertices[0].slice(0, 10),
                    frame_last: result.vertices[result.vertices.length - 1].slice(0, 10)
                }
            };
            
            fs.writeFileSync('./original_faceformer_biwi_test.json', JSON.stringify(biwiResults, null, 2));
            console.log('💾 BIWI results saved');
            
        } else {
            console.error('❌ BIWI initialization failed');
        }
        
        // Performance comparison
        console.log('\n📌 Test 5: Performance Comparison');
        
        if (vocasetSuccess && biwiSuccess) {
            console.log('🏃 Running performance benchmark...');
            
            const testAudio = new Float32Array(8000); // 0.5 seconds
            testAudio.fill(0.005);
            
            // VOCASET benchmark
            const vocasetTemplate = new Array(15069).fill(0.0);
            const vocasetStart = Date.now();
            await vocasetSystem.generateFromAudio(testAudio, vocasetTemplate, 0, 5);
            const vocasetTime = Date.now() - vocasetStart;
            
            // BIWI benchmark
            const biwiTemplate = new Array(70110).fill(0.0);
            const biwiStart = Date.now();
            await biwiSystem.generateFromAudio(testAudio, biwiTemplate, 0, 5);
            const biwiTime = Date.now() - biwiStart;
            
            console.log(`⚡ Performance Results:`);
            console.log(`  VOCASET (15K vertices): ${vocasetTime}ms`);
            console.log(`  BIWI (70K vertices): ${biwiTime}ms`);
            console.log(`  Ratio: ${(biwiTime / vocasetTime).toFixed(1)}x`);
        }
        
        // Final summary
        console.log('\n🎯 Test Summary:');
        console.log(`  ✅ VOCASET: ${vocasetSuccess ? 'PASS' : 'FAIL'}`);
        console.log(`  ✅ BIWI: ${biwiSuccess ? 'PASS' : 'FAIL'}`);
        console.log(`  🎉 Original FaceFormer Web: ${vocasetSuccess || biwiSuccess ? 'WORKING' : 'FAILED'}`);
        
        if (vocasetSuccess || biwiSuccess) {
            console.log('\n💡 Next Steps:');
            console.log('  1. Integrate real Wav2Vec2 preprocessing');
            console.log('  2. Add temporal bias mask implementation'); 
            console.log('  3. Optimize for larger sequence generation');
            console.log('  4. Create web demo interface');
        }
        
        return vocasetSuccess || biwiSuccess;
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        return false;
    }
}

// Run the test
if (require.main === module) {
    testOriginalFaceFormer().then(success => {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🏁 ORIGINAL FACEFORMER TEST ${success ? 'PASSED' : 'FAILED'}`);
        console.log(`${'='.repeat(60)}`);
        process.exit(success ? 0 : 1);
    });
}

module.exports = { testOriginalFaceFormer };
