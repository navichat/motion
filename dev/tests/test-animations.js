#!/usr/bin/env node

/**
 * Animation Testing Demo - Motion Viewer
 * Tests actual animation loading and validation
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class AnimationTester {
    constructor() {
        this.animationsPath = path.join(__dirname, '../../chat/assets/animations');
        this.results = {
            tested: 0,
            passed: 0,
            failed: 0,
            animations: []
        };
    }

    async testAllAnimations() {
        console.log('🎭 Animation Testing Demo - Motion Viewer');
        console.log('==========================================');
        console.log(`📁 Testing animations from: ${this.animationsPath}`);
        console.log('');

        const animationFiles = fs.readdirSync(this.animationsPath)
            .filter(file => file.endsWith('.json') && file !== 'manifest.json');

        for (const file of animationFiles) {
            await this.testAnimation(file);
        }

        this.printResults();
    }

    async testAnimation(filename) {
        const filePath = path.join(this.animationsPath, filename);
        const animationName = filename.replace('.json', '');
        
        try {
            console.log(`🧪 Testing: ${animationName}`);
            
            // Load animation data
            const animationData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
            
            // Validate animation structure
            const validation = this.validateAnimation(animationData, animationName);
            
            if (validation.valid) {
                console.log(`   ✅ Valid animation - ${validation.info}`);
                this.results.passed++;
            } else {
                console.log(`   ❌ Invalid animation - ${validation.error}`);
                this.results.failed++;
            }

            this.results.animations.push({
                name: animationName,
                valid: validation.valid,
                info: validation.info || validation.error,
                ...validation.details
            });

        } catch (error) {
            console.log(`   ❌ Failed to load - ${error.message}`);
            this.results.failed++;
        }

        this.results.tested++;
    }

    validateAnimation(data, name) {
        const details = {};

        // Check for required structure
        if (!data.body) {
            return { valid: false, error: 'Missing body structure' };
        }

        const body = data.body;

        // Validate metadata
        if (!body.fps || !body.frames || !body.duration) {
            return { valid: false, error: 'Missing animation metadata (fps/frames/duration)' };
        }

        details.fps = body.fps;
        details.frames = body.frames;
        details.duration = body.duration;

        // Validate tracks
        if (!body.tracks || !Array.isArray(body.tracks)) {
            return { valid: false, error: 'Missing or invalid tracks array' };
        }

        details.trackCount = body.tracks.length;

        // Count joint types
        const jointTypes = {};
        body.tracks.forEach(track => {
            const jointName = track.key.split('.')[0];
            jointTypes[jointName] = (jointTypes[jointName] || 0) + 1;
        });

        details.joints = Object.keys(jointTypes).length;
        details.jointTypes = jointTypes;

        // Validate track data
        let hasData = false;
        if (body.data && Array.isArray(body.data) && body.data.length > 0) {
            hasData = true;
            details.dataPoints = body.data.length;
        }

        const info = `${details.frames} frames, ${details.fps}fps, ${details.joints} joints, ${details.trackCount} tracks${hasData ? `, ${details.dataPoints} data points` : ''}`;

        return { 
            valid: true, 
            info,
            details 
        };
    }

    printResults() {
        console.log('');
        console.log('🎯 ANIMATION TESTING RESULTS');
        console.log('============================');
        console.log(`📊 Total Tested: ${this.results.tested}`);
        console.log(`✅ Passed: ${this.results.passed}`);
        console.log(`❌ Failed: ${this.results.failed}`);
        console.log(`📈 Success Rate: ${((this.results.passed / this.results.tested) * 100).toFixed(1)}%`);
        console.log('');

        if (this.results.passed > 0) {
            console.log('🏆 VALID ANIMATIONS FOUND:');
            this.results.animations
                .filter(anim => anim.valid)
                .forEach(anim => {
                    console.log(`   📽️  ${anim.name}: ${anim.info}`);
                });
            console.log('');
        }

        if (this.results.failed > 0) {
            console.log('⚠️  ISSUES FOUND:');
            this.results.animations
                .filter(anim => !anim.valid)
                .forEach(anim => {
                    console.log(`   ❌ ${anim.name}: ${anim.info}`);
                });
            console.log('');
        }

        // Summary stats
        if (this.results.passed > 0) {
            const validAnimations = this.results.animations.filter(a => a.valid);
            const totalFrames = validAnimations.reduce((sum, a) => sum + (a.frames || 0), 0);
            const totalDuration = validAnimations.reduce((sum, a) => sum + (a.duration || 0), 0);
            const totalJoints = validAnimations.reduce((sum, a) => sum + (a.joints || 0), 0);

            console.log('📈 ANIMATION STATISTICS:');
            console.log(`   🎬 Total Animation Frames: ${totalFrames.toLocaleString()}`);
            console.log(`   ⏱️  Total Duration: ${totalDuration.toFixed(1)} seconds`);
            console.log(`   🦴 Average Joints per Animation: ${(totalJoints / validAnimations.length).toFixed(1)}`);
            console.log(`   📊 Average FPS: ${(validAnimations.reduce((sum, a) => sum + (a.fps || 0), 0) / validAnimations.length).toFixed(1)}`);
        }

        console.log('');
        console.log('🎉 Animation testing complete!');
        
        if (this.results.passed === this.results.tested) {
            console.log('🏅 ALL ANIMATIONS ARE VALID! Ready for 3D playback!');
        }
    }
}

// Run the animation tests
const tester = new AnimationTester();
tester.testAllAnimations().catch(console.error);
