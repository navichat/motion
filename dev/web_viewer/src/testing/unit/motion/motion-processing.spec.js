/**
 * Motion Processing System Unit Test
 * Tests BVH parsing, motion analysis, and RSMT motion transition
 */

import { test, expect } from '@playwright/test';

test.describe('Motion Processing System Tests', () => {
  test('should parse BVH motion data', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/motion/bvh-processing-test.html');
    
    const result = await page.evaluate(async () => {
      const bvhProcessor = new window.BVHProcessor();
      
      // Sample BVH header and frame data
      const sampleBVH = `HIERARCHY
ROOT Hips
{
  OFFSET 0.00 0.00 0.00
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Chest
  {
    OFFSET 0.00 5.21 0.00
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.00 2.13 0.00
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
0.00 0.00 0.00 5.00 0.00 0.00 10.00 0.00 0.00`;
      
      try {
        const parsedBVH = await bvhProcessor.parse(sampleBVH);
        
        return {
          parsed: true,
          hasHierarchy: !!parsedBVH.hierarchy,
          hasMotion: !!parsedBVH.motion,
          frameCount: parsedBVH.motion ? parsedBVH.motion.frames : 0,
          frameTime: parsedBVH.motion ? parsedBVH.motion.frameTime : 0,
          jointCount: parsedBVH.hierarchy ? parsedBVH.hierarchy.joints.length : 0,
          hasRootJoint: parsedBVH.hierarchy ? !!parsedBVH.hierarchy.root : false
        };
      } catch (error) {
        return {
          parsed: false,
          error: error.message
        };
      }
    });
    
    expect(result.parsed).toBe(true);
    expect(result.hasHierarchy).toBe(true);
    expect(result.hasMotion).toBe(true);
    expect(result.frameCount).toBe(2);
    expect(result.frameTime).toBeCloseTo(0.033333, 5);
    expect(result.jointCount).toBeGreaterThan(0);
    expect(result.hasRootJoint).toBe(true);
  });

  test('should analyze motion features', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/motion/motion-analysis-test.html');
    
    const result = await page.evaluate(async () => {
      const motionAnalyzer = new window.MotionAnalyzer();
      
      // Create sample motion data (walking pattern)
      const frameCount = 60; // 2 seconds at 30fps
      const motionData = [];
      
      for (let frame = 0; frame < frameCount; frame++) {
        const time = frame / 30.0;
        const walkCycle = Math.sin(time * 2 * Math.PI); // 1 Hz walking
        
        motionData.push({
          frame: frame,
          rootPosition: [0, 0, time * 0.5], // Forward movement
          rootRotation: [0, walkCycle * 0.1, 0], // Side-to-side rotation
          joints: {
            leftHip: [walkCycle * 0.3, 0, 0],
            rightHip: [-walkCycle * 0.3, 0, 0],
            leftKnee: [Math.max(0, walkCycle) * 0.5, 0, 0],
            rightKnee: [Math.max(0, -walkCycle) * 0.5, 0, 0]
          }
        });
      }
      
      try {
        const analysis = await motionAnalyzer.analyze(motionData);
        
        return {
          analyzed: true,
          hasFeatures: !!analysis.features,
          hasVelocity: !!analysis.velocity,
          hasRhythm: !!analysis.rhythm,
          frameCount: analysis.frameCount,
          duration: analysis.duration,
          avgVelocity: analysis.velocity ? analysis.velocity.average : 0,
          rhythmFrequency: analysis.rhythm ? analysis.rhythm.frequency : 0
        };
      } catch (error) {
        return {
          analyzed: false,
          error: error.message
        };
      }
    });
    
    expect(result.analyzed).toBe(true);
    expect(result.hasFeatures).toBe(true);
    expect(result.hasVelocity).toBe(true);
    expect(result.frameCount).toBe(60);
    expect(result.duration).toBeCloseTo(2.0, 1);
    expect(result.avgVelocity).toBeGreaterThan(0);
  });

  test('should perform RSMT motion transition', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/motion/rsmt-transition-test.html');
    
    const result = await page.evaluate(async () => {
      const rsmtProcessor = new window.RSMTProcessor();
      
      // Create source and target motion clips
      const sourceMotion = {
        frames: 30,
        data: Array.from({ length: 30 }, (_, i) => ({
          frame: i,
          pose: [0, 0, 0, Math.sin(i * 0.1), 0, 0] // Simple motion
        }))
      };
      
      const targetMotion = {
        frames: 30,
        data: Array.from({ length: 30 }, (_, i) => ({
          frame: i,
          pose: [0, 0, 0, Math.cos(i * 0.1), 0, 0] // Different motion
        }))
      };
      
      try {
        const transition = await rsmtProcessor.createTransition(sourceMotion, targetMotion, {
          transitionLength: 10, // 10 frame transition
          blendMode: 'linear',
          preserveContactPoints: true
        });
        
        return {
          transitionCreated: true,
          hasTransitionFrames: !!transition.frames,
          transitionLength: transition.frames ? transition.frames.length : 0,
          hasBlendWeights: !!transition.blendWeights,
          startFrame: transition.startFrame,
          endFrame: transition.endFrame,
          isSmooth: transition.smoothness > 0.8
        };
      } catch (error) {
        return {
          transitionCreated: false,
          error: error.message
        };
      }
    });
    
    expect(result.transitionCreated).toBe(true);
    expect(result.hasTransitionFrames).toBe(true);
    expect(result.transitionLength).toBe(10);
    expect(result.hasBlendWeights).toBe(true);
    expect(typeof result.startFrame).toBe('number');
    expect(typeof result.endFrame).toBe('number');
  });

  test('should handle motion retargeting', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/motion/motion-retargeting-test.html');
    
    const result = await page.evaluate(async () => {
      const retargeter = new window.MotionRetargeter();
      
      // Define source and target skeletons
      const sourceSkeleton = {
        joints: ['root', 'spine', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg'],
        hierarchy: {
          root: { children: ['spine', 'leftLeg', 'rightLeg'] },
          spine: { children: ['leftArm', 'rightArm'] }
        },
        restPose: {
          root: [0, 0, 0],
          spine: [0, 1, 0],
          leftArm: [-0.5, 1.5, 0],
          rightArm: [0.5, 1.5, 0],
          leftLeg: [-0.2, 0, 0],
          rightLeg: [0.2, 0, 0]
        }
      };
      
      const targetSkeleton = {
        joints: ['root', 'spine', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg'],
        hierarchy: {
          root: { children: ['spine', 'leftLeg', 'rightLeg'] },
          spine: { children: ['leftArm', 'rightArm'] }
        },
        restPose: {
          root: [0, 0, 0],
          spine: [0, 1.2, 0], // Taller character
          leftArm: [-0.6, 1.8, 0], // Longer arms
          rightArm: [0.6, 1.8, 0],
          leftLeg: [-0.25, 0, 0], // Wider stance
          rightLeg: [0.25, 0, 0]
        }
      };
      
      const sourceMotion = {
        frames: [{
          root: [0, 0, 0, 0, 0, 0],
          spine: [0, 0, 0.1, 0, 0, 0],
          leftArm: [0.2, 0, 0, 0, 0, 0],
          rightArm: [-0.2, 0, 0, 0, 0, 0],
          leftLeg: [0.1, 0, 0, 0, 0, 0],
          rightLeg: [-0.1, 0, 0, 0, 0, 0]
        }]
      };
      
      try {
        const retargetedMotion = await retargeter.retarget(
          sourceMotion,
          sourceSkeleton,
          targetSkeleton
        );
        
        return {
          retargeted: true,
          hasFrames: !!retargetedMotion.frames,
          frameCount: retargetedMotion.frames ? retargetedMotion.frames.length : 0,
          hasAllJoints: retargetedMotion.frames && retargetedMotion.frames[0] ? 
            Object.keys(retargetedMotion.frames[0]).length === targetSkeleton.joints.length : false,
          preservesMotion: true // Would need more complex validation
        };
      } catch (error) {
        return {
          retargeted: false,
          error: error.message
        };
      }
    });
    
    expect(result.retargeted).toBe(true);
    expect(result.hasFrames).toBe(true);
    expect(result.frameCount).toBe(1);
    expect(result.hasAllJoints).toBe(true);
  });
});
