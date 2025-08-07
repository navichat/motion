/**
 * AI Model Jobs Unit Test
 * Tests the AI model job creation and scheduling functionality
 */

import { test, expect } from '@playwright/test';

test.describe('AI Model Jobs Tests', () => {
  test('should create TinyLlama job correctly', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/ai/ai-model-jobs-test.html');
    
    const result = await page.evaluate(() => {
      // Test TinyLlama job creation
      const jobFactory = window.AIModelJobFactory;
      const job = jobFactory.createTinyLlamaJob({
        prompt: "Hello world",
        maxTokens: 50
      });
      
      return {
        type: job.type,
        hasPrompt: !!job.prompt,
        hasMaxTokens: !!job.maxTokens
      };
    });
    
    expect(result.type).toBe('TinyLlamaJob');
    expect(result.hasPrompt).toBe(true);
    expect(result.hasMaxTokens).toBe(true);
  });

  test('should create Whisper job correctly', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/ai/ai-model-jobs-test.html');
    
    const result = await page.evaluate(() => {
      const jobFactory = window.AIModelJobFactory;
      const job = jobFactory.createWhisperJob({
        audioData: new Float32Array(1000),
        sampleRate: 16000
      });
      
      return {
        type: job.type,
        hasAudioData: !!job.audioData,
        sampleRate: job.sampleRate
      };
    });
    
    expect(result.type).toBe('WhisperJob');
    expect(result.hasAudioData).toBe(true);
    expect(result.sampleRate).toBe(16000);
  });

  test('should create RSMT job correctly', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/ai/ai-model-jobs-test.html');
    
    const result = await page.evaluate(() => {
      const jobFactory = window.AIModelJobFactory;
      const job = jobFactory.createRSMTJob({
        sourceMotion: 'walk',
        targetStyle: 'happy',
        transitionFrames: 30
      });
      
      return {
        type: job.type,
        sourceMotion: job.sourceMotion,
        targetStyle: job.targetStyle,
        transitionFrames: job.transitionFrames
      };
    });
    
    expect(result.type).toBe('RSMTJob');
    expect(result.sourceMotion).toBe('walk');
    expect(result.targetStyle).toBe('happy');
    expect(result.transitionFrames).toBe(30);
  });
});
