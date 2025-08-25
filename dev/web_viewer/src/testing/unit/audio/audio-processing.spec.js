/**
 * Audio Processing System Unit Test
 * Tests TTS, speech recognition, and audio processing
 */

import { test, expect } from '@playwright/test';

test.describe('Audio Processing System Tests', () => {
  test('should process audio with Kokoro TTS', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/audio/tts-processing-test.html');
    
    const result = await page.evaluate(async () => {
      const kokoroTTS = new window.KokoroTTS();
      
      try {
        const audioData = await kokoroTTS.synthesize(
          "Hello, this is a test message", 
          { 
            voice: 'neutral',
            speed: 1.0,
            pitch: 1.0
          }
        );
        
        return {
          synthesized: true,
          hasAudioData: !!audioData,
          audioLength: audioData ? audioData.length : 0,
          isFloat32Array: audioData instanceof Float32Array
        };
      } catch (error) {
        return {
          synthesized: false,
          error: error.message
        };
      }
    });
    
    expect(result.synthesized).toBe(true);
    expect(result.hasAudioData).toBe(true);
    expect(result.audioLength).toBeGreaterThan(0);
    expect(result.isFloat32Array).toBe(true);
  });

  test('should detect voice activity with VAD', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/audio/vad-processing-test.html');
    
    const result = await page.evaluate(async () => {
      const vadProcessor = new window.VADProcessor();
      
      // Create mock audio with voice activity
      const sampleRate = 16000;
      const duration = 1.0; // 1 second
      const audioData = new Float32Array(sampleRate * duration);
      
      // Add some mock voice activity (sine wave)
      for (let i = 0; i < audioData.length; i++) {
        audioData[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.5;
      }
      
      try {
        const vadResult = await vadProcessor.detectVoiceActivity(audioData, sampleRate);
        
        return {
          processed: true,
          hasVoiceActivity: vadResult.hasVoice,
          confidence: vadResult.confidence,
          segments: vadResult.segments ? vadResult.segments.length : 0
        };
      } catch (error) {
        return {
          processed: false,
          error: error.message
        };
      }
    });
    
    expect(result.processed).toBe(true);
    expect(typeof result.hasVoiceActivity).toBe('boolean');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  test('should transcribe audio with Whisper', async ({ page }) => {
    await page.goto('http://localhost:8081/tests/unit/audio/whisper-processing-test.html');
    
    const result = await page.evaluate(async () => {
      const whisperProcessor = new window.WhisperProcessor();
      
      // Create mock audio data
      const audioData = new Float32Array(16000); // 1 second at 16kHz
      
      try {
        const transcription = await whisperProcessor.transcribe(audioData, {
          language: 'en',
          task: 'transcribe'
        });
        
        return {
          transcribed: true,
          hasText: !!transcription.text,
          textLength: transcription.text ? transcription.text.length : 0,
          hasConfidence: !!transcription.confidence,
          hasSegments: !!transcription.segments
        };
      } catch (error) {
        return {
          transcribed: false,
          error: error.message
        };
      }
    });
    
    expect(result.transcribed).toBe(true);
    expect(result.hasText).toBe(true);
    expect(result.hasConfidence).toBe(true);
  });
});
