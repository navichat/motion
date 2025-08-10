const { test, expect } = require('@playwright/test');

test('ASR/VAD/TTS stubs emit expected metadata', async () => {
  const { WhisperStubTask } = require('../../../src/components/ai/tasks/WhisperStubTask.js');
  const { SileroVadStubTask } = require('../../../src/components/ai/tasks/SileroVadStubTask.js');
  const { SpeechT5StubTask } = require('../../../src/components/ai/tasks/SpeechT5StubTask.js');

  const asr = new WhisperStubTask({ text: 'test phrase' });
  const vad = new SileroVadStubTask({ isSpeech: true });
  const tts = new SpeechT5StubTask({ text: 'hi' });

  const a = await (async () => (await asr.run().next()).value)();
  const v = await (async () => (await vad.run().next()).value)();
  const s = await (async () => (await tts.run().next()).value)();

  expect(a.metadata.text).toBe('test phrase');
  expect(v.metadata.speech).toBe(true);
  expect(s.metadata.model).toBe('speecht5_stub');
  expect(s.metadata.samples).toBeGreaterThan(0);
});
