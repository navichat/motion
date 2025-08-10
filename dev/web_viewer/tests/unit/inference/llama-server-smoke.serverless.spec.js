import { test, expect } from '@playwright/test';

// Opt-in: hits an OpenAI-compatible LLaMA server endpoint when LLAMA_SERVER_URL is set.

test('llama server: chat completion returns text', async ({ request }) => {
  const run = !!process.env.RUN_REAL_INFERENCE && !!process.env.LLAMA_SERVER_URL && !process.env.CI;
  if (!run) test.skip(true, 'Set RUN_REAL_INFERENCE=1 and LLAMA_SERVER_URL to enable');

  const url = process.env.LLAMA_SERVER_URL.replace(/\/$/, '') + '/v1/chat/completions';
  const payload = {
    model: 'llama',
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: 'Say hello.' }
    ],
    max_tokens: 16,
    temperature: 0
  };
  const res = await request.post(url, { data: payload });
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  const text = body?.choices?.[0]?.message?.content || '';
  expect(typeof text).toBe('string');
  expect(text.length).toBeGreaterThan(0);
});
