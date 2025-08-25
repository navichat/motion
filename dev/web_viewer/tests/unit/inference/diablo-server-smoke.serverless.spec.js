import { test, expect } from '@playwright/test';

// Opt-in: hits a DiabloGPT server endpoint when DIABLO_SERVER_URL is set (OpenAI-compatible or simple POST /chat).

test('diablo-gpt server: chat returns text', async ({ request }) => {
  const run = !!process.env.RUN_REAL_INFERENCE && !!process.env.DIABLO_SERVER_URL && !process.env.CI;
  if (!run) test.skip(true, 'Set RUN_REAL_INFERENCE=1 and DIABLO_SERVER_URL to enable');

  const base = process.env.DIABLO_SERVER_URL.replace(/\/$/, '');
  // Try OpenAI-style; fallback to /chat with { prompt }
  const tryOpenAI = async () => {
    const url = base + '/v1/chat/completions';
    const payload = { model: 'diablo', messages: [{ role: 'user', content: 'One short greeting.' }], max_tokens: 16 };
    const res = await request.post(url, { data: payload });
    if (res.ok()) {
      const j = await res.json();
      return j?.choices?.[0]?.message?.content || '';
    }
    return '';
  };
  let text = await tryOpenAI();
  if (!text) {
    const res = await request.post(base + '/chat', { data: { prompt: 'One short greeting.' } });
    if (res.ok()) {
      const j = await res.json();
      text = j?.text || j?.reply || '';
    }
  }
  expect(typeof text).toBe('string');
  expect(text.length).toBeGreaterThan(0);
});
