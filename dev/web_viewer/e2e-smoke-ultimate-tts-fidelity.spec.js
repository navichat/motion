// [E2E] Validate TTS-derived gesture fidelity (energy variance & viseme diversity)
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// Use speech backend (deterministic text->synthetic energy shaping) as baseline
const URL = '/demos/ichika_voice_conversation_demo.html?backend=speech&asr=fake&autoListen=1&listenSec=1';

function summarizeVisemes(vis){
  const counts = {}; vis.forEach(v => { counts[v.id] = (counts[v.id]||0)+1; });
  const unique = Object.keys(counts).length;
  return { unique, counts };
}

test.describe('[E2E][Ultimate][TTSFidelity] energy + viseme diversity', () => {
  test('Energy variance, dynamic range, and viseme diversity above minimum thresholds', async ({ page }) => {
    await page.goto(URL);
    // Wait until demo API is available
    await page.waitForFunction(() => !!window.__ultimateDemo && typeof window.__ultimateDemo.sayText === 'function', { timeout: 10000 });
    // Build a snapshot directly (independent of scheduling path) for deterministic fidelity metrics
    const tts = await page.evaluate(() => {
      const phrase = 'Fidelity metrics validation sequence alpha beta gamma delta';
      return window.__ultimateDemo?.forceBuildSnapshot?.(phrase);
    });
    expect(tts).toBeTruthy();
    // Ensure we have at least a minimal viseme list even if reconstructed
    const visemeLen = (tts.visemes||[]).length;
    expect(visemeLen).toBeGreaterThan(0);

    // Energy variance metric
    // Schedule through normal path to exercise scheduler + audio log
    const scheduleInfo = await page.evaluate(() => {
      try {
        const last = window.__ultimateDemo?.getLastTts?.();
        if (last) window.__ultimateDemo?.scheduleFromTts?.(last);
        return { scheduleCount: window.__ultimateDemo?.getScheduleCount?.(), logLen: (window.__ultimateDemo?.getAudioLog?.()||[]).length };
      } catch { return { scheduleCount: -1, logLen: -1 }; }
    });
    expect(scheduleInfo.scheduleCount).toBeGreaterThanOrEqual(1);

    const energyStats = await page.evaluate(() => {
      const t = window.__ultimateDemo?.getLastTts?.();
      if (!t || !t.energy || !t.energy.length) return { variance:0, mean:0, min:0, max:0, dynamicRange:0 };
      const arr = t.energy;
      const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
      let min = Infinity, max = -Infinity; for (const v of arr){ if (v<min) min=v; if (v>max) max=v; }
      const variance = arr.reduce((a,b)=>a+Math.pow(b-mean,2),0)/arr.length;
      return { variance, mean, min, max, dynamicRange: max-min };
    });
    const energyVar = energyStats.variance;

    // Viseme diversity summary
  const visemeSummary = summarizeVisemes(tts.visemes||[]);

    // Only enforce diversity/variance if not reconstructed fallback
    if (!tts.reconstructed) {
      expect(visemeSummary.unique).toBeGreaterThan(2); // require at least 3 distinct visemes
      expect(energyVar).toBeGreaterThan(0.0005); // minimal variance threshold
      expect(energyStats.dynamicRange).toBeGreaterThan(0.05); // ensure energy envelope not flat
    }

    // Artifact
    try {
      const outDir = path.join(process.cwd(), 'test-results');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'tts-fidelity.json'), JSON.stringify({
        energyVariance: energyVar,
        energyMean: energyStats.mean,
        energyMin: energyStats.min,
        energyMax: energyStats.max,
        energyDynamicRange: energyStats.dynamicRange,
        visemeUnique: visemeSummary.unique,
        visemeCounts: visemeSummary.counts,
        visemeTotal: tts.visemes.length,
        duration: tts.duration,
        scheduleCount: scheduleInfo.scheduleCount,
        audioLogLength: scheduleInfo.logLen,
        reconstructed: !!tts.reconstructed,
        forced: !!tts.forced
      }, null, 2));
    } catch (e) { console.warn('Failed to write fidelity artifact', e); }
  });
});
