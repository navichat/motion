#!/usr/bin/env node
/*
 Verify per-backend audio latency budgets from audio-log-*.json artifacts.
 Each artifact contains metrics (p50, p95, etc). We enforce configurable per-backend (or default) max thresholds.

 Env vars (all optional):
   DEFAULT_MAX_P50_MS (e.g. 5000)
   DEFAULT_MAX_P95_MS (e.g. 8000)
   BACKEND_SPEECH_MAX_P50_MS, BACKEND_SPEECH_MAX_P95_MS
   BACKEND_KOKORO_MAX_P50_MS, BACKEND_KOKORO_MAX_P95_MS
   BACKEND_SPEECHT5_MAX_P50_MS, BACKEND_SPEECHT5_MAX_P95_MS
   (Add more BACKEND_<NAME>_... as new backends appear.)

 Exit code 0 if all within budgets; >0 otherwise.
 Prints a summary table and any violations.
*/
const fs = require('fs');
const path = require('path');

function getEnvNum(name, fallback){
  const v = process.env[name];
  if (v == null || v === '') return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

const DEFAULT_MAX_P50 = getEnvNum('DEFAULT_MAX_P50_MS', 5000);
const DEFAULT_MAX_P95 = getEnvNum('DEFAULT_MAX_P95_MS', 8000);

function backendKeyFromScenario(s){
  const lower = s.toLowerCase();
  if (lower.includes('speech-backend')) return 'speech';
  if (lower.includes('kokoro')) return 'kokoro';
  if (lower.includes('speecht5')) return 'speecht5';
  if (lower.includes('beeps')) return 'beeps';
  return 'other';
}

function resolveBudget(backend, p){
  const upper = backend.toUpperCase();
  const specific = getEnvNum(`BACKEND_${upper}_MAX_${p.toUpperCase()}_MS`, null);
  if (specific != null) return specific;
  return p === 'P50' ? DEFAULT_MAX_P50 : DEFAULT_MAX_P95;
}

function main(){
  const dir = path.join(process.cwd(), 'test-results');
  if (!fs.existsSync(dir)) {
    console.error('[verify-audio-backend-latency] No test-results directory; nothing to verify.');
    process.exit(0);
  }
  const files = fs.readdirSync(dir).filter(f=>/^audio-log-.*\.json$/.test(f));
  if (!files.length) {
    console.warn('[verify-audio-backend-latency] No audio log artifacts found.');
    process.exit(0);
  }
  const rows = [];
  const violations = [];
  for (const f of files){
    try {
      const j = JSON.parse(fs.readFileSync(path.join(dir,f),'utf8'));
      const backend = backendKeyFromScenario(j.scenario || f);
      const m = j.metrics || {};
      const p50 = m.p50 != null ? Number(m.p50) : null;
      const p95 = m.p95 != null ? Number(m.p95) : null;
      const p50Budget = resolveBudget(backend,'P50');
      const p95Budget = resolveBudget(backend,'P95');
      if ((m.samples||0) === 0) {
        violations.push({ backend, file:f, metric:'samples', value:0, limit:'>0', reason:'No latency samples captured' });
      }
      if (p50 != null && p50 > p50Budget) violations.push({ backend, file:f, metric:'p50', value:p50, limit:p50Budget });
      if (p95 != null && p95 > p95Budget) violations.push({ backend, file:f, metric:'p95', value:p95, limit:p95Budget });
      rows.push({ file:f, scenario:j.scenario, backend, samples:m.samples||0, p50, p95, p50Budget, p95Budget });
    } catch (e) {
      violations.push({ backend:'unknown', file:f, metric:'parse', value:'-', limit:'-', reason:'JSON parse error' });
    }
  }
  // Output summary table
  const header = '| Scenario | Backend | Samples | p50 | p50 Budget | p95 | p95 Budget |';
  const sep = '|---|---|---:|---:|---:|---:|---:|';
  const lines = rows.map(r=>`| ${r.scenario} | ${r.backend} | ${r.samples} | ${r.p50??''} | ${r.p50Budget} | ${r.p95??''} | ${r.p95Budget} |`);
  console.log(header); console.log(sep); lines.forEach(l=>console.log(l));
  if (violations.length){
    console.error('\nViolations:');
    for (const v of violations){
      console.error(`- ${v.file} backend=${v.backend} metric=${v.metric} value=${v.value} limit=${v.limit}${v.reason? ' reason='+v.reason:''}`);
    }
    process.exit(2);
  } else {
    console.log('\nAll backend latencies within budgets.');
  }
}

main();
