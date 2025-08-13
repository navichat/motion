#!/usr/bin/env node
/**
 * Generate a consolidated JSON payload for webhooks (Slack, etc.).
 * Output: test-results/perf-webhook-payload.json (unless OUT env overrides)
 * Structure:
 * {
 *   generatedAt, gitSha,
 *   aggregate: { samples, avg, p50, p90, p95, p99, last },
 *   baselineDiff: { rawMarkdown },
 *   backends: { rows: [...], violations: [...], ok },
 *   audioScenarios: [ { scenario, backend, samples, p50, p90, p95, p99, avg, hasLatency } ],
 *   summary: { ok, failingBackends, totalViolations }
 * }
 *
 * Requires that prior steps produced:
 *  - test-results/perf-latency.json
 *  - (optionally) test-results/per-backend-latency.json (if not, script will generate via verifier in soft mode)
 *  - audio-log-*.json artifacts for audio scenario metrics
 */
const fs=require('fs');
const path=require('path');
const { execSync } = require('child_process');
function loadJSON(p){try{return JSON.parse(fs.readFileSync(p,'utf8'));}catch{return null;}}
function run(cmd){try{return execSync(cmd,{stdio:['ignore','pipe','ignore']}).toString();}catch{return ''}}
const outDir=path.join(process.cwd(),'test-results');
if(!fs.existsSync(outDir)) fs.mkdirSync(outDir,{recursive:true});
const perf=loadJSON(path.join(outDir,'perf-latency.json'))||{};
let backends=loadJSON(path.join(outDir,'per-backend-latency.json'));
if(!backends){
  // Produce backend JSON via verifier soft mode
  run('BACKEND_BUDGET_SOFT=1 PER_BACKEND_LATENCY_JSON=test-results/per-backend-latency.json node dev/web_viewer/scripts/verify-audio-backend-latency.js');
  backends=loadJSON(path.join(outDir,'per-backend-latency.json'))||{rows:[],violations:[],ok:true};
}
// Collect audio scenario metrics
const audioScenarios=[];
for(const f of fs.readdirSync(outDir).filter(f=>/^audio-log-.*\.json$/.test(f))){
  const j=loadJSON(path.join(outDir,f)); if(!j) continue;
  const m=j.metrics||{};
  audioScenarios.push({
    scenario:j.scenario,
    backend:j.backend||null,
    samples:m.samples||0,
    avg:m.avgLatencyMs||0,
    p50:m.p50||0,
    p90:m.p90||0,
    p95:m.p95||0,
    p99:m.p99||0,
    hasLatency: !!j.latencyEntry
  });
}
const diffMd=run('node dev/web_viewer/scripts/diff-latency-baseline.js');
const aggMetrics = perf.metrics||{};
const payload={
  generatedAt:new Date().toISOString(),
  gitSha: process.env.GITHUB_SHA || 'local',
  aggregate:{
    samples:aggMetrics.samples||0,
    avg:aggMetrics.avgLatencyMs||0,
    p50:aggMetrics.p50||0,
    p90:aggMetrics.p90||0,
    p95:aggMetrics.p95||0,
    p99:aggMetrics.p99||0,
    last:aggMetrics.lastLatencyMs||0
  },
  baselineDiff:{ rawMarkdown: diffMd.trim() },
  backends:{ rows: backends.rows||[], violations: backends.violations||[], ok: backends.ok!==false },
  audioScenarios,
  summary:{
    ok: (backends.ok!==false),
    failingBackends:[...new Set((backends.violations||[]).map(v=>v.backend))],
    totalViolations:(backends.violations||[]).length
  }
};
const outPath= process.env.OUT || path.join(outDir,'perf-webhook-payload.json');
fs.writeFileSync(outPath, JSON.stringify(payload,null,2));
console.log('Wrote webhook payload to', outPath);
