#!/usr/bin/env node
/**
 * Generate a Markdown performance comment aggregating:
 *  - Aggregate latency metrics
 *  - Aggregate latency diff vs baseline
 *  - Per-backend budgets table
 *  - Recent history (aggregate + backend)
 * Intended for piping into a PR comment (GitHub CLI or API).
 *
 * Env (optional):
 *  LATENCY_HISTORY_ROWS, BACKEND_HISTORY_ROWS to control table depth.
 */
const fs=require('fs');
const path=require('path');
const { execSync } = require('child_process');
function safeReadJSON(p){try{return JSON.parse(fs.readFileSync(p,'utf8'));}catch{return null;}}
function run(cmd){try{return execSync(cmd,{stdio:['ignore','pipe','ignore']}).toString().trim();}catch{return '';}}

const lines=[];
lines.push('### 🤖 Performance Summary');

// Aggregate latency
const perfPath=path.join('test-results','perf-latency.json');
const perf=safeReadJSON(perfPath);
if(perf){
  const m=perf.metrics||{};
  lines.push('\n**Aggregate Latency**');
  lines.push(`Samples **${m.samples}** | Avg ${m.avgLatencyMs?.toFixed?.(1)} ms | P50 ${m.p50} ms | P95 ${m.p95} ms | P99 ${m.p99} ms | Last ${m.lastLatencyMs} ms`);
}

// Diff vs baseline
const diff=run('node dev/web_viewer/scripts/diff-latency-baseline.js');
if(diff){
  lines.push('\n**Baseline Diff**');
  lines.push('```');
  lines.push(diff);
  lines.push('```');
}

// Per-backend budgets table (re-run in soft mode to avoid failure)
const backendTable=run('BACKEND_BUDGET_SOFT=1 node dev/web_viewer/scripts/verify-audio-backend-latency.js');
if(backendTable){
  // Extract just the table region (lines starting with |)
  const tblLines = backendTable.split(/\r?\n/).filter(l=>/^\|/.test(l));
  if(tblLines.length){
    lines.push('\n**Per-Backend Latency Budgets**');
    lines.push('');
    lines.push(tblLines.join('\n'));
  }
}

// Aggregate history
const aggHist=run('node dev/web_viewer/scripts/latency-history-table.js');
if(aggHist){
  lines.push('\n**Recent Aggregate History**');
  lines.push(aggHist);
}
// Backend history
const backendHist=run('node dev/web_viewer/scripts/backend-latency-history-table.js');
if(backendHist){
  lines.push('\n**Recent Backend History**');
  lines.push(backendHist);
}

// Audio scenarios table
const audioTable=run('node dev/web_viewer/scripts/summarize-audio-logs.js');
if(audioTable){
  lines.push('\n**Audio Scenario Metrics**');
  lines.push(audioTable);
}

// Guidance footer
lines.push('\n_Use env vars to tune budgets: DEFAULT_MAX_P50_MS, BACKEND_SPEECH_MAX_P95_MS, etc. Config file: dev/web_viewer/perf-backend-budgets.json_');

const out = lines.join('\n');
if (process.argv[2]) {
  fs.writeFileSync(process.argv[2], out);
  console.log('Wrote performance comment to', process.argv[2]);
} else {
  console.log(out);
}
