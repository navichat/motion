#!/usr/bin/env node
/**
 * Simple latency regression checker.
 * Compares current perf-latency.json (from test run) with baseline file.
 * Baseline search order:
 *  1. env LATENCY_BASELINE_JSON path if provided
 *  2. repo file at dev/web_viewer/perf-baseline.json
 * If no baseline found, exits 0 after copying current metrics as new baseline (unless DISALLOW_BASELINE_WRITE=1).
 * Regression rules (can override via env):
 *  MAX_ABS_INCREASE_P50_MS (default 400)
 *  MAX_ABS_INCREASE_P95_MS (default 800)
 *  MAX_ABS_INCREASE_P99_MS (default 1200)
 *  MAX_REL_INCREASE_P95 (default 2.5 => 250%)
 */
const fs = require('fs');
const path = require('path');

const CURR_PATH = path.join(process.cwd(),'test-results','perf-latency.json');
if (!fs.existsSync(CURR_PATH)) {
  console.error('Current perf-latency.json not found. Ensure latency test ran first.');
  process.exit(2);
}
const curr = JSON.parse(fs.readFileSync(CURR_PATH,'utf8')).metrics || {};

const explicitBaseline = process.env.LATENCY_BASELINE_JSON;
let baselinePath = explicitBaseline || path.join(process.cwd(),'dev','web_viewer','perf-baseline.json');
let baseline = null;
if (fs.existsSync(baselinePath)) {
  try { baseline = JSON.parse(fs.readFileSync(baselinePath,'utf8')).metrics || {}; } catch (e) { console.warn('Failed to parse baseline JSON', e); }
}

if (!baseline) {
  console.log('No baseline present; establishing new baseline at', baselinePath);
  if (process.env.DISALLOW_BASELINE_WRITE === '1') {
    console.log('Baseline write disallowed; skipping without failure.');
    process.exit(0);
  }
  try {
    fs.writeFileSync(baselinePath, JSON.stringify({ created: new Date().toISOString(), metrics: curr }, null, 2));
  } catch (e) { console.warn('Failed to write new baseline', e); }
  process.exit(0);
}

function delta(a,b){ return a-b; }
function rel(a,b){ return b===0? (a>0?Infinity:0) : a/b; }

const limits = {
  MAX_ABS_INCREASE_P50_MS: parseFloat(process.env.MAX_ABS_INCREASE_P50_MS || '400'),
  MAX_ABS_INCREASE_P95_MS: parseFloat(process.env.MAX_ABS_INCREASE_P95_MS || '800'),
  MAX_ABS_INCREASE_P99_MS: parseFloat(process.env.MAX_ABS_INCREASE_P99_MS || '1200'),
  MAX_REL_INCREASE_P95: parseFloat(process.env.MAX_REL_INCREASE_P95 || '2.5'),
};

const issues = [];

if (delta(curr.p50, baseline.p50) > limits.MAX_ABS_INCREASE_P50_MS) {
  issues.push(`p50 increase ${delta(curr.p50, baseline.p50).toFixed(1)}ms > ${limits.MAX_ABS_INCREASE_P50_MS}ms`);
}
if (delta(curr.p95, baseline.p95) > limits.MAX_ABS_INCREASE_P95_MS) {
  issues.push(`p95 increase ${delta(curr.p95, baseline.p95).toFixed(1)}ms > ${limits.MAX_ABS_INCREASE_P95_MS}ms`);
}
if (delta(curr.p99, baseline.p99) > limits.MAX_ABS_INCREASE_P99_MS) {
  issues.push(`p99 increase ${delta(curr.p99, baseline.p99).toFixed(1)}ms > ${limits.MAX_ABS_INCREASE_P99_MS}ms`);
}
if (rel(curr.p95, baseline.p95) > limits.MAX_REL_INCREASE_P95) {
  issues.push(`p95 relative increase ${(rel(curr.p95, baseline.p95)*100).toFixed(1)}% > ${(limits.MAX_REL_INCREASE_P95*100).toFixed(0)}%`);
}

if (issues.length) {
  console.error('Latency regression detected:\n - ' + issues.join('\n - '));
  console.error('Baseline:', baseline); 
  console.error('Current :', curr);
  process.exit(1);
}

console.log('Latency regression check passed.');
console.log('Baseline:', baseline); 
console.log('Current :', curr);
process.exit(0);
