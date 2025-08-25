#!/usr/bin/env node
/*
 Aggregate audio log artifacts (test-results/audio-log-*.json) into a markdown table.
 Columns: Scenario | Events | UniqueTypes | Samples | Avg(ms) | P50 | P95 | P99 | Lat?
 Prints nothing if no artifacts present (non-fatal).
*/
const fs = require('fs');
const path = require('path');

function fmt(n){
  if (n === null || n === undefined) return '';
  if (typeof n !== 'number' || isNaN(n)) return '';
  return n.toFixed(1);
}

function run(){
  const dir = path.join(process.cwd(), 'test-results');
  if (!fs.existsSync(dir)) return;
  const files = fs.readdirSync(dir).filter(f => /^audio-log-.*\.json$/.test(f));
  if (!files.length) return;
  const rows = [];
  for (const f of files){
    try {
      const j = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
      const m = j.metrics || {};
      rows.push({
        scenario: j.scenario || f.replace(/^audio-log-|\.json$/g,''),
        events: j.count || 0,
        uniq: (j.uniqueTypes||[]).length,
        samples: m.samples || 0,
        avg: m.avgLatencyMs,
        p50: m.p50,
        p95: m.p95,
        p99: m.p99,
        lat: j.latencyEntry ? '✅' : '❌'
      });
    } catch {/* ignore broken file */}
  }
  if (!rows.length) return;
  rows.sort((a,b)=> a.scenario.localeCompare(b.scenario));
  const header = '| Scenario | Events | Types | Samples | Avg (ms) | P50 | P95 | P99 | Lat? |';
  const sep = '|---|---:|---:|---:|---:|---:|---:|---:|:---:|';
  const lines = rows.map(r => `| ${r.scenario} | ${r.events} | ${r.uniq} | ${r.samples} | ${fmt(r.avg)} | ${fmt(r.p50)} | ${fmt(r.p95)} | ${fmt(r.p99)} | ${r.lat} |`);
  console.log([header, sep, ...lines].join('\n'));
}

run();
