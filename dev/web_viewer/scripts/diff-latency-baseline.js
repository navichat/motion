#!/usr/bin/env node
/**
 * Outputs a simple delta report between current test-results/perf-latency.json
 * and dev/web_viewer/perf-baseline.json (or env LATENCY_BASELINE_JSON).
 * Non-fatal; intended for observability in CI summaries.
 */
const fs = require('fs');
const path = require('path');

function loadJSON(p){ try { return JSON.parse(fs.readFileSync(p,'utf8')); } catch { return null; } }
const currPath = path.join(process.cwd(),'test-results','perf-latency.json');
const basePath = process.env.LATENCY_BASELINE_JSON || path.join(process.cwd(),'dev','web_viewer','perf-baseline.json');
const curr = loadJSON(currPath);
const base = loadJSON(basePath);
if(!curr){ console.log('No current perf-latency.json found (run latency test first).'); process.exit(0);} 
if(!base){ console.log('No baseline file found; nothing to diff.'); process.exit(0);} 

const metrics = ['p50','avgLatencyMs','p90','p95','p99','lastLatencyMs'];
function fmt(v){ return (v==null)?'–':(Math.round(v*10)/10)+'ms'; }
function classify(m,a,b){
	if(a==null||b==null) return '';
	const delta = a-b;
	if(delta <= 0) return '✅';
	const rel = b===0? Infinity : delta/b;
	// Metric-specific soft thresholds
	const soft = { p50:0.25, avgLatencyMs:0.30, p90:0.35, p95:0.40, p99:0.50, lastLatencyMs:0.50 }[m] || 0.35;
	const hard = soft * 2;
	if(rel > hard) return '🛑';
	if(rel > soft) return '⚠️';
	return '⬆️';
}
function diff(m,a,b){ if(a==null||b==null) return '–'; const d=a-b; const pct = b===0? '∞' : ((d/b)*100).toFixed(1)+'%'; return `${classify(m,a,b)} ${d>=0?'+':''}${Math.round(d*10)/10}ms (${pct})`; }

let lines=['Latency Baseline Diff','Metric | Baseline | Current | Δ (abs, rel)','------ | -------- | ------- | -------------'];
for(const m of metrics){ const cVal = curr.metrics?.[m] ?? curr.metrics?.[m.replace('avgLatencyMs','avgLatencyMs')]; const bVal = base.metrics?.[m]; lines.push(`${m} | ${fmt(bVal)} | ${fmt(cVal)} | ${diff(m,cVal,bVal)}`); }
console.log(lines.join('\n'));
