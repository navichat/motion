#!/usr/bin/env node
/**
 * Outputs a Markdown table for the last N (default 10) latency history rows.
 * Source CSV: dev/web_viewer/perf-latency-history.csv
 */
const fs=require('fs');
const path=require('path');
const max=parseInt(process.env.LATENCY_HISTORY_ROWS||'10',10);
const csvPath=path.join(process.cwd(),'dev','web_viewer','perf-latency-history.csv');
if(!fs.existsSync(csvPath)){ console.log('Latency history not available yet.'); process.exit(0); }
const lines=fs.readFileSync(csvPath,'utf8').trim().split(/\r?\n/);
if(lines.length<=1){ console.log('Latency history empty.'); process.exit(0); }
const header=lines[0].split(',');
const rows=lines.slice(1).slice(-max).map(l=>l.split(','));
function ms(v){ const n=Number(v); if(isNaN(n)) return v; return n.toFixed(1); }
let out=['| # | Timestamp | SHA | samples | avg | p50 | p90 | p95 | p99 | last |','|---|-----------|-----|---------|-----|-----|-----|-----|-----|------|'];
rows.forEach((r,i)=>{ const idx=rows.length-max>0?i+1:i+1; out.push(`| ${i+1} | ${r[0]} | ${r[1].slice(0,7)} | ${r[2]} | ${ms(r[3])} | ${ms(r[4])} | ${ms(r[5])} | ${ms(r[6])} | ${ms(r[7])} | ${ms(r[8])} |`); });
console.log(out.join('\n'));
