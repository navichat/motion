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
  // Compute utilization summary if JSON export exists or can be produced
  const jsonPath = path.join('test-results','per-backend-latency.json');
  if(!fs.existsSync(jsonPath)){
    run('BACKEND_BUDGET_SOFT=1 PER_BACKEND_LATENCY_JSON=test-results/per-backend-latency.json node dev/web_viewer/scripts/verify-audio-backend-latency.js');
  }
  const backendJson = safeReadJSON(jsonPath);
  if(backendJson && Array.isArray(backendJson.rows)){
    const utilHeader='| Backend | Samples | p50 vs Budget | p90 vs Budget | p95 vs Budget | Status |';
    const utilSep='|---|---:|---|---|---|---|';
    const utilLines=[utilHeader,utilSep];
    backendJson.rows.forEach(r=>{
      function fmt(val){return val==null?'':Number(val).toFixed(1);}    
      function cell(metric){
        const v=r[metric]; const b=r[metric+'Budget']; if(v==null||b==null) return '';
        const ratio = b? (v/b):0; const pct=(ratio*100).toFixed(1)+'%';
        return `${fmt(v)} / ${b} (${pct})`;
      }
      // Status emoji heuristics based on highest percentile utilization
      const ratios=[]; ['p50','p90','p95'].forEach(k=>{if(r[k]!=null && r[k+'Budget']!=null) ratios.push(r[k]/r[k+'Budget']);});
      const maxRatio = ratios.length? Math.max(...ratios):0;
      let status='✅';
      if(maxRatio>0.95) status='🛑'; else if(maxRatio>0.85) status='⚠️'; else if(maxRatio>0.70) status='⬆️';
      const row = `| ${r.backend} | ${r.samples} | ${cell('p50')} | ${cell('p90')} | ${cell('p95')} | ${status} |`;
      utilLines.push(row);
    });
    lines.push('\n**Backend Budget Utilization**');
    lines.push(utilLines.join('\n'));
    lines.push('\n_Status legend: ✅ <=70% | ⬆️ 70-85% | ⚠️ 85-95% | 🛑 >95% of budget (any percentile)_');
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
