#!/usr/bin/env node
/**
 * Appends a single-line CSV row of latency metrics for trend analysis.
 * File: dev/web_viewer/perf-latency-history.csv
 * Columns: isoTimestamp,gitSha,samples,avg,p50,p90,p95,p99,last
 */
const fs=require('fs');
const path=require('path');
function load(p){try{return JSON.parse(fs.readFileSync(p,'utf8'));}catch{return null;}}
const currPath=path.join(process.cwd(),'test-results','perf-latency.json');
const data=load(currPath); if(!data){console.error('No perf-latency.json, skipping history append'); process.exit(0);} 
const m=data.metrics||{}; const sha=process.env.GITHUB_SHA||'local';
const row=[new Date().toISOString(),sha,m.samples||0,m.avgLatencyMs||0,m.p50||0,m.p90||0,m.p95||0,m.p99||0,m.lastLatencyMs||0].join(',');
const historyPath=path.join(process.cwd(),'dev','web_viewer','perf-latency-history.csv');
if(!fs.existsSync(historyPath)){
  fs.writeFileSync(historyPath,'isoTimestamp,gitSha,samples,avg,p50,p90,p95,p99,last\n');
}
fs.appendFileSync(historyPath,row+'\n');
console.log('Appended latency history row:', row);
