#!/usr/bin/env node
/**
 * Print recent per-backend latency history as a markdown table.
 * Reads: dev/web_viewer/perf-backend-latency-history.csv
 * Env: BACKEND_HISTORY_ROWS (default 10)
 */
const fs = require('fs');
const path = require('path');
const rowsEnv = parseInt(process.env.BACKEND_HISTORY_ROWS||'10',10);
const p = path.join(process.cwd(),'dev','web_viewer','perf-backend-latency-history.csv');
if(!fs.existsSync(p)) { process.exit(0); }
const lines = fs.readFileSync(p,'utf8').trim().split(/\r?\n/);
if(lines.length<=1){ process.exit(0); }
const header = lines[0].split(',');
const data = lines.slice(1);
const tail = data.slice(-rowsEnv);
function mdRow(cols){ return '| '+cols.join(' | ')+' |'; }
const table = [mdRow(header), mdRow(header.map(()=> '---'))];
for(const line of tail){ table.push(mdRow(line.split(','))); }
console.log(table.join('\n'));
