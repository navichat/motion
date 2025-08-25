#!/usr/bin/env node
/**
 * Post performance summary to a Slack Incoming Webhook.
 * Requires env SLACK_WEBHOOK_URL.
 * Optional env: SLACK_CHANNEL (override default channel if webhook supports), MAX_TABLE_LINES (default 30).
 * Uses existing artifacts; will generate if missing (soft mode) where feasible:
 *  - perf-webhook-payload.json (creates if missing)
 *  - PR markdown comment (for richer text fallback)
 */
const https=require('https');
const url=require('url');
const fs=require('fs');
const path=require('path');
const { execSync } = require('child_process');

function run(cmd){try{return execSync(cmd,{stdio:['ignore','pipe','ignore']}).toString();}catch{return ''}}
function loadJSON(p){try{return JSON.parse(fs.readFileSync(p,'utf8'));}catch{return null;}}

const hook=process.env.SLACK_WEBHOOK_URL; if(!hook){console.error('SLACK_WEBHOOK_URL not set; skipping'); process.exit(0);} 
const outDir=path.join(process.cwd(),'test-results');
if(!fs.existsSync(outDir)) fs.mkdirSync(outDir,{recursive:true});
const payloadPath=path.join(outDir,'perf-webhook-payload.json');
if(!fs.existsSync(payloadPath)){
  run('node dev/web_viewer/scripts/generate-perf-webhook-payload.js');
}
const payload=loadJSON(payloadPath) || {};
const prComment=run('node dev/web_viewer/scripts/generate-perf-pr-comment.js');

const maxLines= parseInt(process.env.MAX_TABLE_LINES||'30',10);
function trimLines(text){const lines=text.split(/\r?\n/); if(lines.length<=maxLines)return text; return lines.slice(0,maxLines).join('\n')+'\n…(truncated)…';}

const statusEmoji = payload.summary?.ok ? '✅' : '🛑';
const failing = (payload.summary?.failingBackends||[]).join(', ') || 'none';
const agg = payload.aggregate || {};
const title = `${statusEmoji} Perf ${agg.p50}ms p50 / ${agg.p95}ms p95 (samples ${agg.samples}) backendsFail=${failing}`;

let mdSection = prComment ? trimLines(prComment) : '';
if(!mdSection){
  mdSection = 'No detailed comment generated.';
}

// Slack payload (mrkdwn)
const slackBody = {
  text: title,
  blocks: [
    { type: 'section', text: { type:'mrkdwn', text: '*'+title+'*' } },
    { type: 'section', text: { type:'mrkdwn', text: '```'+mdSection+'```' } }
  ]
};

const hookUrl = url.parse(hook);
const req = https.request({ method:'POST', hostname:hookUrl.hostname, path:hookUrl.path, protocol:hookUrl.protocol, headers:{'Content-Type':'application/json'}}, res => {
  let data=''; res.on('data',d=>data+=d); res.on('end',()=>{console.log('Slack webhook response', res.statusCode, data);});
});
req.on('error', e=>{ console.error('Slack webhook error', e); });
req.write(JSON.stringify(slackBody));
req.end();
