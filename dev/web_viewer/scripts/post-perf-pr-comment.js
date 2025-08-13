#!/usr/bin/env node
/**
 * Post (or upsert) performance summary as a PR comment.
 * Uses GITHUB_EVENT_PATH to get PR number; requires a token (GITHUB_TOKEN or GH_TOKEN) with repo scope.
 * Idempotent: updates existing comment containing marker.
 */
const fs=require('fs');
const { execSync } = require('child_process');
const path=require('path');

const marker='### 🤖 Performance Summary';

function run(cmd){try{return execSync(cmd,{stdio:['ignore','pipe','ignore']}).toString();}catch{return ''}}
function genComment(){ return run('node dev/web_viewer/scripts/generate-perf-pr-comment.js'); }

const repo=process.env.GITHUB_REPOSITORY; // owner/repo
const eventPath=process.env.GITHUB_EVENT_PATH;
const token=process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
if(!repo || !eventPath || !token){
  console.error('Missing repo/event/token; skipping PR comment.');
  process.exit(0);
}
let prNumber=null;
try{
  const ev=JSON.parse(fs.readFileSync(eventPath,'utf8'));
  prNumber=ev.pull_request?.number || ev.number || null;
}catch(e){ console.error('Failed to parse event', e.message); }
if(!prNumber){
  console.log('Not a pull_request event; skipping.');
  process.exit(0);
}
const [owner,repoName]=repo.split('/');

// Fetch existing comments
let existing=[];
try{
  const raw=run(`curl -s -H 'Authorization: Bearer ${token}' -H 'Accept: application/vnd.github+json' https://api.github.com/repos/${owner}/${repoName}/issues/${prNumber}/comments?per_page=100`);
  existing=JSON.parse(raw);
}catch{}
const match=existing.find(c=>typeof c.body==='string' && c.body.includes(marker));
const body=genComment() || marker+'\n(No data)';
if(match){
  // Update
  try{
    run(`curl -s -X PATCH -H 'Authorization: Bearer ${token}' -H 'Accept: application/vnd.github+json' https://api.github.com/repos/${owner}/${repoName}/issues/comments/${match.id} -d ${JSON.stringify(JSON.stringify({body}))}`);
    console.log('Updated existing perf comment id', match.id);
  }catch(e){ console.error('Failed to update comment', e.message); }
}else{
  try{
    run(`curl -s -X POST -H 'Authorization: Bearer ${token}' -H 'Accept: application/vnd.github+json' https://api.github.com/repos/${owner}/${repoName}/issues/${prNumber}/comments -d ${JSON.stringify(JSON.stringify({body}))}`);
    console.log('Created new perf comment');
  }catch(e){ console.error('Failed to create comment', e.message); }
}
