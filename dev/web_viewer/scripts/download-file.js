#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const dir = path.dirname(dest);
    fs.mkdirSync(dir, { recursive: true });
    const out = fs.createWriteStream(dest);
    const mod = url.startsWith('https') ? https : http;
    const req = mod.get(url, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        // follow redirect
        return download(res.headers.location, dest).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      res.pipe(out);
      out.on('finish', () => out.close(() => resolve(dest)));
    });
    req.on('error', (err) => {
      fs.unlink(dest, () => reject(err));
    });
  });
}

if (require.main === module) {
  const [url, dest] = process.argv.slice(2);
  if (!url || !dest) {
    console.error('Usage: download-file <url> <dest>');
    process.exit(1);
  }
  download(url, dest)
    .then((p) => console.log('[download-file] saved', p))
    .catch((e) => { console.error('[download-file] failed', e.message); process.exit(2); });
}

module.exports = { download };
