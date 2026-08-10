/**
 * Dev server for Segment Anything LiteRT.js demo.
 *
 * Usage: node server.js [port]
 *
 * Serves static files with COOP/COEP headers required for
 * SharedArrayBuffer (needed by LiteRT.js JSPI + WebNN).
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.argv[2] || '8080', 10);
const ROOT = __dirname;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.mjs':  'application/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif':  'image/gif',
  '.tflite': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.zip':  'application/zip',
};

const server = http.createServer((req, res) => {
  // Parse URL
  let urlPath = req.url.split('?')[0].split('#')[0];

  // Security: prevent directory traversal
  if (urlPath.includes('..')) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  // Default to index.html
  if (urlPath === '/') urlPath = '/index.html';

  const filePath = path.join(ROOT, urlPath);

  // Serve file
  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const mimeType = MIME_TYPES[ext] || 'application/octet-stream';

    // Required headers for SharedArrayBuffer / JSPI
    // credentialless: allows cross-origin CDN resources without explicit CORS
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');

    // Allow jsDelivr CDN for LiteRT.js
    res.setHeader('Content-Type', mimeType);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Accept-Ranges', 'bytes');

    // Cache .tflite files aggressively
    if (ext === '.tflite') {
      res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    }

    // Handle Range requests (for large model files)
    if (req.headers.range && ext === '.tflite') {
      const range = req.headers.range;
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stats.size - 1;
      const chunkSize = end - start + 1;

      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${stats.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': mimeType,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'cross-origin',
      });

      const stream = fs.createReadStream(filePath, { start, end });
      stream.pipe(res);
      return;
    }

    res.setHeader('Content-Length', stats.size);
    res.writeHead(200);

    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
  });
});

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════╗
║   Segment Anything — LiteRT.js + WebNN Demo          ║
║                                                      ║
║   Open in browser:                                   ║
║   http://localhost:${PORT}                              ║
║                                                      ║
║   Models:                                            ║
║   - encoder.tflite (26.6 MB)                         ║
║   - decoder.tflite (23.7 MB)                         ║
║                                                      ║
║   Requirements:                                      ║
║   - Chrome 121+ or Edge 121+                         ║
║   - chrome://flags/#web-machine-learning-            ║
║     neural-network → Enabled                         ║
║   - chrome://flags/#enable-experimental-             ║
║     webassembly-features → Enabled (JSPI)            ║
║                                                      ║
║   Press Ctrl+C to stop                               ║
╚══════════════════════════════════════════════════════╝
`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is in use. Try: node server.js ${PORT + 1}`);
  } else {
    console.error('Server error:', err);
  }
  process.exit(1);
});
