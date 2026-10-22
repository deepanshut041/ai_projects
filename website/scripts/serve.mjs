import http from 'node:http';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../dist', import.meta.url));
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.md': 'text/markdown; charset=utf-8', '.txt':'text/plain; charset=utf-8', '.cpp':'text/plain; charset=utf-8', '.hpp':'text/plain; charset=utf-8', '.py':'text/plain; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.vtt': 'text/vtt', '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf' };
export async function serve(port = 4273, { liveReload = false, directory = root, local = false, ready = async () => {} } = {}) {
  const { base } = JSON.parse(await fs.readFile(path.join(directory, 'build.json'), 'utf8'));
  const clients = new Set();
  const server = http.createServer(async (req, res) => {
    try {
      if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405, { Allow: 'GET, HEAD' }); res.end(); return; }
      await ready();
      const url = new URL(req.url, 'http://localhost');
      if (liveReload && url.pathname === `${base}__dev/events`) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' }); res.write(': connected\n\n'); clients.add(res); res.on('close', () => clients.delete(res)); return;
      }
      if (url.pathname === '/' && base !== '/' || url.pathname === base.slice(0, -1) && base !== '/') { res.writeHead(302, { Location: base }); res.end(); return; }
      if (!url.pathname.startsWith(base)) throw new Error('Not found');
      const relative = url.pathname.slice(base.length);
      if (decodeURIComponent(relative) === 'local-files.json') throw new Error('Not found');
      if (relative.startsWith('local-files/')) {
        if (!local) throw new Error('Not found');
        const files = JSON.parse(await fs.readFile(path.join(directory, 'local-files.json'), 'utf8'));
        const target = files[relative.slice('local-files/'.length)];
        if (!target || await fs.realpath(target) !== target) throw new Error('Not found');
        const stat = await fs.stat(target);
        const types = { '.mp4': 'video/mp4', '.m4a': 'audio/mp4', '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.srt': 'text/plain; charset=utf-8', '.vtt': 'text/vtt' };
        const headers = { 'Content-Type': types[path.extname(target)] ?? mime[path.extname(target)] ?? 'text/plain; charset=utf-8', 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache' };
        let start = 0, end = stat.size - 1;
        if (req.headers.range) {
          const range = req.headers.range.match(/^bytes=(\d*)-(\d*)$/);
          if (range && (range[1] || range[2])) {
            start = range[1] ? Number(range[1]) : Math.max(0, stat.size - Number(range[2]));
            end = range[1] && range[2] ? Math.min(Number(range[2]), end) : end;
          }
          if (!range || !(range[1] || range[2]) || start > end || start >= stat.size) { res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` }); res.end(); return; }
          headers['Content-Range'] = `bytes ${start}-${end}/${stat.size}`;
        }
        headers['Content-Length'] = Math.max(0, end - start + 1);
        res.writeHead(req.headers.range ? 206 : 200, headers);
        if (req.method === 'HEAD' || stat.size === 0) res.end();
        else { const stream = createReadStream(target, { start, end }); stream.on('error', () => res.destroy()); res.on('close', () => stream.destroy()); stream.pipe(res); }
        return;
      }
      let target = path.resolve(directory, decodeURIComponent(relative));
      if (target !== directory && !target.startsWith(directory + path.sep)) throw new Error('Not found');
      if ((await fs.stat(target)).isDirectory()) {
        if (!url.pathname.endsWith('/')) { res.writeHead(302, { Location: url.pathname + '/' }); res.end(); return; }
        target = path.join(target, 'index.html');
      }
      let data = await fs.readFile(target);
      if (liveReload && path.extname(target) === '.html') data = Buffer.from(data.toString().replace('</body>', `<script>const events=new EventSource(${JSON.stringify(base + '__dev/events')});events.addEventListener('reload',()=>location.reload());</script></body>`));
      res.writeHead(200, { 'Content-Type': mime[path.extname(target)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' }); res.end(data);
    } catch { res.writeHead(404, { 'Content-Type': 'text/html' }); res.end(await fs.readFile(path.join(directory, '404.html')).catch(() => 'Not found')); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
  return { server, origin: `http://127.0.0.1:${server.address().port}`, base, reload: () => { for (const client of clients) client.write('event: reload\ndata: updated\n\n'); } };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { origin, base } = await serve(Number(process.env.PORT ?? 4273));
  console.log(`AI Projects: ${origin}${base}`);
}
