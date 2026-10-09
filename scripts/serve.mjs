import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../dist', import.meta.url));
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.md': 'text/markdown; charset=utf-8', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf' };
export async function serve(port = 4273, { liveReload = false } = {}) {
  const { base } = JSON.parse(await fs.readFile(path.join(root, 'build.json'), 'utf8'));
  const clients = new Set();
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      if (liveReload && url.pathname === `${base}__dev/events`) {
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' }); res.write(': connected\n\n'); clients.add(res); res.on('close', () => clients.delete(res)); return;
      }
      if (url.pathname === '/' && base !== '/' || url.pathname === base.slice(0, -1) && base !== '/') { res.writeHead(302, { Location: base }); res.end(); return; }
      if (!url.pathname.startsWith(base)) throw new Error('Not found');
      let target = path.resolve(root, decodeURIComponent(url.pathname.slice(base.length)));
      if (target !== root && !target.startsWith(root + path.sep)) throw new Error('Not found');
      if ((await fs.stat(target)).isDirectory()) {
        if (!url.pathname.endsWith('/')) { res.writeHead(302, { Location: url.pathname + '/' }); res.end(); return; }
        target = path.join(target, 'index.html');
      }
      let data = await fs.readFile(target);
      if (liveReload && path.extname(target) === '.html') data = Buffer.from(data.toString().replace('</body>', `<script>const events=new EventSource(${JSON.stringify(base + '__dev/events')});events.addEventListener('reload',()=>location.reload());</script></body>`));
      res.writeHead(200, { 'Content-Type': mime[path.extname(target)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' }); res.end(data);
    } catch { res.writeHead(404, { 'Content-Type': 'text/html' }); res.end(await fs.readFile(path.join(root, '404.html')).catch(() => 'Not found')); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
  return { server, origin: `http://127.0.0.1:${server.address().port}`, base, reload: () => { for (const client of clients) client.write('event: reload\ndata: updated\n\n'); } };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { origin, base } = await serve(Number(process.env.PORT ?? 4273));
  console.log(`AI Projects: ${origin}${base}`);
}
