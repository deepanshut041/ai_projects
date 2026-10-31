import { watch } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { build, root, docsRoot } from './build.mjs';
import { serve } from './serve.mjs';
await fs.mkdir(path.resolve(docsRoot, '../../.ai/video-library'), { recursive: true });
let built = await build({ local: true });
let timer, pending = Promise.resolve();
const { origin, base, reload } = await serve(Number(process.env.PORT ?? 4273), { liveReload: true, local: true, directory: built.out, ready: () => pending });
console.log(`AI Projects: ${origin}${base}`);
const inputs = [
  ...['content', 'catalog', 'src', 'public'].map(folder => path.join(root, folder)),
  ...['tutorials', 'guide', 'implementation'].map(folder => path.join(docsRoot, folder)),
];
const watchers = new Map();
function refreshWatches() {
  const wanted = new Map([...inputs.map(folder => ({ folder, recursive: true })), ...built.library.watches].map(item => [item.folder, item]));
  for (const [folder, watcher] of watchers) if (!wanted.has(folder)) { watcher.close(); watchers.delete(folder); }
  for (const { folder, recursive } of wanted.values()) if (!watchers.has(folder)) {
    try { watchers.set(folder, watch(folder, { recursive }, schedule)); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
}
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(() => { pending = pending.then(async () => { built = await build({ local: true }); refreshWatches(); reload(); }).catch(error => console.error(error.message)); }, 200);
}
refreshWatches();
