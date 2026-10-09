import { watch } from 'node:fs';
import path from 'node:path';
import { build, root } from './build.mjs';
import { serve } from './serve.mjs';
await build();
const { origin, base, reload } = await serve(Number(process.env.PORT ?? 4273), { liveReload: true });
console.log(`AI Projects: ${origin}${base}`);
let timer, pending = Promise.resolve();
for (const folder of ['content', 'catalog', 'src', 'public', 'tutorials']) watch(path.join(root, folder), { recursive: true }, () => {
  clearTimeout(timer);
  timer = setTimeout(() => { pending = pending.then(async () => { await build(); reload(); }).catch(error => console.error(error.message)); }, 200);
});
// Docs-level canonical checkpoints are single-source inputs to the rendered site.
watch(root, (event, filename) => {
  if (!['START_HERE.md', 'roadmap.md', 'progress.md', 'resume.md', 'decisions.md', 'SETUP.md'].includes(String(filename))) return;
  clearTimeout(timer);
  timer = setTimeout(() => { pending = pending.then(async () => { await build(); reload(); }).catch(error => console.error(error.message)); }, 200);
});
