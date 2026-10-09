import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as bundle } from 'esbuild';
import config from '../site.config.mjs';
import { readRecords, normalizeBase, validateCatalog, escape as e, categories } from './content.mjs';
import { shell, home, recordPage, recordCard, sourcePage, heading, guide } from './templates.mjs';
import { workspaceRecords } from './workspace.mjs';
import { tutorialRecords } from './tutorials.mjs';

export const root = fileURLToPath(new URL('../', import.meta.url));
export async function build() {
  const base = normalizeBase(config.base);
  const tutorials=await tutorialRecords(root);
  const records = await readRecords(path.join(root, 'content'), base, [...await workspaceRecords(root, config.companionUrl),...tutorials.records]);
  const catalog = validateCatalog(JSON.parse(await fs.readFile(path.join(root, 'catalog/sources.json'), 'utf8')));
  const out = path.join(root, 'dist');
  await fs.rm(out, { recursive: true, force: true });
  await fs.mkdir(out, { recursive: true });
  const write = async (file, body) => { const dest = path.join(out, file); await fs.mkdir(path.dirname(dest), { recursive: true }); await fs.writeFile(dest, body); };
  await fs.cp(path.join(root, 'public'), out, { recursive: true });
  for(const asset of tutorials.assets) await write(asset.file,asset.bytes);
  if(tutorials.index) await write('tutorials/index.json',JSON.stringify(tutorials.index));
  await bundle({ entryPoints: [path.join(root, 'src/app.js')], outdir: path.join(out, 'assets'), bundle: true, format: 'esm', minify: true, loader: { '.woff': 'file', '.woff2': 'file', '.ttf': 'file' }, logLevel: 'warning' });
  const wrap = (title, body, active = '') => shell({ config, records, title, body, active });
  await write('index.html', wrap('Projects workbench', home(records, catalog, base)));
  for (const category of categories) {
    const chosen = records.filter(record => record.category === category.id);
    await write(`${category.id}/index.html`, wrap(category.title, `${heading(category.title, category.description)}<div class="record-grid">${chosen.map(recordCard).join('')}</div>${chosen.length ? '' : `<div class="empty"><span class="empty-symbol">↗</span><h2>A place for the next ${category.id === 'papers' ? 'paper review' : category.id === 'experiments' ? 'experiment' : 'question'}</h2><p>No ${e(category.title.toLowerCase())} have been added yet.</p><a class="button secondary" href="${base}guide/">Read the authoring guide</a></div>`}`, category.id));
  }
  for (const record of records) {
    await write(`records/${record.id}/index.html`, wrap(record.title, recordPage(record, base), record.category));
    await write(`source/${record.file}`, record.raw);
  }
  await write('sources/index.html', wrap('Source register', sourcePage(catalog, base), 'sources'));
  await write('guide/index.html', wrap('Writing guide', guide(base), 'guide'));
  await write('404.html', wrap('Page not found', `${heading('This record is not here', 'Browse the Projects workbench to find a record or source.')}<a class="button" href="${base}">Open the workbench</a>`));
  await write('catalog/sources.json', JSON.stringify(catalog));
  await write('search.json', JSON.stringify(records.map(({ html, raw, ...record }) => record)));
  await write('build.json', JSON.stringify({ base, records: records.map(r => ({ id: r.id, url: r.url })), sources: catalog.sources.length }));
  await write('.nojekyll', '');
  console.log(`Built ${records.length} project records and ${catalog.sources.length} source references at ${base}.`);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await build();
