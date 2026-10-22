import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const catalogPath = path.join(root, 'catalog/sources.json');
const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
let files = 0, bytes = 0;
async function walk(folder) {
  for (const entry of await fs.readdir(folder, { withFileTypes: true }).catch(error => { if (error.code === 'ENOENT') return []; throw error; })) {
    if (entry.isSymbolicLink()) continue;
    const file = path.join(folder, entry.name);
    if (entry.isDirectory()) await walk(file);
    else if (entry.isFile() && entry.name !== 'metadata.json') { files++; bytes += (await fs.stat(file)).size; }
  }
}
await walk(path.resolve(root, '../sources'));
catalog.archive = { files, bytes };
await fs.writeFile(catalogPath + '.tmp', JSON.stringify(catalog, null, 2) + '\n');
await fs.rename(catalogPath + '.tmp', catalogPath);
console.log(`Recorded ${files} raw archived files (${bytes} bytes); original bytes were not changed.`);
