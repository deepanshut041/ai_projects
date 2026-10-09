import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { normalizeBase, readRecords, validateCatalog, renderer, sourceURL } from '../scripts/content.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
test('records and source catalog validate without opening the raw archive', async () => {
  const records = await readRecords(path.join(root, 'content'), '/project_docs/');
  assert.ok(records.length >= 4);
  assert.ok(records.every(r => r.url.startsWith('/project_docs/records/')));
  const catalog = validateCatalog(JSON.parse(await fs.readFile(path.join(root, 'catalog/sources.json'), 'utf8')));
  assert.ok(catalog.sources.length > 0);
  assert.equal(catalog.archive.files, 0);
  assert.ok(catalog.sources.every(s => s.access === 'link-checked' && s.review));
});
test('root and nested deployment paths are supported', () => {
  for (const base of ['/', '/project_docs/', '/quantum/research/']) assert.equal(normalizeBase(base), base);
  for (const base of ['../', '//', '/research', '/a/../']) assert.throws(() => normalizeBase(base));
});
test('record links validate and preserve deployment prefixes and anchors', () => {
  const { md } = renderer(new Set(['implementations/other.md']), 'guides/one.md', '/project_docs/');
  assert.match(md.render('[Paper](../implementations/other.md#claims)'), /href="\/project_docs\/records\/implementations\/other\/#claims"/);
  assert.throws(() => md.render('[Missing](missing.md)'), /Broken research link/);
  assert.match(md.render('[Sources](/sources/)'), /href="\/project_docs\/sources\/"/);
});
test('unsafe HTML stays inert and math renders', () => {
  const { md } = renderer(new Set(), 'papers/test.md', '/');
  assert.match(md.render('$x^2$'), /class="katex"/);
  assert.doesNotMatch(md.render('<script>alert(1)</script>'), /<script>/);
  assert.throws(() => sourceURL('javascript:alert(1)'));
});
test('duplicate catalog references and missing record metadata fail before publishing', async () => {
  const source = { id: 'c/a', collection: 'c', title: 'A', kind: 'Paper', url: 'https://example.com' };
  assert.throws(() => validateCatalog({ schema: 1, collections: [{ id: 'c' }], sources: [source, source] }), /Invalid source/);
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'quantum-research-'));
  try {
    await fs.mkdir(path.join(temp, 'implementations'));
    await fs.writeFile(path.join(temp, 'implementations/one.md'), '---\ntitle: A\ndescription: B\nupdated: "2026-10-02"\n---\nText');
    await assert.rejects(readRecords(temp, '/'), /missing status/);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
