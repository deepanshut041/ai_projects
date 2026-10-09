import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { normalizeBase, readRecords, validateCatalog, renderer, sourceURL } from '../scripts/content.mjs';
import { workspaceRecords } from '../scripts/workspace.mjs';
import { artifact, tutorialRecords } from '../scripts/tutorials.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
test('tutorial exports preserve prefixes and reject changed or escaping artifacts', async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'ai-tutorials-'));
  try {
    await fs.cp(path.join(root, 'tutorials'), path.join(temp, 'tutorials'), { recursive: true });
    const tutorial = await tutorialRecords(temp);
    assert.equal(tutorial.index.projects.length, 20);
    assert.equal(tutorial.records.length, 1);
    const records = await readRecords(path.join(temp, 'content'), '/nested/', tutorial.records);
    assert.match(records[0].html, /href="\/nested\/tutorial-assets\/p01/);
    assert.match(records[0].html, /class="practice-reveal"/);
    assert.match(records[0].html, /data-variable="mask"/);
    const expectedDiagrams = [];
    for (const project of tutorial.index.projects) {
      for (const relative of project.lessons) {
        const lesson = JSON.parse(await fs.readFile(path.join(temp, 'tutorials', relative, 'lesson.json'), 'utf8'));
        expectedDiagrams.push(...lesson.diagrams.map(diagram =>
          `tutorial-assets/${project.id.toLowerCase()}/${lesson.slug}/${lesson.revision}/diagrams/${diagram.id}.svg`));
      }
    }
    assert.deepEqual(tutorial.assets.filter(asset => asset.file.endsWith('.svg')).map(asset => asset.file).sort(), expectedDiagrams.sort());
    await assert.rejects(artifact(temp, '../README.md'), /Unsafe/);
    const chapter = path.join(temp, 'tutorials/projects/p01/gpt-1-foundations/v1/chapters/orientation.md');
    await fs.appendFile(chapter, '\nUnvalidated alteration.');
    await assert.rejects(tutorialRecords(temp), /changed after rendering/);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
test('canonical checkpoints render from owner sources with portable links and current dates', async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'ai-canonical-'));
  const site = path.join(temp, 'ai_projects/docs');
  await fs.mkdir(path.join(site, 'content'), { recursive: true });
  try {
    assert.deepEqual(await workspaceRecords(site, 'https://example.com/research/'), []);
    for (const name of ['roadmap', 'progress', 'resume', 'decisions']) {
      await fs.writeFile(path.join(site, `${name}.md`), `# ${name}\nUpdated: 2026-11-02\n[Resume](resume.md)\n[Research](../../ai_research/roadmap.md)`);
    }
    const extra = await workspaceRecords(site, 'https://example.com/research/');
    const records = await readRecords(path.join(site, 'content'), '/custom/', extra);
    assert.equal(records.length, 4);
    assert.ok(records.every(record => record.updated === '2026-11-02'));
    assert.match(records[0].html, /href="\/custom\/records\/guides\/resume\/"/);
    assert.match(records[0].html, /href="https:\/\/example.com\/research\/"/);
    await fs.writeFile(path.join(site, 'resume.md'), '# resume\nUpdated: 2026-11-03\nCurrent: P02');
    const changed = await readRecords(path.join(site, 'content'), '/', await workspaceRecords(site, 'https://example.com/research/'));
    assert.ok(changed.some(record => record.id === 'guides/resume' && record.updated === '2026-11-03' && record.html.includes('P02')));
    await fs.unlink(path.join(site, 'progress.md'));
    await assert.rejects(workspaceRecords(site, 'https://example.com/research/'), /Missing canonical/);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
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
