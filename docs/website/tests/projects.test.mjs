import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { projectLibrary, collectionCards, projectsSidebar } from '../scripts/projects.mjs';
import { serve } from '../scripts/serve.mjs';
import { renderer, videoDefinition } from '../scripts/content.mjs';

test('inline videos resolve every asset, escape titles and require local captioned media', () => {
  const spec = { title: '<script>Unsafe title</script>', src: '../clip/video.mp4', poster: '../clip/poster.jpg', captions: '../clip/captions.vtt', transcript: '../clip/transcript.md', duration: 178 };
  const { md } = renderer(new Set(), '', '/project_docs/', null, { resolveLink: file => '/project_docs/local-files/' + file.split('/').at(-1) });
  const html = md.render('```video\n' + JSON.stringify(spec) + '\n```');
  assert.match(html, /<video controls playsinline preload="none"/);
  assert.match(html, /&lt;script&gt;Unsafe title/);
  assert.doesNotMatch(html, /<script>|autoplay/);
  assert.match(html, /<track kind="captions" src="\/project_docs\/local-files\/captions.vtt"/);
  assert.match(html, /2:58/);
  for (const change of [{ src: 'https://remote/video.mp4' }, { captions: '' }, { duration: -1 }, { autoplay: true }]) assert.throws(() => videoDefinition(JSON.stringify({ ...spec, ...change })));
  const missing = renderer(new Set(), '', '/', null, { resolveLink: () => null });
  assert.throws(() => missing.md.render('```video\n' + JSON.stringify(spec) + '\n```'), /Missing or disallowed/);
});

test('project collections discover docs locally, preserve relative links and keep public content separate', async () => {
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'project-reader-'));
  try {
    const projects = path.join(temporary, 'ai_projects/projects');
    const project = path.join(projects, 'foundations/bike');
    const write = async (file, text) => { await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, text); };
    await write(path.join(projects, 'foundations/README.md'), '# Foundations\n\n1. [Bike](bike/README.md)\n');
    await write(path.join(project, 'README.md'), '# Bike\n\nHourly predictions. [Docs](docs/README.md)');
    await write(path.join(project, 'docs/README.md'), '# My docs\n\n[Notes](notes/understanding.md#example)');
    await write(path.join(project, 'docs/notes/understanding.md'), '# Understanding\n\nPRIVATE SENTINEL\n\n## Example\n\n[Home](../README.md) [Code][code] [Notebook](../../notebooks/example.ipynb) [Secret](../../.env) [Raw data](../../data/hour.csv) [Outside](../../../../../../outside.txt)\n\n[code]: ../../src/example.py\n\n$$x^2$$');
    await write(path.join(project, 'src/example.py'), 'print(42)\n');
    await write(path.join(project, 'notebooks/example.ipynb'), JSON.stringify({ cells: [], metadata: {}, nbformat: 4, nbformat_minor: 5 }));
    await write(path.join(project, '.env'), 'SECRET=private');
    await write(path.join(project, 'data/hour.csv'), 'raw data');
    await write(path.join(temporary, 'outside.txt'), 'outside');
    await fs.symlink(path.join(temporary, 'outside.txt'), path.join(project, 'docs/linked.md'));
    for (const base of ['/', '/project_docs/']) {
      const publicLibrary = await projectLibrary(projects, base);
      assert.equal(publicLibrary.pages.length, 2);
      assert.deepEqual(publicLibrary.files, {});
      assert.ok(publicLibrary.pages.every(p => !p.raw.includes('PRIVATE SENTINEL')));
      assert.doesNotMatch(publicLibrary.groups[0].projects[0].overview.html, /href="[^"]*docs/);
      const local = await projectLibrary(projects, base, { local: true });
      assert.equal(local.pages.length, 4, 'Symlink documents must not be traversed');
      const notes = local.pages.find(p => p.title === 'Understanding');
      assert.ok(notes.html.includes(`href="${base}projects/foundations/bike/docs/"`));
      assert.match(notes.html, /class="katex/);
      assert.ok(local.pages.find(p => p.title === 'My docs').html.includes(`${base}projects/foundations/bike/docs/notes/understanding/#example`));
      assert.deepEqual(Object.values(local.files).sort(), [path.join(project, 'src/example.py'), path.join(project, 'notebooks/example.ipynb')].sort());
      assert.match(notes.html, /href="[^\"]+example\.ipynb"/, 'Explicit notebook links must survive local rendering');
      assert.ok(!local.watches.some(w => w.folder.includes('data') || w.folder.includes('.venv')));
    }
    const next = path.join(projects, 'foundations/another');
    await write(path.join(next, 'README.md'), '# Another project\n\nNewly added.');
    assert.equal((await projectLibrary(projects, '/')).groups[0].projects.length, 2, 'New folders need no hand-maintained registry');
  } finally { await fs.rm(temporary, { recursive: true, force: true }); }
});

test('Quantization Lab is reachable from collections, projects and the existing tutorial in both builds', async () => {
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'quantization-reader-'));
  try {
    const projects = path.join(temporary, 'projects');
    const group = path.join(projects, 'quantization-lab');
    const write = async (file, text) => { await fs.mkdir(path.dirname(file), { recursive: true }); await fs.writeFile(file, text); };
    const lesson = path.join(temporary, 'docs/tutorials/assistant/lessons/01-start.md');
    await write(path.join(group, 'README.md'), '# Quantization Lab\n\n[Document Assistant](document-assistant/README.md) [Compressed Search](compressed-semantic-search/README.md)');
    await write(path.join(group, 'document-assistant/README.md'), '# Document Assistant\n\n[Start the tutorial](../../../docs/tutorials/assistant/lessons/01-start.md)');
    await write(path.join(group, 'compressed-semantic-search/README.md'), '# Compressed Semantic Search\n\nSearch with a smaller index.');
    await write(path.join(group, 'document-assistant/docs/implementation/00_setup/00-resume.md'), '# Setup\n\nPRIVATE QUANTIZATION NOTES');
    for (const base of ['/', '/project_docs/']) {
      for (const local of [false, true]) {
        const courseURL = `${base}records/tutorials/assistant/chapters/start/`;
        const library = await projectLibrary(projects, base, { local, links: new Map([[lesson, courseURL]]) });
        const collection = library.groups.find(item => item.id === 'quantization-lab');
        assert.ok(collection, 'The project collection must be discoverable');
        assert.equal(collection.projects.length, 2);
        assert.ok(collectionCards(library.groups).includes(`href="${base}projects/quantization-lab/"`));
        assert.ok(projectsSidebar(library, base).includes(`href="${base}projects/quantization-lab/"`));
        const assistant = collection.projects.find(item => item.id === 'document-assistant');
        assert.ok(collection.overview.html.includes(`href="${assistant.url}"`));
        assert.ok(assistant.overview.html.includes(`href="${courseURL}"`), 'Project README must lead to the real lesson route');
        assert.equal(library.pages.some(page => page.raw.includes('PRIVATE QUANTIZATION NOTES')), local);
      }
    }
  } finally { await fs.rm(temporary, { recursive: true, force: true }); }
});

test('local assets use an allowlist, stream byte ranges and hide the filesystem manifest', async () => {
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'project-media-'));
  let server;
  try {
    const media = path.join(temporary, 'video.mp4');
    await fs.writeFile(media, Buffer.from('0123456789'));
    await fs.writeFile(path.join(temporary, 'build.json'), JSON.stringify({ base: '/project_docs/' }));
    await fs.writeFile(path.join(temporary, '404.html'), 'Not found');
    await fs.writeFile(path.join(temporary, 'local-files.json'), JSON.stringify({ 'abc/video.mp4': media }));
    const running = await serve(0, { directory: temporary, local: true });
    server = running.server;
    const url = running.origin + '/project_docs/';
    const part = await fetch(url + 'local-files/abc/video.mp4', { headers: { Range: 'bytes=2-5' } });
    assert.equal(part.status, 206);
    assert.equal(part.headers.get('content-range'), 'bytes 2-5/10');
    assert.equal(part.headers.get('content-type'), 'video/mp4');
    assert.equal(await part.text(), '2345');
    assert.equal((await fetch(url + 'local-files/abc/video.mp4', { headers: { Range: 'bytes=20-' } })).status, 416);
    assert.equal((await fetch(url + 'local-files/abc/video.mp4', { headers: { Range: 'bytes=-3' } })).status, 206);
    assert.equal((await fetch(url + 'local-files.json')).status, 404);
    assert.equal((await fetch(url + '%6cocal-files.json')).status, 404);
    assert.equal((await fetch(url + 'local-files/not-allowed/video.mp4')).status, 404);
    assert.equal((await fetch(url + 'local-files/abc/video.mp4', { method: 'HEAD' })).headers.get('content-length'), '10');
    const publicServer = await serve(0, { directory: temporary });
    try { assert.equal((await fetch(publicServer.origin + '/project_docs/local-files/abc/video.mp4')).status, 404); }
    finally { await new Promise(resolve => publicServer.server.close(resolve)); }
  } finally { if (server) await new Promise(resolve => server.close(resolve)); await fs.rm(temporary, { recursive: true, force: true }); }
});
