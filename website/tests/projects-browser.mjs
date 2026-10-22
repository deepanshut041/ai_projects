import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { build, root } from '../scripts/build.mjs';
import { serve } from '../scripts/serve.mjs';

const { out, library } = await build({ local: true, directory: path.join(root, 'test-results/local-site') });
const { server, origin, base } = await serve(0, { directory: out, local: true });
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  const ready = () => page.waitForFunction(() => document.documentElement.dataset.ready === 'true');
  await page.goto(origin + base); await ready();
  await page.locator('.project-collections').getByRole('link', { name: /Foundations/ }).click(); await ready();
  assert.equal(await page.locator('.project-card').count(), 4);
  assert.equal(await page.locator('.project-card h2').first().innerText(), 'Bike Demand Planner');
  await page.screenshot({ path: 'test-results/project-foundations-desktop.png', fullPage: true });
  await page.locator('.project-card').filter({ hasText: 'Bike Demand Planner' }).click(); await ready();
  assert.equal(await page.getByRole('heading', { name: 'Phases', exact: true }).count(), 1);
  assert.equal(await page.getByRole('heading', { name: 'Implementation', exact: true }).count(), 1);
  assert.equal(await page.getByRole('heading', { name: 'October 2026', exact: true }).count(), 0);
  assert.ok(await page.locator('.implementation-folder > summary').filter({ hasText: '02_hourly-data-audit-and-splits' }).isVisible());
  await page.locator('.implementation-folder > summary').filter({ hasText: '02_hourly-data-audit-and-splits' }).click();
  assert.ok(await page.locator('.project-folders').getByRole('link', { name: /02-notes-to-understanding.md Understanding the hourly data/ }).isVisible());
  await page.getByRole('link', { name: 'Open project docs', exact: false }).click(); await ready();
  const docsURL = new URL(page.url()).pathname;
  assert.equal(docsURL, base + 'projects/foundations/bike-demand-planner/docs/');
  const bike = library.groups[0].projects.find(project => project.id === 'bike-demand-planner');
  const notes = bike.pages.find(p => p.file.endsWith('02_hourly-data-audit-and-splits/02-notes-to-understanding.md'));
  const phase = bike.pages.find(p => p.file === 'docs/phases/01-data-and-splits/README.md');
  const phaseFolder = page.getByRole('navigation', { name: 'Project contents' }).locator('details').filter({ has: page.locator('summary').filter({ hasText: /^Phases$/ }) });
  await phaseFolder.locator(':scope > summary').click();
  await page.getByRole('navigation', { name: 'Project contents' }).getByRole('link', { name: phase.title, exact: true }).click(); await ready();
  assert.equal(new URL(page.url()).pathname, phase.url);
  await page.locator('article').getByRole('link', { name: 'current implementation record', exact: true }).click(); await ready();
  await page.locator('article').getByRole('link', { name: 'understanding notes', exact: true }).first().click(); await ready();
  assert.equal(new URL(page.url()).pathname, notes.url);
  assert.ok(await page.locator('.project-diagram svg').count() > 0, 'Mermaid must render as a diagram');
  const players = page.locator('.section-video video');
  assert.equal(await players.count(), 6, 'Short videos belong in the understanding notes');
  for (const player of await players.all()) {
    assert.equal(await player.getAttribute('preload'), 'none');
    assert.equal(await player.getAttribute('autoplay'), null);
    assert.ok(await player.evaluate(video => video.controls && video.playsInline && video.parentElement.previousElementSibling.tagName === 'H2'), 'Each native player precedes its matching written section');
    await player.evaluate(video => { video.load(); });
    await player.evaluate(video => new Promise((resolve, reject) => { if (video.readyState >= 1) resolve(); else { video.addEventListener('loadedmetadata', resolve, { once: true }); video.addEventListener('error', reject, { once: true }); } }));
    assert.ok(await player.evaluate(video => video.duration > 270 && video.duration < 480), 'Expanded clips include the new worked examples');
    await player.evaluate(video => video.play());
    await page.waitForFunction(video => video.currentTime > .15, await player.elementHandle());
    await player.evaluate(video => { video.pause(); video.currentTime = Math.min(30, video.duration / 2); video.textTracks[0].mode = 'hidden'; });
    await page.waitForFunction(video => video.currentTime >= 29.9 && video.paused && video.textTracks[0].cues?.length > 0, await player.elementHandle());
    assert.ok(await player.evaluate(video => [...video.textTracks[0].cues].every(cue => cue.startTime >= 0 && cue.endTime <= video.duration + .1)));
    const later = await player.evaluate(video => { video.currentTime = video.duration - 20; return video.currentTime; });
    await player.evaluate(video => video.play());
    await page.waitForFunction(({ video, later }) => video.currentTime > later + .15 && video.readyState >= 2, { video: await player.elementHandle(), later });
    await player.evaluate(video => video.pause());
  }
  assert.equal(await page.locator('.katex-error').count(), 0);
  const current = page.getByRole('navigation', { name: 'Project contents' }).locator('[aria-current="page"]');
  assert.equal(await current.getAttribute('href'), notes.url);
  const download = await page.getByRole('link', { name: 'Download Markdown' }).getAttribute('href');
  assert.equal(await (await fetch(origin + download)).text(), notes.raw);
  await page.screenshot({ path: 'test-results/project-notes-desktop.png' });
  // Check every generated document and linked local resource without making
  // dozens of duplicate screenshots of long working notes.
  for (const document of library.pages) {
    assert.equal((await fetch(origin + document.url)).status, 200, document.url);
    for (const [, link] of document.html.matchAll(/(?:href|src)="([^"]+)"/g)) {
      if (link.startsWith(base)) assert.equal((await fetch(origin + link, { method: 'HEAD' })).status, 200, link);
    }
  }
  const video = Object.entries(library.files).find(([, file]) => file.endsWith('out/video.mp4'));
  assert.ok(video, 'The existing walkthrough video remains reachable');
  const clip = await fetch(origin + base + 'local-files/' + video[0], { headers: { Range: 'bytes=0-1023' } });
  assert.equal(clip.status, 206);
  assert.equal((await clip.arrayBuffer()).byteLength, 1024);
  const routes = [base, base + 'projects/', ...library.groups.map(g => g.url), bike.url, docsURL, phase.url, notes.url];
  for (const width of [1440, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    for (const route of routes) {
      await page.goto(origin + route); await ready();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Overflow at ${width}px: ${route}`);
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(origin + docsURL); await ready();
  await page.locator('.project-menu > summary').click();
  assert.ok(await page.getByRole('navigation', { name: 'Project contents' }).isVisible());
  await page.screenshot({ path: 'test-results/project-docs-mobile.png' });
  await page.goto(origin + base + 'projects/foundations/'); await ready();
  await page.screenshot({ path: 'test-results/project-foundations-mobile.png', fullPage: true });
  const plain = await browser.newPage({ javaScriptEnabled: false });
  await plain.goto(origin + notes.url);
  assert.ok(await plain.getByRole('navigation', { name: 'Project contents' }).isVisible());
  await plain.getByRole('navigation', { name: 'Breadcrumb' }).getByRole('link', { name: 'Foundations', exact: true }).click();
  assert.equal(new URL(plain.url()).pathname, base + 'projects/foundations/');
  await plain.close();
  const exported = JSON.parse(await fs.readFile(path.join(root, 'dist/projects/index.json'), 'utf8'));
  assert.ok(exported.every(g => g.projects.every(p => p.pages.length === 1 && p.pages[0].file === 'README.md')), 'Default static export must exclude private project docs');
  await assert.rejects(fs.stat(path.join(root, 'dist/local-files.json')), { code: 'ENOENT' });
  await assert.rejects(fs.stat(path.join(root, 'dist/projects/foundations/bike-demand-planner/docs')), { code: 'ENOENT' });
  assert.deepEqual(errors, []);
  console.log(`Project browser checks passed at ${base}: ${library.groups.length} collections, ${library.pages.length} pages, numbered folders and filenames, desktop/mobile navigation, Mermaid, downloads, video ranges and private-export separation.`);
} finally { if (browser) await browser.close(); await new Promise(resolve => server.close(resolve)); }
