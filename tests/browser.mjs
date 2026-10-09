import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import { serve } from '../scripts/serve.mjs';

const { server, origin, base } = await serve(0);
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
  await fs.mkdir('test-results', { recursive: true });
  const ready = () => page.waitForFunction(() => document.documentElement.dataset.ready === 'true');
  await page.goto(origin + base); await ready();
  assert.ok(await page.locator('.record-card').count() >= 4);
  await page.screenshot({ path: 'test-results/research-desktop.png', fullPage: true });
  await page.getByRole('searchbox', { name: 'Search records' }).fill('Train a linear regressor');
  assert.equal(await page.locator('.record-card:visible').count(), 1);
  await page.getByRole('searchbox').fill('unfindable-947');
  assert.ok(await page.locator('#record-empty').isVisible());
  await page.getByRole('searchbox').fill('');
  const records = await (await fetch(origin + base + 'search.json')).json();
  for (const record of records) {
    await page.goto(origin + record.url); await ready();
    assert.equal(await page.getByRole('heading', { name: record.title, exact: true }).count(), 1);
    const md = await fetch(origin + base + 'source/' + record.file);
    assert.equal(md.status, 200); assert.match(await md.text(), /^---/);
    for (const link of await page.locator('article a[href^="' + base + '"]').evaluateAll(links => links.map(a => a.getAttribute('href')))) assert.equal((await fetch(origin + link)).status, 200, link);
  }
  await page.evaluate(() => { window.print = () => { window.__printed = true; }; });
  await page.getByRole('button', { name: 'Print / Save PDF' }).click();
  assert.equal(await page.evaluate(() => window.__printed), true);
  await page.screenshot({ path: 'test-results/research-record.png', fullPage: true });
  await page.pdf({ path: 'test-results/record.pdf', format: 'A4', printBackground: true });
  await page.goto(origin + base + 'sources/'); await ready();
  const catalog = await (await fetch(origin + base + 'catalog/sources.json')).json();
  assert.equal(await page.locator('.source-row').count(), catalog.sources.length);
  await page.getByLabel('Collection', { exact: true }).selectOption('attention');
  assert.equal(await page.locator('.source-row:visible').count(), 1);
  await page.getByLabel('Collection', { exact: true }).selectOption('');
  await page.getByLabel('Type', { exact: true }).selectOption('Paper');
  assert.equal(await page.locator('.source-row:visible').count(), catalog.sources.filter(s => s.kind === 'Paper').length);
  await page.getByLabel('Search sources').fill('unfindable-947');
  assert.equal(await page.locator('.source-row:visible').count(), 0);
  assert.ok(await page.locator('#source-empty').isVisible());
  await page.getByLabel('Search sources').fill('');
  await page.screenshot({ path: 'test-results/source-register.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of ['', 'sources/', 'implementations/', 'guides/', 'experiments/', 'questions/', 'guide/', 'records/' + records[0].id + '/']) {
    await page.goto(origin + base + route); await ready();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `Mobile overflow: ${route}`);
  }
  await page.goto(origin + base); await ready();
  await page.screenshot({ path: 'test-results/research-mobile.png', fullPage: true });
  const traversal = await fetch(origin + base + '%2e%2e%2fpackage.json');
  assert.equal(traversal.status, 404);
  assert.deepEqual(errors, []);
  console.log(`Browser checks passed: ${records.length} records, ${catalog.sources.length} sources, search, filters, downloads, record links, print action, and mobile layouts.`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
