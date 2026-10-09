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
  if (records.some(record => record.id === 'guides/roadmap')) {
    await page.getByRole('link', { name: 'Roadmap', exact: true }).click(); await ready();
    assert.equal(await page.getByRole('heading', { name: 'Projects learning roadmap', exact: true }).count(), 1);
    assert.ok((await page.locator('article').innerText()).includes('P19'));
    await page.screenshot({ path: 'test-results/roadmap-desktop.png', fullPage: true });
    await page.getByRole('link', { name: 'resume.md', exact: true }).first().click(); await ready();
    assert.ok((await page.locator('article').innerText()).includes('P01-M1'));
  }
  for (const record of records) {
    await page.goto(origin + record.url); await ready();
    assert.equal(await page.getByRole('heading', { name: record.title, exact: true }).count(), 1);
    const md = await fetch(origin + base + 'source/' + record.file);
    assert.equal(md.status, 200); assert.match(await md.text(), /^---/);
    for (const link of await page.locator('article a[href^="' + base + '"]').evaluateAll(links => links.map(a => a.getAttribute('href')))) assert.equal((await fetch(origin + link)).status, 200, link);
  }
  const tutorial = records.find(record => record.category === 'tutorials');
  if (tutorial) {
    const index = JSON.parse(await fs.readFile('tutorials/index.json', 'utf8'));
    const authoredLessons = await Promise.all(index.projects.flatMap(project => project.lessons)
      .map(relative => fs.readFile(`tutorials/${relative}/lesson.json`, 'utf8').then(JSON.parse)));
    const authored = authoredLessons.find(lesson => lesson.title === tutorial.title);
    assert.ok(authored, 'Tutorial must have an authored manifest');
    await page.goto(origin + tutorial.url); await ready();
    assert.equal(await page.locator('.tutorial-board').count(), authored.diagrams.length);
    await page.locator('.tutorial-board img').evaluateAll(images => Promise.all(images.map(img => img.decode())));
    assert.ok(await page.locator('.tutorial-board img').evaluateAll(images => images.every(img => img.naturalWidth > 0)));
    const control = page.locator('.tutorial-scenario');
    for (const [mask, future, result] of [['causal','8','2'],['causal','98','2'],['unmasked','8','4.666666667'],['unmasked','98','34.666666667']]) {
      await control.getByLabel('Visibility rule').selectOption(mask);
      await control.getByLabel('Future value').selectOption(future);
      assert.equal(await control.locator('[data-outcome]').innerText(), result);
    }
    assert.equal(await page.locator('.practice-answer:visible').count(), 0);
    await page.locator('.practice-reveal').first().click();
    assert.equal(await page.locator('.practice-answer:visible').count(), 1);
    await page.locator('.practice-reveal').first().click();
    await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
    assert.equal(await page.locator('.practice-answer:visible').count(), authored.practice.length);
    await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
    assert.equal(await page.locator('.practice-answer:visible').count(), 0);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: 'test-results/tutorial-desktop.png', fullPage: true });
    await page.screenshot({ path: 'test-results/tutorial-top-desktop.png' });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile overflow: tutorial');
    assert.ok(await page.locator('.diagram-scroll').first().evaluate(el => el.scrollWidth > el.clientWidth), 'Large diagrams should remain readable by scrolling');
    await page.screenshot({ path: 'test-results/tutorial-mobile.png', fullPage: true });
    await page.screenshot({ path: 'test-results/tutorial-top-mobile.png' });
    await page.setViewportSize({ width: 1440, height: 1000 });
  }
  await page.evaluate(() => { window.print = () => { window.__printed = true; }; });
  await page.getByRole('button', { name: 'Print / Save PDF' }).click();
  assert.equal(await page.evaluate(() => window.__printed), true);
  await page.screenshot({ path: 'test-results/research-record.png', fullPage: true });
  await page.emulateMedia({ media: 'print' });
  assert.equal(await page.locator('body').evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(255, 255, 255)');
  assert.equal(await page.locator('.sidebar').evaluate(el => getComputedStyle(el).display), 'none');
  assert.equal(await page.locator('.workspace').evaluate(el => getComputedStyle(el).minHeight), '0px');
  await page.pdf({ path: 'test-results/record.pdf', format: 'A4', printBackground: true });
  await page.emulateMedia({ media: 'screen' });
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
  if (records.some(record => record.id === 'guides/roadmap')) {
    await page.goto(origin + base + 'records/guides/roadmap/'); await ready();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile overflow: roadmap');
    await page.screenshot({ path: 'test-results/roadmap-mobile.png', fullPage: true });
  }
  const traversal = await fetch(origin + base + '%2e%2e%2fpackage.json');
  assert.equal(traversal.status, 404);
  assert.deepEqual(errors, []);
  console.log(`Browser checks passed: ${records.length} records, ${catalog.sources.length} sources, search, filters, downloads, record links, print action, and mobile layouts.`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
