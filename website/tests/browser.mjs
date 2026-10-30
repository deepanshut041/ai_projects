import { p00LegacyRoutes } from '../scripts/p00-layout.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import { serve } from '../scripts/serve.mjs';

const { server, origin, base } = await serve(0);
let browser;
// Long complete lessons can exceed practical browser raster limits. Their
// chapters and all controls are checked below; keep overview captures bounded
// and readable while retaining full-page captures for shorter documents.
async function captureLesson(page, file) {
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  await page.evaluate(() => window.scrollTo(0, 0));
  if (height <= 16000) return page.screenshot({ path: file, fullPage: true });
  const last = height - page.viewportSize().height;
  await page.screenshot({ path: file });
  for (const [label, y] of [['middle', Math.floor(last / 2)], ['end', last]]) {
    await page.evaluate(y => window.scrollTo(0, y), y);
    await page.screenshot({ path: file.replace(/\.png$/, `-${label}.png`) });
  }
  await page.evaluate(() => window.scrollTo(0, 0));
}
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
  await page.getByRole('searchbox', { name: 'Search records' }).fill('The command workflow, repository boundaries');
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
    assert.ok((await page.locator('article').innerText()).includes('P00-M1'));
  }
  for (const record of records) {
    await page.goto(origin + record.url); await ready();
    assert.equal(await page.getByRole('heading', { name: record.title, exact: true }).count(), 1);
    assert.equal(await page.locator('.katex-error').count(), 0, `Math rendering: ${record.id}`);
    await page.locator('article img').evaluateAll(images => Promise.all(images.map(img => img.decode())));
    // Check every record, including late chapters and supporting documents.
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Page overflow at ${width}px: ${record.id}`);
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    const md = await fetch(origin + base + 'source/' + record.file);
    assert.equal(md.status, 200); assert.match(await md.text(), /^---/);
    for (const link of await page.locator('article a[href^="' + base + '"]').evaluateAll(links => links.map(a => a.getAttribute('href')))) assert.equal((await fetch(origin + link)).status, 200, link);
  }
  const tutorialIndex = JSON.parse(await fs.readFile('../tutorials/index.json', 'utf8'));
  const authoredLessons = await Promise.all(tutorialIndex.projects.flatMap(project => project.lessons)
    .map(relative => fs.readFile(`../tutorials/${relative}/lesson.json`, 'utf8').then(JSON.parse)));
  await page.goto(origin + base); await ready();
  const searchableChapter = records.find(record => record.tutorial?.kind === 'chapter');
  assert.ok(searchableChapter, 'An authored chapter must be available for search');
  await page.getByRole('searchbox', { name: 'Search records' }).fill(searchableChapter.title);
  assert.ok(await page.locator('.record-card:visible').count() >= 1, 'Chapter text should find its tutorial');
  const searchableOverview = records.find(record => record.file === searchableChapter.tutorial.overviewFile);
  const chapterSearchResults = await page.locator('.record-card:visible').evaluateAll(links => links.map(link => link.getAttribute('href')));
  assert.ok(chapterSearchResults.includes(searchableOverview.url), 'Chapter search must include its owning tutorial, even when guides also refer to it');
  const tutorialOverviews = records.filter(record => record.tutorial?.kind === 'overview');
  await page.goto(origin + base + 'tutorials/'); await ready();
  assert.equal(await page.locator('.record-card').count(), authoredLessons.length, 'One card per tutorial, not per chapter');
  const numberedCourses = tutorialOverviews.filter(record => Number.isInteger(record.tutorial.courseNumber));
  const visibleTitles = await page.locator('.record-card h2').allTextContents();
  assert.deepEqual(visibleTitles.slice(0, numberedCourses.length), numberedCourses.map(record => record.title), 'Numbered courses must appear before references in reader order');
  for (const tutorial of tutorialOverviews) {
    const authored = authoredLessons.find(lesson => lesson.title === tutorial.title);
    assert.ok(authored, 'Tutorial must have an authored manifest');
    const pages = records.filter(record => record.tutorial?.overviewFile === tutorial.file);
    const chapters = pages.filter(record => record.tutorial.kind === 'chapter');
    await page.goto(origin + tutorial.url); await ready();
    assert.equal(await page.locator('.tutorial-board').count(), 0, 'Overview should not load every chapter');
    assert.equal(await page.getByRole('navigation', { name: 'Tutorial contents' }).locator('a').count(), pages.length);
    await page.screenshot({ path: `test-results/${authored.project_id.toLowerCase()}-reader-overview.png`, fullPage: true });
    const course = (tutorial.tutorial.courseFiles ?? []).map(file => pages.find(record => record.file === file));
    await page.locator('.tutorial-pager [rel="next"]').click(); await ready();
    assert.equal(new URL(page.url()).pathname, chapters[0].url);
    const secondPage = course.length ? course[1] : chapters[1];
    if (secondPage) {
      await page.locator('.tutorial-pager [rel="next"]').click(); await ready();
      assert.equal(new URL(page.url()).pathname, secondPage.url);
      await page.goBack(); await ready();
    } else assert.equal(await page.locator('.tutorial-pager [rel="next"]').count(), 0);
    assert.equal(new URL(page.url()).pathname, chapters[0].url);
    await page.reload(); await ready();
    for (const [index, chapter] of chapters.entries()) {
      await page.goto(origin + chapter.url); await ready();
      const navigation = page.getByRole('navigation', { name: 'Tutorial contents' });
      assert.equal(await navigation.locator('[aria-current="page"]').getAttribute('href'), chapter.url);
      const manifestChapter = authored.chapters[index];
      const chapterRaw = await fs.readFile(`../tutorials/projects/${authored.project_id.toLowerCase()}/${authored.slug}/${authored.revision}/${manifestChapter.content_path}`, 'utf8');
      assert.equal(await page.locator('.tutorial-board').count(), manifestChapter.diagram_ids.length + (chapterRaw.match(/!\[[^\]]*\]\(/g) ?? []).length);
      assert.equal(await page.locator('.tutorial-scenario').count(), manifestChapter.scenario_ids.length);
      assert.equal(await page.locator('.practice-reveal').count(), manifestChapter.practice_ids.length);
      assert.ok(!(await page.locator('article').innerText()).includes('[[diagram:'), 'Authoring insertions must be compiled');
      assert.equal(await page.locator('article h2, article h3').first().evaluate(el => el.tagName), 'H2');
      assert.ok(await page.locator('.chapter-outline a').evaluateAll(links => links.every(a => document.getElementById(a.getAttribute('href').slice(1)))));
      for (const region of await page.locator('article pre, .markdown-table, .katex-display').all()) {
        assert.equal(await region.getAttribute('tabindex'), '0', 'Wide code, tables and equations must be keyboard-scrollable');
      }
      if (tutorial.tutorial.projectArea) {
        const pager = page.getByRole('navigation', { name: 'Lesson navigation' });
        assert.equal(await pager.locator('[rel="prev"]').getAttribute('href'), index ? chapters[index - 1].url : tutorial.url);
        if (index + 1 < chapters.length) {
          await pager.locator('[rel="next"]').click(); await ready();
          assert.equal(new URL(page.url()).pathname, chapters[index + 1].url);
          await page.goBack(); await ready();
        } else assert.equal(await pager.locator('[rel="next"]').count(), 0);
      }
      if (index === 0) {
        await page.locator('.chapter-outline summary').focus();
        await page.keyboard.press('Enter');
        assert.notEqual(await page.locator('.chapter-outline').getAttribute('open'), null);
        await page.locator('.chapter-outline a').first().click();
        assert.ok(new URL(page.url()).hash, 'Chapter section links should navigate to an anchor');
        await page.locator('.chapter-outline summary').click();
        await page.evaluate(() => window.scrollTo(0, 0));
        const screenshot = `test-results/${authored.project_id.toLowerCase()}-reader-chapter`;
        await page.screenshot({ path: `${screenshot}-desktop.png`, fullPage: true });
        await page.screenshot({ path: `${screenshot}-top-desktop.png` });
        await page.setViewportSize({ width: 390, height: 844 });
        await page.reload(); await ready();
        assert.equal(await page.locator('.tutorial-menu').getAttribute('open'), null);
        await page.locator('.tutorial-menu > summary').click();
        assert.ok(await navigation.isVisible());
        await page.screenshot({ path: `${screenshot}-mobile-menu.png`, fullPage: true });
        if (course.length && !tutorial.tutorial.projectArea) await navigation.locator('.reference-menu').first().locator('summary').click();
        const mobileTarget = chapters[1] ?? chapters[0];
        await navigation.locator(`a[href="${mobileTarget.url}"]`).click(); await ready();
        assert.equal(new URL(page.url()).pathname, mobileTarget.url);
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
        await page.screenshot({ path: `${screenshot}-mobile.png`, fullPage: true });
        await page.screenshot({ path: `${screenshot}-top-mobile.png` });
        await page.setViewportSize({ width: 1440, height: 1000 });
      }
    }
    for (const support of pages.filter(record => record.tutorial.kind === 'support')) {
      await page.goto(origin + support.url); await ready();
      assert.equal(await page.getByRole('heading', { name: support.title, exact: true }).count(), 1);
      assert.equal(await page.locator('.tutorial-menu [aria-current="page"]').getAttribute('href'), support.url);
      if (support.tutorial.sourcePath.startsWith('build/')) {
        const buildPages = pages.filter(record => record.tutorial?.sourcePath.startsWith('build/'));
        const index = buildPages.findIndex(record => record.url === support.url);
        await page.locator('.chapter-outline summary').click();
        const outline = page.getByRole('navigation', { name: 'Page sections' });
        assert.ok(await outline.getByRole('link').count() > 0, 'Build pages need section links');
        assert.ok(await outline.getByRole('link').evaluateAll(links => links.every(a => document.getElementById(a.hash.slice(1)))), 'Build section links must target real headings');
        const pager = page.getByRole('navigation', { name: 'Lesson navigation' });
        if (index + 1 < buildPages.length) {
          await pager.locator('[rel="next"]').click(); await ready();
          assert.equal(new URL(page.url()).pathname, buildPages[index + 1].url, 'Build navigation must stay in the build sequence');
          await page.goBack(); await ready();
        } else {
          assert.equal(await pager.locator('[rel="next"]').count(), 0);
        }
        assert.equal(await pager.locator('[rel="prev"]').getAttribute('href'), index > 0 ? buildPages[index - 1].url : chapters[0].url);
      }
      if (['workbook.md', 'project/implementation.md'].includes(support.tutorial.sourcePath)) {
        const challengeLink = page.getByRole('link', { name: 'the challenge bank', exact: true });
        if (await challengeLink.count()) {
          await challengeLink.click(); await ready();
          assert.ok(new URL(page.url()).pathname.endsWith(tutorial.tutorial.projectArea ? '/documents/practice/questions/' : '/documents/whiteboard/questions/'));
        }
      }
    }
    if (authored.project_id === 'P00' && authored.slug === 'transformer-foundations' && authored.revision === 'v1') {
      for (const [from, to] of Object.entries(p00LegacyRoutes)) {
        const root = tutorial.url;
        await page.goto(origin + root + from + '/');
        await page.waitForURL(origin + root + to + '/'); await ready();
      }
    }
    const complete = pages.find(record => record.tutorial.kind === 'complete');
    await page.goto(origin + complete.url); await ready();
    const completeRaw = await fs.readFile(`../tutorials/projects/${authored.project_id.toLowerCase()}/${authored.slug}/${authored.revision}/lesson.md`, 'utf8');
    assert.equal(await page.locator('.tutorial-board').count(), (completeRaw.match(/!\[[^\]]*\]\(/g) ?? []).length);
    await page.locator('.tutorial-board img').evaluateAll(images => Promise.all(images.map(img => img.decode())));
    assert.ok(await page.locator('.tutorial-board img').evaluateAll(images => images.every(img => img.naturalWidth > 0)));
    assert.equal(await page.locator('.tutorial-scenario').count(), authored.scenarios.length);
    const scenarioIds = await page.locator('.tutorial-scenario').evaluateAll(controls =>
      controls.map(control => JSON.parse(control.dataset.scenario).id));
    for (const scenario of authored.scenarios) {
      const scenarioIndex = scenarioIds.indexOf(scenario.id);
      assert.ok(scenarioIndex >= 0, `Missing scenario: ${scenario.id}`);
      const control = page.locator('.tutorial-scenario').nth(scenarioIndex);
      for (const item of scenario.cases) {
        for (const variable of scenario.variables) {
          await control.getByRole('combobox', { name: variable.label, exact: true }).selectOption(item.when[variable.id]);
        }
        assert.equal(await control.locator('[data-outcome]').innerText(), item.outcome);
        assert.equal(await control.locator('[data-narrative]').innerText(), item.narrative);
      }
    }
    assert.equal(await page.locator('.practice-answer:visible').count(), 0);
    for (let practiceIndex = 0; practiceIndex < authored.practice.length; practiceIndex++) {
      await page.locator('.practice-reveal').nth(practiceIndex).click();
      assert.equal(await page.locator('.practice-answer:visible').count(), 1);
      assert.equal(await page.locator('.practice-reveal').nth(practiceIndex).innerText(), 'Hide worked answer');
      await page.locator('.practice-reveal').nth(practiceIndex).click();
    }
    await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
    assert.equal(await page.locator('.practice-answer:visible').count(), authored.practice.length);
    await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
    assert.equal(await page.locator('.practice-answer:visible').count(), 0);
    await page.evaluate(() => window.scrollTo(0, 0));
    const prefix = `test-results/${authored.project_id.toLowerCase()}-${authored.slug}-tutorial`;
    await captureLesson(page, `${prefix}-desktop.png`);
    await page.screenshot({ path: `${prefix}-top-desktop.png` });
    const diagramPage = await browser.newPage({ viewport: { width: 1600, height: 800 } });
    try {
      for (const diagram of authored.diagrams) {
        const diagramImage = page.locator(`.tutorial-board img[src$="/${diagram.id}.svg"]`);
        assert.equal(await diagramImage.count(), 1, `Declared diagram must appear once: ${diagram.id}`);
        const source = await diagramImage.getAttribute('src');
        await diagramPage.goto(origin + source);
        await diagramPage.locator('svg').screenshot({ path: `${prefix}-${diagram.id}.png` });
      }
    } finally { await diagramPage.close(); }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile overflow: tutorial');
    const scrollableBoards = page.locator('.tutorial-board:not([data-source-capture]) .diagram-scroll');
    if (await scrollableBoards.count()) {
      assert.ok(await scrollableBoards.first().evaluate(el => el.scrollWidth > el.clientWidth), 'Large mathematical diagrams should remain readable by scrolling');
    }
    for (const preview of await page.locator('.tutorial-board[data-source-capture]').all()) {
      const image = preview.locator('img');
      assert.ok(await image.evaluate(img => img.complete && img.naturalWidth > 0 && img.getBoundingClientRect().width <= img.parentElement.clientWidth + 1), 'The complete source diagram must fit its mobile preview');
      assert.equal(await preview.getByRole('link', { name: 'Open diagram at full size ↗', exact: true }).getAttribute('href'), await image.getAttribute('src'), 'Small source labels must remain available at full size');
    }
    await captureLesson(page, `${prefix}-mobile.png`);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: `${prefix}-top-mobile.png` });
    await page.setViewportSize({ width: 1440, height: 1000 });
  }
  const withoutJS = await browser.newPage({ javaScriptEnabled: false });
  const staticChapter = records.find(record => record.tutorial?.kind === 'chapter');
  await withoutJS.goto(origin + staticChapter.url);
  assert.ok(await withoutJS.getByRole('navigation', { name: 'Tutorial contents' }).isVisible());
  const staticNext = withoutJS.locator('.tutorial-pager [rel="next"]');
  if (await staticNext.count()) await staticNext.click();
  else await withoutJS.locator('.tutorial-pager [rel="prev"]').click();
  assert.notEqual(new URL(withoutJS.url()).pathname, staticChapter.url);
  await withoutJS.close();
  // Print a standalone chapter; complete-course controls are exercised above.
  // A chapter also keeps the exported PDF and screenshot within raster limits.
  await page.goto(origin + staticChapter.url); await ready();
  await page.evaluate(() => { window.print = () => { window.__printed = true; }; });
  await page.getByRole('button', { name: 'Print / Save PDF' }).click();
  assert.equal(await page.evaluate(() => window.__printed), true);
  await page.screenshot({ path: 'test-results/research-record.png', fullPage: true });
  await page.emulateMedia({ media: 'print' });
  assert.equal(await page.locator('body').evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(255, 255, 255)');
  assert.equal(await page.locator('.sidebar').evaluate(el => getComputedStyle(el).display), 'none');
  assert.equal(await page.locator('.tutorial-navigation').evaluate(el => getComputedStyle(el).display), 'none');
  assert.equal(await page.locator('.tutorial-pager').evaluate(el => getComputedStyle(el).display), 'none');
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
