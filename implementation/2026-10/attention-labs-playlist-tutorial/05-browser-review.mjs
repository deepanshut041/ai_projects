import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { build } from '../../../website/scripts/build.mjs';
import { serve } from '../../../website/scripts/serve.mjs';

const require = createRequire(new URL('../../../website/package.json', import.meta.url));
const { chromium } = require('playwright');
const website = fileURLToPath(new URL('../../../website/', import.meta.url));
const label = process.env.SITE_BASE === '/' ? 'root' : 'prefix';
const output = path.join(website, 'test-results', `attention-${label}`);
await fs.mkdir(output, { recursive: true });
const directory = path.join(output, 'site');
await build({ directory });
const { server, origin, base } = await serve(0, { directory });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const paths = [
  ['attention-foundations', ['map', 'summary', 'additive', 'scale', 'self', 'projection', 'causal']],
  ['attention-heads', ['heads', 'layout', 'boundary']],
];
const review = { base, chapters: [], original_decks_loaded: [], widths: [1440, 390, 320] };
const ready = () => page.waitForFunction(() => document.documentElement.dataset.ready === 'true');
try {
  await page.goto(origin + base + 'projects/attention-labs/');
  await ready();
  assert.equal(await page.locator('h1').innerText(), 'Attention Labs');
  const collection = await (await fetch(origin + base + 'projects/index.json')).json();
  const group = collection.find(item => item.id === 'attention-labs');
  assert.deepEqual(group.projects.map(item => item.id).sort(), ['attention-foundations', 'attention-heads']);
  await page.screenshot({ path: path.join(output, 'collection-desktop.png') });
  let lecture = 0;
  for (const [slug, chapters] of paths) {
    const overview = `${base}records/tutorials/attention-labs-${slug}-v1/`;
    await page.goto(origin + overview);
    await ready();
    assert.ok(await page.getByRole('link', { name: /Begin Lesson 1/ }).count() >= 1);
    for (const chapter of chapters) {
      const route = overview + `chapters/${chapter}/`;
      await page.goto(origin + route);
      await ready();
      assert.equal(await page.locator('.katex-error').count(), 0);
      assert.ok(await page.locator('.katex').count() > 0 || chapter === 'map');
      assert.equal(await page.getByRole('navigation', { name: 'Tutorial contents' }).locator('[aria-current="page"]').count(), 1);
      const reveal = page.locator('.practice-reveal').first();
      await reveal.click();
      assert.equal(await reveal.getAttribute('aria-expanded'), 'true');
      const download = await page.getByRole('link', { name: 'Download Markdown' }).getAttribute('href');
      assert.equal((await fetch(origin + download)).status, 200);
      for (const image of await page.locator('.tutorial-board img').all()) {
        await image.evaluate(img => img.decode());
        assert.ok(await image.evaluate(img => img.naturalWidth > 0));
        await page.setViewportSize({ width: 1920, height: 1080 });
        await image.screenshot({ path: path.join(output, `${slug}-${chapter}-figure.png`) });
        await page.setViewportSize({ width: 1440, height: 960 });
        const src = await image.getAttribute('src');
        // Measure every authored SVG text element at its intrinsic size.
        const clipped = await page.evaluate(async url => {
          const doc = new DOMParser().parseFromString(await (await fetch(url)).text(), 'image/svg+xml');
          const svg = document.importNode(doc.documentElement, true);
          svg.style.position = 'absolute'; svg.style.left = '-10000px';
          document.body.append(svg);
          const width = svg.viewBox.baseVal.width;
          const height = svg.viewBox.baseVal.height;
          const bad = [...svg.querySelectorAll('text')].filter(el => {
            const b = el.getBBox(); return b.x < 0 || b.y < 0 || b.x + b.width > width || b.y + b.height > height;
          }).map(el => el.textContent);
          svg.remove(); return bad;
        }, src);
        assert.deepEqual(clipped, [], `Clipped SVG labels: ${route}`);
      }
      const presentation = page.locator('.original-presentation');
      if (await presentation.count()) {
        lecture++;
        const expected = `https://lecture${lecture}-attention-mechanisms.vercel.app/#/1`;
        assert.equal(await presentation.locator('iframe').getAttribute('src'), expected);
        assert.equal(await presentation.getByRole('link', { name: 'Open original presentation' }).getAttribute('href'), expected);
        await presentation.locator('summary').click();
        const frame = page.frameLocator('.original-presentation iframe');
        await frame.locator('.slidev-page').first().waitFor({ state: 'visible', timeout: 30000 });
        assert.ok((await frame.locator('body').innerText()).length > 80);
        const next = frame.getByRole('button', { name: 'Go to next slide', exact: true });
        await next.click();
        review.original_decks_loaded.push(lecture);
        if ([1, 6, 8].includes(lecture)) await presentation.screenshot({ path: path.join(output, `original-deck-${lecture}.png`) });
      }
      assert.equal(await page.locator('iframe[src*="lecture9-"]').count(), 0);
      for (const width of review.widths) {
        await page.setViewportSize({ width, height: 960 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Page overflow at ${width}: ${route}`);
        for (const diagram of await page.locator('.diagram-scroll').all()) {
          if (await diagram.evaluate(el => el.scrollWidth > el.clientWidth)) {
            await diagram.focus();
            await page.keyboard.press('End');
            await diagram.evaluate(el => { el.scrollLeft = el.scrollWidth; });
            assert.ok(await diagram.evaluate(el => el.scrollLeft > 0), `Diagram cannot scroll: ${route}`);
            await diagram.evaluate(el => { el.scrollLeft = 0; });
          }
        }
      }
      if (['map', 'projection', 'causal', 'heads', 'layout'].includes(chapter)) {
        await page.screenshot({ path: path.join(output, `${chapter}-mobile.png`) });
        await page.setViewportSize({ width: 1440, height: 960 });
        await page.screenshot({ path: path.join(output, `${chapter}-desktop.png`) });
      }
      await page.emulateMedia({ media: 'print' });
      assert.equal(await page.locator('.practice-answer').first().evaluate(el => getComputedStyle(el).display), 'block');
      if (await presentation.count()) {
        assert.equal(await presentation.locator('details').evaluate(el => getComputedStyle(el).display), 'none');
        assert.ok(await presentation.getByRole('link', { name: 'Open original presentation' }).isVisible());
      }
      await page.emulateMedia({ media: 'screen' });
      await page.setViewportSize({ width: 1440, height: 960 });
      review.chapters.push({ slug, chapter, route, markdown_download: true, practice_reveal: true, print: true });
    }
  }
  assert.deepEqual(review.original_decks_loaded, [1, 2, 3, 4, 5, 6, 7, 8]);
  review.passed = true;
  await fs.writeFile(path.join(output, 'review.json'), JSON.stringify(review, null, 2) + '\n');
  console.log(JSON.stringify({ base, chapters: review.chapters.length, decks: review.original_decks_loaded.length, widths: review.widths, passed: true }));
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
