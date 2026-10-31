import test from 'node:test';
import assert from 'node:assert/strict';
import { renderer, presentationDefinition } from '../scripts/content.mjs';

test('original presentations keep usable source links and bounded interactive viewers', () => {
  for (let lecture = 1; lecture <= 8; lecture++) {
    const block = { title: `Vizuara · Lecture ${lecture}`, url: `https://lecture${lecture}-attention-mechanisms.vercel.app/#/1` };
    const { md } = renderer(new Set(), '', '/nested/');
    const html = md.render('```presentation\n' + JSON.stringify(block) + '\n```');
    assert.ok(html.includes(`href="${block.url}"`));
    assert.ok(html.includes(`src="${block.url}"`));
    assert.match(html, /<details><summary>Read the original slides here/);
    assert.match(html, /loading="lazy" sandbox="allow-scripts allow-same-origin"/);
    assert.match(html, /referrerpolicy="no-referrer"/);
    assert.doesNotMatch(html, /<pre/);
  }
});

test('presentation input cannot introduce arbitrary origins, attributes or markup', () => {
  const url = 'https://lecture1-attention-mechanisms.vercel.app/';
  for (const bad of ['javascript:alert(1)', 'https://example.org/', url + '?redirect=https://example.org',
    url + 'elsewhere', 'https://lecture1-attention-mechanisms.vercel.app.evil.test/',
    'https://user:password@lecture1-attention-mechanisms.vercel.app/',
    'https://lecture9-attention-mechanisms.vercel.app/', url + '#/<script>']) {
    assert.throws(() => presentationDefinition(JSON.stringify({ title: 'Slides', url: bad })));
  }
  assert.throws(() => presentationDefinition(JSON.stringify({ title: 'Slides', url, onload: 'alert(1)' })));
  const { md } = renderer(new Set(), '', '/');
  const html = md.render('```presentation\n' + JSON.stringify({ title: '<script>unsafe</script>', url }) + '\n```');
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;/);
});
