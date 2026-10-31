import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { normalizeBase, readRecords, validateCatalog, renderer, sourceURL } from '../scripts/content.mjs';
import { workspaceRecords } from '../scripts/workspace.mjs';
import { artifact, tutorialRecords } from '../scripts/tutorials.mjs';
import { recordPage } from '../scripts/templates.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const docsRoot = fileURLToPath(new URL('../../', import.meta.url));
test('tutorial exports preserve prefixes and reject changed or escaping artifacts', async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'ai-tutorials-'));
  try {
    await fs.cp(path.join(docsRoot, 'tutorials'), path.join(temp, 'tutorials'), { recursive: true });
    const tutorial = await tutorialRecords(temp);
    assert.equal(tutorial.index.projects.filter(project => /^P\d{2}$/.test(project.id)).length, 20);
    assert.ok(tutorial.index.projects.some(project => project.id === 'bike-demand-planner'));
    const lessonCount = tutorial.index.projects.reduce((sum, project) => sum + project.lessons.length, 0);
    assert.ok(tutorial.records.length > lessonCount);
    assert.ok(tutorial.records.some(record => record.file.startsWith('tutorials/p00-')));
    const records = await readRecords(path.join(temp, 'content'), '/nested/', tutorial.records);
    const overviews = records.filter(record => record.tutorial.kind === 'overview');
    assert.equal(overviews.length, lessonCount);
    const numbered = overviews.filter(record => Number.isInteger(record.tutorial.courseNumber));
    assert.deepEqual(numbered.map(record => record.tutorial.courseNumber), tutorial.index.course_sequence.map((_, index) => index + (tutorial.index.course_number_start ?? 1)));
    for (const [index, relative] of tutorial.index.course_sequence.entries()) {
      assert.ok(numbered[index].tutorial.assetBase.endsWith(relative.split('/').slice(2).join('/')));
      assert.equal(numbered[index].tutorial.courseRole, 'course');
    }
    assert.equal(overviews.filter(record => record.tutorial.courseRole === 'reference').length, tutorial.index.reference_lessons.length);
    for (const overview of overviews) {
      const lessonPath = tutorial.index.projects.flatMap(project => project.lessons).find(relative => overview.tutorial.assetBase.endsWith(relative.split('/').slice(2).join('/')));
      const lesson = JSON.parse(await fs.readFile(path.join(temp, 'tutorials', lessonPath, 'lesson.json'), 'utf8'));
      const pages = records.filter(record => record.tutorial.overviewFile === overview.file);
      const chapters = pages.filter(record => record.tutorial.kind === 'chapter');
      assert.equal(chapters.length, lesson.chapters.length);
      assert.equal(pages.filter(record => record.tutorial.kind === 'complete').length, 1);
      assert.doesNotMatch(overview.html, /class="practice-reveal"|class="tutorial-board"/);
      for (const [index, chapter] of chapters.entries()) {
        const authored = lesson.chapters[index];
        assert.ok(chapter.id.endsWith('/chapters/' + authored.id));
        assert.equal(chapter.tutorial.sourcePath, authored.content_path);
        const chapterRaw = await fs.readFile(path.join(temp, 'tutorials', lessonPath, authored.content_path), 'utf8');
        const authoredImages = (chapterRaw.match(/!\[[^\]]*\]\(/g) ?? []).length;
        assert.equal((chapter.html.match(/class="tutorial-board"/g) ?? []).length, authored.diagram_ids.length + authoredImages);
        assert.equal((chapter.html.match(/class="practice-reveal"/g) ?? []).length, authored.practice_ids.length);
        assert.equal((chapter.html.match(/class="tutorial-scenario"/g) ?? []).length, authored.scenario_ids.length);
        assert.doesNotMatch(chapter.html, /\[\[(?:diagram|table|practice|scenario):/);
        assert.match(chapter.html, /<h2\b/, 'A standalone chapter needs sections below its h1');
        const page = recordPage(chapter, '/nested/');
        assert.match(page, /aria-label="Tutorial contents"/);
        assert.match(page, /aria-label="Chapter sections"/);
        assert.doesNotMatch(page, /<section id="references">/, 'Keep the complete bibliography on the overview');
        if (chapter.sources.length) assert.ok(page.includes(`${overview.url}#references`));
        assert.match(page, new RegExp(`href="${chapter.url}" aria-current="page"`));
        assert.match(page, /rel="prev"/);
        if (index < chapters.length - 1) assert.match(page, /rel="next"/);
        else assert.doesNotMatch(page, /rel="next"/);
      }
      if (overview.tutorial.courseFiles.length) {
        const course = overview.tutorial.courseFiles.map(file => pages.find(record => record.file === file));
        const expectedCourseLength = overview.tutorial.projectArea ? lesson.chapters.length : 1 + pages.filter(page => page.tutorial.sourcePath.startsWith('build/')).length;
        assert.equal(course.length, expectedCourseLength);
        assert.equal(new Set(course.map(page => page.file)).size, expectedCourseLength);
        if (lesson.project_id === 'P00' && lesson.slug === 'transformer-foundations') {
          assert.deepEqual(course.map(page => page.tutorial.sourcePath), lesson.chapters.map(chapter => chapter.content_path));
          assert.equal(course[0].tutorial.sourcePath, 'lessons/01-prediction.md');
          assert.equal(course[1].tutorial.sourcePath, 'lessons/02-workspace.md');
          assert.doesNotMatch(overview.html, /Begin with one prediction|Setup comes in Lesson 2|picks up the same example/);
        }
        assert.match(overview.html, /Begin Lesson 1/);
        assert.ok(overview.html.includes(course[0].url));
        for (const [index, lesson] of course.entries()) {
          const html = recordPage(lesson, '/nested/');
          assert.match(html, /aria-label="Lesson navigation"/);
          const previous = index ? course[index - 1] : overview;
          assert.ok(html.includes(`rel="prev" href="${previous.url}"`));
          if (index < course.length - 1) assert.ok(html.includes(`rel="next" href="${course[index + 1].url}"`));
          else assert.doesNotMatch(html, /rel="next"/);
          assert.match(html, /IMPLEMENTATION AND SOURCES/);
          assert.match(html, /PRACTICE AND REVIEW/);
          assert.doesNotMatch(html, /SUPPORTING MATERIAL|Reference chapters/);
        }
        if (overview.tutorial.projectArea) assert.equal(pages.filter(page => page.tutorial.sourcePath.startsWith('lessons/')).length, lesson.chapters.length);
        if (lesson.project_id === 'P00' && lesson.slug === 'transformer-foundations') assert.ok(tutorial.redirects.length > 0);
        for (const redirect of tutorial.redirects) assert.ok(records.some(record => record.url === '/nested/' + redirect.to));
      }
      const complete = pages.find(record => record.tutorial.kind === 'complete');
      assert.ok(complete.html.includes(`href="/nested/${overview.tutorial.assetBase}/`));
      assert.match(complete.html, /class="practice-reveal"/);
      for (const scenario of lesson.scenarios) for (const variable of scenario.variables) {
        assert.ok(complete.html.includes(`data-variable="${variable.id}"`));
      }
      const completeRaw = await fs.readFile(path.join(temp, 'tutorials', lessonPath, 'lesson.md'), 'utf8');
      assert.equal((complete.html.match(/class="tutorial-board"/g) ?? []).length, (completeRaw.match(/!\[[^\]]*\]\(/g) ?? []).length);
      const defensePath = overview.tutorial.projectArea ? 'practice/explain-your-model.md' : 'whiteboard/defense.md';
      const defense = pages.find(record => record.tutorial.sourcePath === defensePath);
      if (['transformer-foundations', 'small-gpt'].includes(lesson.slug)) assert.ok(defense);
    }
    const expectedSvgAssets = [];
    for (const project of tutorial.index.projects) {
      for (const relative of project.lessons) {
        const lesson = JSON.parse(await fs.readFile(path.join(temp, 'tutorials', relative, 'lesson.json'), 'utf8'));
        // Authored passive SVG figures are frozen assets too, alongside
        // schema-generated relationship diagrams.
        const rendered = JSON.parse(await fs.readFile(path.join(temp, 'tutorials', relative, 'render.json'), 'utf8'));
        expectedSvgAssets.push(...Object.keys(rendered.files).filter(file => file.endsWith('.svg')).map(file =>
          `tutorial-assets/${project.id.toLowerCase()}/${lesson.slug}/${lesson.revision}/${file}`));
      }
    }
    assert.deepEqual(tutorial.assets.filter(asset => asset.file.endsWith('.svg')).map(asset => asset.file).sort(), expectedSvgAssets.sort());
    await assert.rejects(artifact(temp, '../README.md'), /Unsafe/);
    const indexFile = path.join(temp, 'tutorials/index.json');
    const originalIndex = await fs.readFile(indexFile, 'utf8');
    const invalidIndex = JSON.parse(originalIndex);
    invalidIndex.reference_lessons.push(invalidIndex.course_sequence[0]);
    await fs.writeFile(indexFile, JSON.stringify(invalidIndex));
    await assert.rejects(tutorialRecords(temp), /Invalid tutorial course sequence/);
    await fs.writeFile(indexFile, originalIndex);
    await fs.writeFile(indexFile, JSON.stringify({ ...JSON.parse(originalIndex), course_number_start: -1 }));
    await assert.rejects(tutorialRecords(temp), /Invalid tutorial course number start/);
    await fs.writeFile(indexFile, originalIndex);
    const chapter = path.join(temp, 'tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md');
    await fs.appendFile(chapter, '\nUnvalidated alteration.');
    await assert.rejects(tutorialRecords(temp), /changed after rendering/);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
test('canonical checkpoints render from owner sources with portable links and current dates', async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'ai-canonical-'));
  const site = path.join(temp, 'ai_projects/docs');
  await fs.mkdir(path.join(site, 'content'), { recursive: true });
  await fs.mkdir(path.join(site, 'guide'), { recursive: true });
  try {
    assert.deepEqual(await workspaceRecords(site, 'https://example.com/research/'), []);
    for (const name of ['roadmap', 'progress', 'resume', 'decisions']) {
      await fs.writeFile(path.join(site, `guide/${name}.md`), `# ${name}\nUpdated: 2026-11-02\n[Resume](resume.md)\n[Research](../../../ai_research/roadmap.md)`);
    }
    const extra = await workspaceRecords(site, 'https://example.com/research/');
    const records = await readRecords(path.join(site, 'content'), '/custom/', extra);
    assert.equal(records.length, 4);
    assert.ok(records.every(record => record.updated === '2026-11-02'));
    assert.match(records[0].html, /href="\/custom\/records\/guides\/resume\/"/);
    assert.match(records[0].html, /href="https:\/\/example.com\/research\/"/);
    await fs.writeFile(path.join(site, 'guide/resume.md'), '# resume\nUpdated: 2026-11-03\nCurrent: P02');
    const changed = await readRecords(path.join(site, 'content'), '/', await workspaceRecords(site, 'https://example.com/research/'));
    assert.ok(changed.some(record => record.id === 'guides/resume' && record.updated === '2026-11-03' && record.html.includes('P02')));
    await fs.unlink(path.join(site, 'guide/progress.md'));
    await assert.rejects(workspaceRecords(site, 'https://example.com/research/'), /Missing canonical/);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
test('records and source catalog validate without opening the raw archive', async () => {
  const records = await readRecords(path.join(root, 'content'), '/project_docs/');
  assert.deepEqual(records.map(record => record.id).sort(),
    ['guides/cli-and-repositories', 'guides/roadmap-workflow']);
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
test('tutorial links connect validated Markdown pages, fragments, assets and source references', () => {
  for (const base of ['/', '/nested/']) {
    const { md } = renderer(new Set(), 'tutorials/p00-topic-v1/chapters/one.md', base, { links: {
      'chapters/two.md': 'records/tutorials/p00-topic-v1/chapters/two/',
      'whiteboard/defense.md': 'records/tutorials/p00-topic-v1/documents/whiteboard/defense/',
      'diagrams/one.svg': 'tutorial-assets/p00/topic/v1/diagrams/one.svg',
      'source:paper': 'https://example.com/paper',
    } });
    assert.ok(md.render('[Next](chapters/two.md#shapes)').includes(`href="${base}records/tutorials/p00-topic-v1/chapters/two/#shapes"`));
    assert.ok(md.render('[Defense](whiteboard/defense.md)').includes(`href="${base}records/tutorials/p00-topic-v1/documents/whiteboard/defense/"`));
    assert.ok(md.render('![Board](diagrams/one.svg)').includes(`src="${base}tutorial-assets/p00/topic/v1/diagrams/one.svg"`));
    assert.match(md.render('[Paper](source:paper#section)'), /href="https:\/\/example.com\/paper#section"/);
    assert.throws(() => md.render('[Missing](chapters/undeclared.md)'), /not in validated export/);
  }
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

test('a new one-lesson project uses its own start and reachable project area', async () => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'ai-new-course-'));
  try {
    const relative = 'projects/p99/weighted-mean/v1';
    const directory = path.join(temp, 'tutorials', relative);
    await fs.mkdir(path.join(directory, 'lessons'), { recursive: true });
    await fs.mkdir(path.join(directory, 'project'));
    const data = {
      project_id: 'P99', slug: 'weighted-mean', revision: 'v1', title: 'Combine measurements',
      question: 'How do unequal counts affect a mean?', sources: [], examples: [],
      objectives: [{ id: 'mean', text: 'Calculate a weighted mean.' }],
      chapters: [{ id: 'start', title: 'Lesson 1 — combine measurements', content_path: 'lessons/01-start.md', diagram_ids: [], table_ids: [], scenario_ids: [], practice_ids: [] }],
      diagrams: [], tables: [], scenarios: [], practice: [],
    };
    const files = {
      'lesson.json': JSON.stringify(data),
      'lesson.md': '# Combine measurements\n\nThe complete worked explanation.',
      'lessons/01-start.md': '## Lesson 1 — combine measurements\n\n### Count observations\n\nUse the count of observations in the denominator.\n\n[Implementation](project/implementation.md) · [Readiness](project/readiness.md)',
      'project/implementation.md': '# Implementation checklist\n\n[Begin](lessons/01-start.md)',
      'project/readiness.md': '# Project readiness\n\nExplain the denominator; learner understanding is unassessed.',
    };
    const hashes = {};
    for (const [file, content] of Object.entries(files)) {
      await fs.writeFile(path.join(directory, file), content);
      hashes[file] = createHash('sha256').update(content).digest('hex');
    }
    await fs.writeFile(path.join(directory, 'render.json'), JSON.stringify({ project_id: 'P99', slug: 'weighted-mean', revision: 'v1', validated_at: '2026-10-09T00:00:00Z', files: hashes }));
    await fs.writeFile(path.join(temp, 'tutorials/index.json'), JSON.stringify({ schema_version: 1, languages: ['cpp', 'python'], projects: [{ id: 'P99', status: 'author-checked', lessons: [relative] }] }));
    const tutorial = await tutorialRecords(temp);
    for (const base of ['/', '/nested/']) {
      const records = await readRecords(path.join(temp, 'content'), base, tutorial.records);
      const overview = records.find(record => record.tutorial.kind === 'overview');
      const start = records.find(record => record.tutorial.kind === 'chapter');
      assert.equal(overview.tutorial.projectArea, true);
      assert.deepEqual(overview.tutorial.courseFiles, [start.file]);
      assert.match(overview.html, /Begin Lesson 1 — combine measurements/);
      assert.doesNotMatch(overview.html, /make one prediction|Setup comes in Lesson 2/);
      assert.ok(overview.html.includes(start.url));
      const html = recordPage(start, base);
      assert.ok(html.includes(`rel="prev" href="${overview.url}"`));
      assert.match(html, /Project 99 · 1 lesson<\/summary>/);
      assert.doesNotMatch(html, /rel="next"|SUPPORTING MATERIAL|Project 0|PRACTICE AND REVIEW/);
      for (const sourcePath of ['project/implementation.md', 'project/readiness.md']) {
        const projectRecord = records.find(record => record.tutorial.sourcePath === sourcePath);
        assert.ok(projectRecord && overview.html.includes(projectRecord.url) && start.html.includes(projectRecord.url));
      }
    }
    assert.equal(tutorial.redirects.length, 0);
  } finally { await fs.rm(temp, { recursive: true, force: true }); }
});
