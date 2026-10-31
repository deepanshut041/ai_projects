import { p00LegacyRoutes } from './p00-layout.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { createReadStream } from 'node:fs';

const hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const fileHash = async file => {
  const digest = crypto.createHash('sha256');
  for await (const chunk of createReadStream(file)) digest.update(chunk);
  return digest.digest('hex');
};

// A course may reference explicitly selected finished movies in ignored storage.
// Small companions remain ordinary frozen course assets. No run directory is exposed.
export async function localTutorialVideos(directory, frozen, workspace, local) {
  if (!frozen.files['local-videos.json']) return { links: {}, files: {}, sources: [] };
  const manifest = JSON.parse((await artifact(directory, 'local-videos.json')).toString());
  if (manifest.schema_version !== 1 || !Array.isArray(manifest.videos)) throw new Error('Invalid tutorial movie manifest');
  const links = {}, files = {}, sources = [];
  const mediaRoot = path.join(workspace, '.ai/videos/projects');
  for (const movie of manifest.videos) {
    if (!/^media\/[a-z0-9-]+\/video\.mp4$/.test(movie.src) || sources.includes(movie.src)
      || !/^[a-z0-9-]+$/.test(movie.package) || !/^out\/(?:[a-z0-9-]+\/)*video\.mp4$/.test(movie.file)
      || !/^[a-f0-9]{64}$/.test(movie.sha256)) throw new Error('Unsafe tutorial movie reference');
    sources.push(movie.src);
    links[movie.src] = null;
    if (!local) continue;
    const file = path.join(mediaRoot, movie.package, movie.file);
    if (await fs.realpath(file).catch(() => null) !== file || await fileHash(file) !== movie.sha256) throw new Error(`Tutorial movie missing or changed: ${movie.package}`);
    const id = hash(Buffer.from(path.relative(workspace, file))).slice(0, 24);
    const name = `${id}/video.mp4`;
    files[name] = file;
    links[movie.src] = `local-files/${name}`;
  }
  return { links, files, sources };
}

function tableText(table) {
  const row = cells => '| ' + cells.map(value => value.replaceAll('|', '\\|').replaceAll('\n', ' ')).join(' | ') + ' |';
  return `### ${table.title}\n\n${[row(table.columns), row(table.columns.map(() => '---')), ...table.rows.map(item => row(item.cells))].join('\n')}\n\n`;
}

// Compose each authored file using the same insertion contract as learn render.
function chapterBody(data, chapter, raw) {
  const lookup = new Map();
  for (const diagram of data.diagrams) lookup.set(`diagram:${diagram.id}`, `![${diagram.title}](diagrams/${diagram.id}.svg)\n\n${diagram.description}\n\n`);
  for (const table of data.tables) lookup.set(`table:${table.id}`, tableText(table));
  for (const scenario of data.scenarios) lookup.set(`scenario:${scenario.id}`, `${scenario.description}\n\nIllustrative cases, not a model execution.\n\n[[tutorial-scenario:${scenario.id}]]\n\n` + tableText({
    title: scenario.title, columns: [...scenario.variables.map(variable => variable.label), 'Outcome', 'Explanation'],
    rows: scenario.cases.map(item => ({ cells: [...scenario.variables.map(variable => item.when[variable.id]), item.outcome, item.narrative] })),
  }));
  for (const item of data.practice) lookup.set(`practice:${item.id}`, `[[tutorial-practice:${item.id}]]\n\n`);
  const used = new Set();
  const insert = key => {
    if (!lookup.has(key)) throw new Error(`Unknown tutorial insertion: ${key}`);
    used.add(key);
    return lookup.get(key);
  };
  let body = raw.replace(/\[\[([a-z]+:[a-z0-9-]+)\]\]/g, (match, key) => insert(key));
  for (const prefix of ['diagram', 'table', 'scenario', 'practice']) {
    for (const id of chapter[prefix + '_ids']) {
      const key = `${prefix}:${id}`;
      if (!used.has(key)) body += '\n\n' + insert(key);
    }
  }
  return body;
}

function completeBody(body, data) {
  return body.replace(/### Predict or explain: ([^\n]+)\n\n<details(?: data-practice-id="([a-z0-9-]+)")?><summary>Reveal worked answer and rubric<\/summary>\n\n[\s\S]*?<\/details>/g, (match, question, id) => {
    const matches = data.practice.filter(item => id ? item.id === id : item.question === question);
    if (matches.length !== 1) throw new Error('Unmapped tutorial practice');
    return `[[tutorial-practice:${matches[0].id}]]`;
  }).replace(/\[\[interactive:([a-z0-9-]+)\]\]/g, '[[tutorial-scenario:$1]]');
}

const withoutTitle = body => body.replace(/^#{1,2} [^\n]*\n(?:\n)?/, '');
const titleOf = (body, fallback) => body.match(/^#{1,2} (.+)$/m)?.[1].replace(/^\d+\.\s*/, '') ?? fallback;

export async function artifact(root,relative) {
  if (path.isAbsolute(relative) || relative.includes('\\') || relative.split('/').some(part=>['.','..',''].includes(part))) throw new Error('Unsafe tutorial artifact path');
  const target=path.resolve(root,relative);
  if (!target.startsWith(path.resolve(root)+path.sep)) throw new Error('Tutorial artifact escapes package');
  for (let current=target;current!==path.resolve(root);current=path.dirname(current)) {
    if ((await fs.lstat(current)).isSymbolicLink()) throw new Error('Tutorial symlinks are not self-contained');
  }
  const raw=await fs.readFile(target);
  if (raw.length>10*1024*1024) throw new Error('Tutorial artifact exceeds 10 MiB');
  return raw;
}

export async function tutorialRecords(siteRoot, { local = false } = {}) {
  const root=path.join(siteRoot,'tutorials');
  let index;
  try { index=JSON.parse(await fs.readFile(path.join(root,'index.json'),'utf8')); }
  catch(error) { if(error.code==='ENOENT') return {records:[],assets:[],index:null};throw error; }
  if (index.schema_version!==1 || !Array.isArray(index.projects) || index.languages?.join(',')!=='cpp,python') throw new Error('Invalid tutorial index/language policy');
  const sequence = index.course_sequence ?? [];
  const courseNumberStart = index.course_number_start ?? 1;
  if (![0, 1].includes(courseNumberStart)) throw new Error('Invalid tutorial course number start');
  const references = index.reference_lessons ?? [];
  const declared = index.projects.flatMap(project => project.lessons);
  if (![sequence, references].every(items => Array.isArray(items) && new Set(items).size === items.length && items.every(item => declared.includes(item))) || sequence.some(item => references.includes(item))) throw new Error('Invalid tutorial course sequence/reference list');
  const ids=new Set();const records=[];const assets=[];const redirects=[];const sourcePages={};const files={};
  for (const project of index.projects) {
    if (!/^(?:P\d{2}|[a-z][a-z0-9-]{0,79})$/.test(project.id) || ids.has(project.id.toLowerCase())) throw new Error('Invalid/duplicate tutorial project ID');
    ids.add(project.id.toLowerCase());
    for (const relative of project.lessons) {
      const raw=await artifact(root,relative+'/lesson.json');
      const data=JSON.parse(raw);
      const directory=path.join(root,relative);
      const frozen=JSON.parse(await artifact(directory,'render.json'));
      if (data.project_id!==project.id || frozen.project_id!==project.id || data.slug!==frozen.slug || data.revision!==frozen.revision) throw new Error('Tutorial index/manifest identity mismatch');
      if (!/^[a-z][a-z0-9-]*$/.test(data.slug) || !/^[a-z][a-z0-9-]*$/.test(data.revision)) throw new Error('Invalid tutorial slug/revision');
      const url=`tutorial-assets/${project.id.toLowerCase()}/${data.slug}/${data.revision}`;
      for (const [file,digest] of Object.entries(frozen.files)) {
        if (!/\.(?:md|json|txt|svg|png|jpe?g|webp|mmd|cpp|hpp|py|vtt)$/.test(file)) throw new Error('Unsupported tutorial artifact');
        const bytes=await artifact(directory,file);
        if (hash(bytes)!==digest) throw new Error(`Tutorial changed after rendering: ${relative}/${file}. Inspect and render it again.`);
        if(file.endsWith('.svg') && /<script|<foreignObject|\bon\w+\s*=|(?:href|src)\s*=|<!DOCTYPE|<!ENTITY/i.test(bytes.toString())) throw new Error('Tutorial SVG must be passive');
        assets.push({file:`${url}/${file}`,bytes});
      }
      if (frozen.files['lesson.json']!==hash(raw) || !frozen.files['lesson.md']) throw new Error('Tutorial render lacks its authoring manifest/lesson');
      const id = `tutorials/${project.id.toLowerCase()}-${data.slug}-${data.revision}`;
      const overviewFile = `${id}.md`;
      const pages = [{ file: overviewFile, title: 'Overview', kind: 'overview' }];
      const bodies = new Map();
      const pageSources = new Map();
      const chapterIds = new Set();
      if (!Array.isArray(data.chapters) || !data.chapters.length) throw new Error('Tutorial needs chapters');
      for (const chapter of data.chapters) {
        if (!/^[a-z][a-z0-9-]*$/.test(chapter.id) || chapterIds.has(chapter.id)) throw new Error('Invalid/duplicate tutorial chapter ID');
        chapterIds.add(chapter.id);
        if (!frozen.files[chapter.content_path] || pageSources.has(chapter.content_path)) throw new Error('Tutorial chapter is missing from validated export or reused');
        const rawChapter = (await artifact(directory, chapter.content_path)).toString();
        const file = `${id}/chapters/${chapter.id}.md`;
        pages.push({ file, title: titleOf(rawChapter, chapter.title), kind: 'chapter', number: chapterIds.size, sourcePath: chapter.content_path });
        pageSources.set(chapter.content_path, file);
        bodies.set(file, withoutTitle(chapterBody(data, chapter, rawChapter)));
      }
      const supports = Object.keys(frozen.files).filter(file => file.endsWith('.md') && file !== 'lesson.md' && !file.startsWith('media/') && !pageSources.has(file));
      supports.sort((a, b) => a === 'workbook.md' ? -1 : b === 'workbook.md' ? 1 : a.localeCompare(b));
      for (const source of supports) {
        if (!/^(?:[a-z0-9-]+\/)*[a-z0-9-]+\.md$/.test(source)) throw new Error('Invalid tutorial document path');
        const body = (await artifact(directory, source)).toString();
        const file = `${id}/documents/${source}`;
        pages.push({ file, title: titleOf(body, source.replace(/\.md$/, '')), kind: 'support', sourcePath: source });
        pageSources.set(source, file);
        bodies.set(file, withoutTitle(body));
      }
      const completeFile = `${id}/complete.md`;
      pages.push({ file: completeFile, title: 'Complete lesson', kind: 'complete' });
      pageSources.set('lesson.md', completeFile);
      bodies.set(completeFile, withoutTitle(completeBody((await artifact(directory, 'lesson.md')).toString(), data)));
      const links = Object.fromEntries(Object.keys(frozen.files).map(file => [file, pageSources.has(file) ? `records/${pageSources.get(file).replace(/\.md$/, '')}/` : `${url}/${file}`]));
      const movies = await localTutorialVideos(directory, frozen, path.resolve(siteRoot, '../..'), local);
      Object.assign(links, movies.links);
      Object.assign(files, movies.files);
      for (const source of data.sources) links[`source:${source.id}`] = source.kind === 'external' ? source.url : links[source.excerpt_path];
      const buildPages = pages.filter(page => page.sourcePath?.startsWith('build/'));
      // Integrated courses keep their teaching and implementation in lessons/.
      // Retain the older chapter/build route for packages using that layout.
      const projectArea = data.chapters.every(chapter => chapter.content_path.startsWith('lessons/'));
      const coursePages = projectArea ? pages.filter(page => page.kind === 'chapter') : buildPages.length ? [pages.find(page => page.kind === 'chapter'), ...buildPages] : [];
      const courseFiles = coursePages.map(page => page.file);
      const foundations = project.id === 'P00' && data.slug === 'transformer-foundations' && data.revision === 'v1';
      const overviewBody = coursePages.length
        ? `## Begin with Lesson 1

${foundations ? 'Follow the topic outline from mathematical foundations through encoding, embeddings, attention, core sublayers and training. Each lesson contains its essential explanation, worked calculations, checks and personal notes prompts. Use Continue to follow the prerequisite order.' : 'Start with the first lesson below. Work through its examples and checks, then use Continue when another lesson follows. The project records help you resume implementation and review the evidence needed to finish.'}

**[Begin ${/^Lesson 1\b/i.test(coursePages[0].title) ? coursePages[0].title : 'Lesson 1 — ' + coursePages[0].title}](/records/${coursePages[0].file.replace(/\.md$/, '')}/)**

## The learning path

${coursePages.map(page => `- [${page.title}](/records/${page.file.replace(/\.md$/, '')}/)`).join('\n')}

A lesson can take several sessions. Stop after a small check and return to that part. Each lesson includes the concepts, derivations, worked examples, file edits and checks in the order you use them.

## Project work\n\n${pages.filter(page => page.kind === 'support').map(page => `- [${page.title}](/records/${page.file.replace(/\.md$/, '')}/)`).join('\n')}\n\n## Code and experiments\n\n${data.examples.map(example => `- [${example.path.split('/').pop()}](/${url}/${example.path}): ${example.purpose}`).join('\n')}\n\n## What you are building

${data.objectives.map(item => `- ${item.text}`).join('\n')}
`
        : `Learning material for ${project.id}. Model implementation/training and learner mastery are separate states.

## Learning goals

${data.objectives.map(item => `- ${item.text}`).join('\n')}

## Read by chapter

${pages.filter(page => page.kind === 'chapter').map(page => `${page.number}. [${page.title}](/records/${page.file.replace(/\.md$/, '')}/)`).join('\n')}

## Supporting material

${pages.filter(page => page.kind === 'support').map(page => `- [${page.title}](/records/${page.file.replace(/\.md$/, '')}/)`).join('\n')}
`;
      bodies.set(overviewFile, overviewBody + `
## Complete lesson

[Read or print all lessons](/records/${completeFile.replace(/\.md$/, '')}/) · [Download complete Markdown](/${url}/lesson.md)
`);
      for (const [source, file] of pageSources) sourcePages[`tutorials/${relative}/${source}`] = file;
      const legacyRoutes = projectArea && data.revision === 'v1'
        ? project.id === 'P00' && data.slug === 'transformer-foundations' ? p00LegacyRoutes
          : project.id === 'attention-labs' && data.slug === 'attention-heads'
            ? { 'chapters/layout': 'chapters/heads', 'chapters/boundary': 'chapters/heads' } : {}
        : {};
      for (const [from, to] of Object.entries(legacyRoutes)) {
        const target = `${id}/${to}.md`;
        if (!pages.some(page => page.file === target)) throw new Error('Missing consolidated tutorial redirect target');
        redirects.push({ from: `records/${id}/${from}/`, to: `records/${id}/${to}/` });
      }
      for (const [position, page] of pages.entries()) {
        const courseNumber = sequence.includes(relative) ? sequence.indexOf(relative) + courseNumberStart : null;
        const courseRole = references.includes(relative) ? 'reference' : 'course';
        const catalogOrder = courseNumber ?? (sequence.length + declared.indexOf(relative) + 1);
        const metadata = { practice: data.practice, scenarios: data.scenarios, project_id: project.id, projectTitle: project.title, courseNumber, courseRole, lessonTitle: data.title, overviewFile, pages, courseFiles, projectArea, position, kind: page.kind, chapterCount: data.chapters.length, links, localVideos: movies.sources, sourcePath: [...pageSources].find(([, file]) => file === page.file)?.[0] ?? 'lesson.json', assetBase: url };
        const fields = { title: page.kind === 'overview' ? data.title : page.kind === 'complete' ? `${data.title} — complete lesson` : page.title, description: data.question, updated: frozen.validated_at.slice(0,10), status: project.status.replaceAll('-',' ')+'; model progress separate', tags: ['tutorials', project.id, 'cpp', 'python'], sources: data.sources.filter(source => source.kind === 'external').map(source => ({ title: source.title, url: source.url })), tutorial: metadata };
        const frontmatter = Object.entries(fields).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n');
        if (records.some(record => record.file === page.file)) throw new Error('Duplicate tutorial page');
        records.push({ file: page.file, catalogOrder, raw: `---\n${frontmatter}\n---\n\n${bodies.get(page.file)}` });
      }
    }
  }
  records.sort((a, b) => a.catalogOrder - b.catalogOrder);
  return {records,assets,index,redirects,sourcePages,files};
}
