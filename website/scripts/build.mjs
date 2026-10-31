import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build as bundle } from 'esbuild';
import config from '../site.config.mjs';
import { readRecords, normalizeBase, validateCatalog, escape as e, categories } from './content.mjs';
import { shell, home, recordBrowser, recordPage, recordCard, sourcePage, heading, guide } from './templates.mjs';
import { workspaceRecords } from './workspace.mjs';
import { tutorialRecords } from './tutorials.mjs';
import { projectLibrary, projectsHome, projectPage, collectionCards, projectsSidebar } from './projects.mjs';
import { videoLibrary, videoHome } from './videos.mjs';
import { tutorialCollectionLibrary, tutorialCollectionCards, tutorialCollectionsHome, tutorialCollectionPage, tutorialsSidebar } from './tutorial-collections.mjs';

export const root = fileURLToPath(new URL('../', import.meta.url));
export const docsRoot = fileURLToPath(new URL('../../', import.meta.url));
export async function build({ local = false, directory = null, videoCatalog = null } = {}) {
  const base = normalizeBase(config.base);
  const tutorials=await tutorialRecords(docsRoot, { local });
  const records = await readRecords(path.join(root, 'content'), base, [...await workspaceRecords(docsRoot, config.companionUrl, tutorials.sourcePages),...tutorials.records]);
  const listed = records.filter(record => !record.tutorial || record.tutorial.kind === 'overview');
  for (const record of listed.filter(record => record.tutorial)) {
    record.search += ' ' + records.filter(page => page.tutorial?.overviewFile === record.file).map(page => page.search).join(' ');
  }
  const catalog = validateCatalog(JSON.parse(await fs.readFile(path.join(root, 'catalog/sources.json'), 'utf8')));
  const projectLinks = new Map(Object.entries(tutorials.sourcePages).map(([source, file]) => [path.join(docsRoot, source), `${base}records/${file.replace(/\.md$/, '')}/`]));
  for (const name of ['roadmap', 'progress', 'resume', 'decisions', 'COLAB', 'getting-started', 'attention-labs-route']) projectLinks.set(path.join(docsRoot, 'guide', name + '.md'), `${base}records/guides/${name === 'COLAB' ? 'colab' : name === 'getting-started' ? 'start-here' : name}/`);
  const library = await projectLibrary(path.resolve(docsRoot, '../projects'), base, { local, links: projectLinks });
  const tutorialLibrary = tutorialCollectionLibrary(records, library.groups, base);
  const videoRoutes = new Map(projectLinks);
  for (const page of library.pages) videoRoutes.set(page.source, page.url);
  for (const record of records) videoRoutes.set(path.join(root, 'content', record.file), record.url);
  const videos = local ? await videoLibrary({ workspace: path.resolve(docsRoot, '../..'), site: 'project-docs', base,
    routes: videoRoutes, pages: [...records, ...library.pages], catalogData: videoCatalog }) : null;
  if (videos) library.watches.push(...videos.watches.map(folder => ({ folder, recursive: true })));
  const destination = directory ?? path.join(root, local ? '.local-dist' : 'dist');
  const out = path.join(root, '.cache', `${local ? 'local' : 'public'}-site-${process.pid}`);
  await fs.rm(out, { recursive: true, force: true });
  await fs.mkdir(out, { recursive: true });
  const write = async (file, body) => { const dest = path.join(out, file); await fs.mkdir(path.dirname(dest), { recursive: true }); await fs.writeFile(dest, body); };
  await fs.cp(path.join(root, 'public'), out, { recursive: true });
  for(const asset of tutorials.assets) await write(asset.file,asset.bytes);
  if(tutorials.index) await write('tutorials/index.json',JSON.stringify(tutorials.index));
  await bundle({ entryPoints: [path.join(root, 'src/app.js')], outdir: path.join(out, 'assets'), bundle: true, splitting: true, format: 'esm', minify: true, loader: { '.woff': 'file', '.woff2': 'file', '.ttf': 'file' }, logLevel: 'warning' });
  const wrap = (title, body, active = '') => shell({ config, records: listed, title, body, active, projectNavigation: projectsSidebar(library, base, active), tutorialNavigation: tutorialsSidebar(tutorialLibrary, active), local });
  if (videos) {
    await write('videos/index.html', wrap('Videos', videoHome(videos.items, base), 'videos'));
    await write('videos/index.json', JSON.stringify({ schema_version: 1, bindings: videos.bindings }));
    for (const page of videos.pages) await write(page.url.slice(base.length) + 'index.html', wrap(page.title, page.body, 'videos'));
  }
  const section = (id, title, description, url, cards) => `<section id="${id}" class="${id === 'project-docs' ? 'project-collections' : 'tutorial-collections'} landing-section" aria-labelledby="${id}-heading"><div class="section-heading"><div><h2 id="${id}-heading">${title}</h2><p class="section-description">${description}</p></div><a class="text-link" href="${url}">All collections →</a></div>${cards}</section>`;
  const projectSection = section('project-docs', 'Project Docs', 'Project goals, setup, phase overviews and implementation records.', `${base}projects/`, collectionCards(library.groups));
  const tutorialSection = section('tutorials', 'Tutorials', 'Guided lessons, worked maths, source visuals and runnable examples.', `${base}tutorials/`, tutorialCollectionCards(tutorialLibrary.groups));
  await write('index.html', wrap('Project docs and tutorials', home(listed, catalog, base, projectSection, tutorialSection)));
  await write('projects/index.html', wrap('Project Docs', projectsHome(library), 'projects'));
  await write('search/index.html', wrap('Search', recordBrowser(listed), 'search'));
  await write('tutorials/index.html', wrap('Tutorials', tutorialCollectionsHome(tutorialLibrary), 'tutorials'));
  for (const group of tutorialLibrary.groups) await write(`tutorials/${group.id}/index.html`, wrap(`${group.title} tutorials`, tutorialCollectionPage(group, base), `tutorials/${group.id}`));
  await write('tutorials/collections.json', JSON.stringify(tutorialLibrary.groups.map(group => ({ id: group.id, title: group.title, url: group.url, courses: group.courses.map(record => ({ title: record.title, url: record.url, project_id: record.tutorial.project_id, course_number: record.tutorial.courseNumber })) }))));
  for (const page of library.pages) {
    const relative = decodeURIComponent(page.url.slice(base.length));
    await write(relative + 'index.html', wrap(page.title, projectPage(page, base), `projects/${page.group.id}`));
    const source = page.url.slice((base + 'projects/').length);
    await write(`project-source/${decodeURIComponent(source)}document.md`, page.raw);
  }
  await write('projects/index.json', JSON.stringify(library.groups.map(group => ({ id: group.id, title: group.title, url: group.url, projects: group.projects.map(project => ({ id: project.id, title: project.title, url: project.url, pages: project.pages.map(page => ({ title: page.title, url: page.url, file: page.file })) })) }))));
  if (local) await write('local-files.json', JSON.stringify({ ...library.files, ...tutorials.files, ...videos.files }));
  for (const category of categories) {
    if (category.id === 'tutorials') continue;
    const chosen = listed.filter(record => record.category === category.id);
    const courses = chosen.filter(record => record.tutorial?.courseRole !== 'reference');
    const references = chosen.filter(record => record.tutorial?.courseRole === 'reference');
    const referenceSection = references.length ? `<section class="tutorial-references"><div class="section-heading"><div><h2>Additional foundations exercises</h2><p>Earlier arithmetic, setup and tensor exercises remain available for review.</p></div></div><div class="record-grid">${references.map(recordCard).join('')}</div></section>` : '';
    await write(`${category.id}/index.html`, wrap(category.title, `${heading(category.title, category.description)}<div class="record-grid">${courses.map(recordCard).join('')}</div>${referenceSection}${chosen.length ? '' : `<div class="empty"><span class="empty-symbol">↗</span><h2>A place for the next ${category.id === 'papers' ? 'paper review' : category.id === 'experiments' ? 'experiment' : 'question'}</h2><p>No ${e(category.title.toLowerCase())} have been added yet.</p><a class="button secondary" href="${base}guide/">Read the authoring guide</a></div>`}`, category.id));
  }
  for (const record of records) {
    await write(`records/${record.id}/index.html`, wrap(record.title, recordPage(record, base), record.tutorial?.collection ? `tutorials/${record.tutorial.collection.id}` : record.category));
    await write(`source/${record.file}`, record.raw);
  }
  for (const redirect of tutorials.redirects ?? []) {
    const target = base + redirect.to;
    await write(redirect.from + 'index.html', `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=${e(target)}"><link rel="canonical" href="${e(target)}"><title>Lesson moved</title></head><body><a href="${e(target)}">Continue to the consolidated lesson</a></body></html>`);
  }
  await write('sources/index.html', wrap('Source register', sourcePage(catalog, base), 'sources'));
  await write('guide/index.html', wrap('Writing guide', guide(base), 'guide'));
  await write('404.html', wrap('Page not found', `${heading('This record is not here', 'Browse the Projects workbench to find a record or source.')}<a class="button" href="${base}">Open the workbench</a>`));
  await write('catalog/sources.json', JSON.stringify(catalog));
  await write('search.json', JSON.stringify(records.map(({ html, raw, ...record }) => record)));
  await write('build.json', JSON.stringify({ base, records: records.map(r => ({ id: r.id, url: r.url })), sources: catalog.sources.length }));
  await write('.nojekyll', '');
  const previous = out + '-previous';
  await fs.rm(previous, { recursive: true, force: true });
  try { await fs.rename(destination, previous); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await fs.mkdir(path.dirname(destination), { recursive: true });
  try { await fs.rename(out, destination); }
  catch (error) { await fs.rename(previous, destination).catch(() => {}); throw error; }
  await fs.rm(previous, { recursive: true, force: true });
  console.log(`Built ${records.length} project records and ${catalog.sources.length} source references at ${base}.`);
  return { out: destination, library, tutorialLibrary };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await build();
