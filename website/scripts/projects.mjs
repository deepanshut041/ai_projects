import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import { renderer, escape as e, videoDefinition } from './content.mjs';
import { heading } from './templates.mjs';

export const collections = [
  { id: 'foundations', title: 'Foundations', description: 'Practical machine learning, vision and time series.' },
  { id: 'transformer-labs', title: 'Transformer Labs', description: 'Attention, embeddings and small encoder and decoder systems.' },
  { id: 'language-models', title: 'Language Models', description: 'Complete small language models, training and recovery.' },
];
const inside = (root, file) => file === root || file.startsWith(root + path.sep);
const slugTitle = value => value.replace(/^\d+-/, '').replaceAll('-', ' ').replace(/\b\w/g, c => c.toUpperCase());
const route = file => file.split(path.sep).map(encodeURIComponent).join('/');
const pageRoute = file => /(?:^|\/)README\.md$/i.test(file) ? file.replace(/README\.md$/i, '') : file.replace(/\.md$/i, '') + '/';
async function read(file) {
  try { return (await fs.lstat(file)).isFile() ? await fs.readFile(file, 'utf8') : null; }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}
async function markdownFiles(folder) {
  const files = [];
  try {
    if (!(await fs.lstat(folder)).isDirectory()) return files;
    for (const entry of (await fs.readdir(folder, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith('.') || ['node_modules', 'data', 'runs', 'dist', 'sources', '__pycache__'].includes(entry.name)) continue;
      if (entry.isDirectory() && /^(?:build(?:-|$)|output$|before-|supporting-files-before-)/.test(entry.name)) continue;
      const file = path.join(folder, entry.name);
      if (entry.isDirectory()) files.push(...await markdownFiles(file));
      else if (entry.isFile() && /\.md$/i.test(entry.name)) files.push(file);
    }
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  return files;
}
function document(raw, fallback) {
  const { content } = matter(raw);
  const title = content.match(/^#\s+(.+)$/m)?.[1] ?? fallback;
  const body = content.replace(/^#\s+[^\n]+\n?\s*/, '');
  const description = body.split(/\n\s*\n/).find(p => /^[A-Za-z]/.test(p) && !p.includes('|')) ?? '';
  const plain = description.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[*`]/g, '').replace(/\s+/g, ' ');
  return { title, body, description: plain.length > 220 ? plain.slice(0, 220).replace(/\s+\S*$/, '') + '…' : plain };
}

// Public builds contain READMEs. Private study pages exist only in the local
// output selected explicitly by the development server.
export async function projectLibrary(projectsRoot, base, { local = false, links = new Map() } = {}) {
  projectsRoot = path.resolve(projectsRoot);
  const workspace = path.resolve(projectsRoot, '../..');
  const library = { groups: [], pages: [], files: {}, watches: [{ folder: projectsRoot, recursive: false }] };
  const targets = new Map(links);
  for (const definition of collections) {
    const folder = path.join(projectsRoot, definition.id);
    const raw = await read(path.join(folder, 'README.md'));
    if (raw === null) continue;
    const group = { ...definition, root: folder, raw, url: `${base}projects/${definition.id}/`, projects: [] };
    library.watches.push({ folder, recursive: false });
    const overview = { ...document(raw, group.title), raw, source: path.join(folder, 'README.md'), file: 'README.md', url: group.url, group };
    group.overview = overview;
    library.groups.push(group); library.pages.push(overview); targets.set(overview.source, overview.url);
    for (const entry of await fs.readdir(folder, { withFileTypes: true })) {
      if (!entry.isDirectory() || !/^[a-z0-9][a-z0-9-]*$/.test(entry.name)) continue;
      const projectRoot = path.join(folder, entry.name);
      const projectRaw = await read(path.join(projectRoot, 'README.md'));
      if (projectRaw === null) continue;
      const project = { id: entry.name, root: projectRoot, group, url: group.url + entry.name + '/', pages: [], ...document(projectRaw, slugTitle(entry.name)) };
      group.projects.push(project);
      library.watches.push({ folder: projectRoot, recursive: false });
      if (local) {
        const docs = path.join(projectRoot, 'docs');
        if ((await fs.lstat(docs).catch(() => null))?.isDirectory()) library.watches.push({ folder: docs, recursive: true });
      }
      const sources = [path.join(projectRoot, 'README.md'), ...(local ? await markdownFiles(path.join(projectRoot, 'docs')) : [])];
      for (const source of sources) {
        const raw = await fs.readFile(source, 'utf8');
        const file = path.relative(projectRoot, source).split(path.sep).join('/');
        const page = { ...document(raw, slugTitle(path.basename(file, '.md'))), raw, source, file, url: project.url + route(pageRoute(file)), group, project };
        if (library.pages.some(other => other.url === page.url)) throw new Error(`Duplicate project document route: ${file}`);
        project.pages.push(page); library.pages.push(page); targets.set(source, page.url);
      }
      project.overview = project.pages[0];
    }
    const ordered = [...raw.matchAll(/^\d+\.\s+\[[^\]]+\]\(([^/]+)\/README\.md\)/gm)].map(match => match[1]);
    group.projects.sort((a, b) => {
      const position = p => { const index = ordered.length ? ordered.indexOf(p.id) : raw.indexOf(p.id + '/README.md'); return index < 0 ? Infinity : index; };
      return position(a) - position(b) || a.title.localeCompare(b.title);
    });
  }
  const fileURL = async absolute => {
    if (!local) return null;
    // Only a document's explicitly referenced source/media is exposed. The
    // manifest stays outside the public build and cannot be fetched itself.
    const project = library.groups.flatMap(g => g.projects).find(p => inside(p.root, absolute));
    const relative = project && path.relative(project.root, absolute);
    const source = project && (/^(?:src|tests)[/\\]/.test(relative) || !relative.includes(path.sep));
    const notes = project && inside(path.join(project.root, 'docs'), absolute);
    const video = inside(path.join(workspace, '.ai/videos'), absolute);
    if (!(source || notes || video) || !/\.(?:py|cpp|hpp|h|c|toml|json|txt|md|ipynb|png|jpg|jpeg|webp|svg|mp4|m4a|mp3|wav|srt|vtt)$/i.test(absolute)) return null;
    const real = await fs.realpath(absolute).catch(() => null);
    if (!real || real !== absolute || !(await fs.stat(real)).isFile()) return null;
    const id = createHash('sha256').update(path.relative(workspace, real)).digest('hex').slice(0, 24);
    const name = `${id}/${encodeURIComponent(path.basename(real))}`;
    library.files[name] = real;
    return `${base}local-files/${name}`;
  };
  for (const page of library.pages) {
    const resolved = new Map();
    // Parse Markdown tokens rather than matching link syntax; this also
    // handles reference links and filenames written inside angle brackets.
    const parser = new MarkdownIt({ html: false });
    const displayBody = page.body.replace(/^<!-- section-video:[a-z0-9-]+:(?:start|end) -->\n/gm, '');
    const tokens = parser.parse(displayBody, {});
    const references = [];
    for (const token of tokens) {
      if (token.type === 'fence' && token.info.trim() === 'video') {
        const video = videoDefinition(token.content);
        references.push(...['src', 'poster', 'captions', 'transcript'].map(field => video[field]));
      }
      for (const child of token.children ?? []) if (['link_open', 'image'].includes(child.type)) references.push(child.attrGet(child.type === 'image' ? 'src' : 'href'));
    }
    for (const href of references) {
      if (/^(?:https?:|mailto:|#)/i.test(href)) continue;
      const match = href.match(/^([^#?]*)(.*)$/);
      const absolute = path.resolve(path.dirname(page.source), decodeURIComponent(match[1]));
      let target = targets.get(absolute) ?? targets.get(path.join(absolute, 'README.md'));
      if (!target) target = await fileURL(absolute);
      resolved.set(href, target ? target + match[2] : null);
    }
    const { md, headings } = renderer(new Set(), '', base, null, { resolveLink: href => resolved.get(href) });
    page.html = md.render(displayBody); page.headings = headings;
  }
  return library;
}

export function collectionCards(groups) {
  return `<div class="collection-grid">${groups.map(group => `<a class="collection-card" href="${group.url}"><span class="eyebrow">${group.projects.length} ${group.projects.length === 1 ? 'project' : 'projects'}</span><h2>${e(group.title)} <span aria-hidden="true">↗</span></h2><p>${e(group.description)}</p></a>`).join('')}</div>`;
}
export function projectsHome(library) {
  return `${heading('Projects', 'Choose a collection, then open a project and its documentation.')}${collectionCards(library.groups)}`;
}
export function projectsSidebar(library, base) {
  return `<p class="sidebar-label">COLLECTIONS</p>${library.groups.map(group => `<a class="collection-nav" href="${group.url}"><span>${e(group.title)}</span><small>${group.projects.length}</small></a>`).join('')}`;
}
function breadcrumbs(page, base) {
  return `<nav class="breadcrumbs" aria-label="Breadcrumb"><a href="${base}projects/">Projects</a><span>/</span><a href="${page.group.url}">${e(page.group.title)}</a>${page.project ? `<span>/</span><a href="${page.project.url}">${e(page.project.title)}</a>` : ''}${page.project && page !== page.project.overview ? `<span>/</span><span>${e(page.title)}</span>` : ''}</nav>`;
}
function contents(project, active) {
  const root = { children: new Map(), pages: [] };
  for (const page of project.pages) {
    let node = root;
    for (const part of page.file.split('/').slice(0, -1)) {
      if (!node.children.has(part)) node.children.set(part, { children: new Map(), pages: [] });
      node = node.children.get(part);
    }
    node.pages.push(page);
  }
  const draw = (node, parts = []) => {
    const pages = node.pages.map(page => `<a href="${page.url}" ${page === active ? 'aria-current="page"' : ''}>${page.file.startsWith('docs/implementation/') && !/README\.md$/i.test(page.file) ? `<code>${e(path.basename(page.file))}</code><span class="document-title">${e(page.title)}</span>` : e(page === project.overview ? 'Project overview' : page.file.toLowerCase() === 'docs/readme.md' ? 'Documentation home' : page.title)}</a>`).join('');
    const children = [...node.children].sort(([a], [b]) => {
      if (parts.at(-1) === 'docs') {
        const priority = name => name === 'phases' ? 0 : name === 'implementation' ? 1 : 2;
        return priority(a) - priority(b) || a.localeCompare(b);
      }
      return a.localeCompare(b);
    });
    return pages + children.map(([name, child]) => {
      const folder = [...parts, name];
      if (name === 'docs') return draw(child, folder);
      const main = child.pages.find(page => /02-notes-to-understanding\.md$/i.test(page.file)) ?? child.pages.find(page => /(?:README|00-resume)\.md$/i.test(page.file));
      const phase = parts.at(-1) === 'phases' && child.pages.length === 1 && !child.children.size;
      if (phase) return draw(child, folder);
      const open = name === 'implementation' || active.file.startsWith(folder.join('/') + '/');
      const label = name === 'phases' ? 'Phases' : name === 'implementation' ? 'Implementation' : /^\d+_/.test(name) ? `${name.split('_')[0]}_ ${main?.title ?? slugTitle(name.replace(/^\d+_/, ''))}` : main?.title ?? slugTitle(name);
      return `<details${open ? ' open' : ''}><summary>${e(label)}</summary>${draw(child, folder)}</details>`;
    }).join('');
  };
  return `<aside class="project-contents"><details class="project-menu" open><summary>Project contents</summary><nav aria-label="Project contents">${draw(root)}</nav></details></aside>`;
}
function projectFolders(project) {
  const phases = project.pages.filter(page => /^docs\/phases\/[^/]+\/README\.md$/i.test(page.file));
  const implementation = new Map();
  for (const page of project.pages) {
    const match = page.file.match(/^docs\/implementation\/(\d{2,}_[^/]+)\/([^/]+)$/);
    if (!match) continue;
    if (!implementation.has(match[1])) implementation.set(match[1], []);
    implementation.get(match[1]).push(page);
  }
  const phaseIndex = project.pages.find(page => page.file.toLowerCase() === 'docs/phases/readme.md');
  const implementationIndex = project.pages.find(page => page.file.toLowerCase() === 'docs/implementation/readme.md');
  if (!phases.length && !implementation.size) return '';
  const link = page => `<a href="${page.url}">${e(page.title)}<span aria-hidden="true">→</span></a>`;
  return `<section class="project-folders" aria-label="Documentation folders">${phases.length ? `<details class="project-folder"><summary><h2>Phases</h2><small>${phases.length} phase overviews</small></summary><p>The project route and what each stage covers.</p>${phaseIndex ? `<a class="text-link" href="${phaseIndex.url}">Open phase overview →</a>` : ''}<div class="folder-items">${phases.map(link).join('')}</div></details>` : ''}${implementation.size ? `<details class="project-folder" open><summary><h2>Implementation</h2><small>Numbered working records</small></summary><p>Understanding notes, learning points and working records.</p>${implementationIndex ? `<a class="text-link" href="${implementationIndex.url}">Open implementation overview →</a>` : ''}${[...implementation].sort(([a], [b]) => a.localeCompare(b)).map(([folder, pages]) => {
    const main = pages.find(page => /02-notes-to-understanding\.md$/i.test(page.file)) ?? pages.find(page => /(?:README|00-resume)\.md$/i.test(page.file)) ?? pages[0];
    return `<details class="implementation-folder"><summary><code>${e(folder)}</code><span>${e(main.title)}</span><small>${pages.length} documents</small></summary><div class="folder-items">${pages.map(page => `<a href="${page.url}"><span><code>${e(path.basename(page.file))}</code><span class="document-title">${e(page.title)}</span></span><span aria-hidden="true">→</span></a>`).join('')}</div></details>`;
  }).join('')}</details>` : ''}</section>`;
}
export function projectPage(page, base) {
  const project = page.project;
  const files = project?.pages ?? [];
  const docs = files.find(p => p.file.toLowerCase() === 'docs/readme.md');
  const groupCards = !project ? `<div class="record-grid project-grid">${page.group.projects.map(p => `<a class="record-card project-card" href="${p.url}"><span class="eyebrow">${p.pages.length > 1 ? `${p.pages.length - 1} documentation pages` : 'Project overview'}</span><h2>${e(p.title)}</h2><p>${e(p.description)}</p><span class="text-link">Open project →</span></a>`).join('')}</div>${page.group.projects.length ? '' : '<p class="empty">No project implementations have been added to this collection yet. Its overview describes the planned builds.</p>'}<details class="collection-overview"><summary>Read the collection overview</summary><article class="markdown">${page.html}</article></details>` : '';
  if (!project) return `${breadcrumbs(page, base)}${heading(page.group.title, page.group.description, 'Project collection')}${groupCards}`;
  const toolbar = `<div class="record-toolbar">${docs && page === project.overview ? `<a class="button secondary" href="${docs.url}">Open project docs →</a>` : ''}<a href="${base}project-source/${page.url.slice((base + 'projects/').length)}document.md" download>Download Markdown ↓</a><button class="print-button" type="button">Print / Save PDF ↓</button></div>`;
  const folders = page === project.overview && files.length > 1 ? projectFolders(project) : '';
  const article = `<article class="markdown">${page.html}</article>`;
  const body = folders ? `${folders}<details class="collection-overview"><summary>Project overview and scope</summary>${article}</details>` : article;
  return `${breadcrumbs(page, base)}<div class="project-layout">${contents(project, page)}<div class="project-reading">${heading(page.title, '', page.group.title)}${toolbar}<details class="chapter-outline"><summary>On this page</summary><nav aria-label="Page sections">${page.headings.map(h => `<a href="#${e(h.slug)}">${e(h.title)}</a>`).join('')}</nav></details>${body}</div></div>`;
}
