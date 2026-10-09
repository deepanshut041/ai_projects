import fs from 'node:fs/promises';
import path from 'node:path';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import anchor from 'markdown-it-anchor';
import { katex } from '@mdit/plugin-katex';

export const categories = [
  { id: 'implementations', title: 'Implementations', description: 'Runnable local projects, contracts, tests and scope.' },
  { id: 'guides', title: 'Workspace guides', description: 'Use and extend the CLI, repositories and reproducible workflows.' },
  { id: 'tutorials', title: 'Project tutorials', description: 'Learn each model with diagrams, C++/Python examples, worked cases and whiteboard practice.' },
  { id: 'experiments', title: 'Learning experiments', description: 'Inspect known-model behavior; hand focused research questions to Research.' },
  { id: 'questions', title: 'Project questions', description: 'Next steps and implementation decisions requiring evidence.' },
];

export const escape = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
export function normalizeBase(base) {
  if (!/^\/(?:[A-Za-z0-9_-]+\/)*$/.test(base)) throw new Error('SITE_BASE must be / or /repository/.');
  return base;
}
export function sourceURL(value) {
  const url = new URL(value);
  if (!['https:', 'http:'].includes(url.protocol)) throw new Error('Sources require an HTTP(S) URL.');
  return url.href;
}
export function validateCatalog(catalog) {
  if (catalog.schema !== 1 || !Array.isArray(catalog.collections) || !Array.isArray(catalog.sources)) throw new Error('Invalid source catalog.');
  const groups = new Set(catalog.collections.map(c => c.id));
  if (groups.size !== catalog.collections.length) throw new Error('Duplicate collection ID.');
  const ids = new Set();
  for (const source of catalog.sources) {
    if (ids.has(source.id) || !groups.has(source.collection) || !source.title || !source.kind) throw new Error(`Invalid source: ${source.id}`);
    ids.add(source.id); sourceURL(source.url);
  }
  return catalog;
}
export function renderer(recordFiles, relativeFile, base, tutorial = null) {
  const headings = [];
  const md = new MarkdownIt({ html: false, linkify: true, typographer: true })
    .use(anchor, { level: [2, 3], callback: (token, info) => headings.push({ level: Number(token.tag.slice(1)), ...info }) })
    .use(katex, { throwOnError: true, trust: false, macros: { '\\ket': '\\left|#1\\right\\rangle', '\\bra': '\\left\\langle#1\\right|' } });
  if (tutorial) md.block.ruler.before('paragraph','tutorial-control',(state,startLine,endLine,silent)=>{
    const line=state.src.slice(state.bMarks[startLine]+state.tShift[startLine],state.eMarks[startLine]).trim();
    const match=line.match(/^\[\[tutorial-(practice|scenario):([a-z0-9-]+)\]\]$/);
    if(!match) return false;
    if(silent) return true;
    const token=state.push('html_block','',0);
    if(match[1]==='practice') {
      const item=tutorial.practice.find(item=>item.id===match[2]);
      if(!item) throw new Error('Unknown tutorial practice ID');
      token.content=`<section class="tutorial-practice"><h3>${escape(item.question)}</h3><p>Attempt your explanation or drawing before revealing the worked answer.</p><button type="button" class="practice-reveal" aria-expanded="false">Reveal worked answer and rubric</button><div class="practice-answer" hidden>${md.render(item.answer)}<p><strong>Rubric:</strong> ${escape(item.rubric)}</p></div></section>`;
    } else {
      const item=tutorial.scenarios.find(item=>item.id===match[2]);
      if(!item) throw new Error('Unknown tutorial scenario ID');
      token.content=`<section class="tutorial-scenario" data-scenario="${escape(JSON.stringify(item))}"><h3>Explore: ${escape(item.title)}</h3><div class="scenario-inputs">${item.variables.map(variable=>`<label>${escape(variable.label)}<select data-variable="${escape(variable.id)}">${variable.options.map(option=>`<option>${escape(option)}</option>`).join('')}</select></label>`).join('')}</div><p><strong data-outcome></strong></p><p data-narrative></p></section>`;
    }
    state.line=startLine+1;return true;
  });
  md.core.ruler.after('inline', 'record-links', state => {
    for (const token of state.tokens) for (const child of token.children ?? []) {
      if (child.type === 'image') {
        const source=child.attrGet('src');
        if(source.startsWith('/'))child.attrSet('src',base+source.slice(1));
        continue;
      }
      if (child.type !== 'link_open') continue;
      const href = child.attrGet('href');
      if (/^(?:https?:|mailto:|#)/i.test(href)) continue;
      const [file, fragment] = href.split('#');
      if (file.startsWith('/tutorial-assets/')) child.attrSet('href',base+file.slice(1));
      else if (file.endsWith('.md')) {
        const target = path.posix.normalize(path.posix.join(path.posix.dirname(relativeFile), file));
        if (!recordFiles.has(target)) throw new Error(`Broken research link: ${href}`);
        child.attrSet('href', `${base}records/${target.replace(/\.md$/, '')}/${fragment ? '#' + fragment : ''}`);
      } else if (file.startsWith('/')) child.attrSet('href', base + file.slice(1) + (fragment ? '#' + fragment : ''));
      else throw new Error(`Unsupported local research link: ${href}`);
    }
  });
  if(tutorial) {
    const original=md.renderer.rules.image;
    md.renderer.rules.image=(tokens,index,options,env,self)=>{
      const token=tokens[index];const src=token.attrGet('src');
      return `<figure class="tutorial-board"><a href="${escape(src)}" target="_blank" rel="noopener">Open diagram at full size ↗</a><p class="diagram-hint">Scroll sideways to follow the complete diagram.</p><div class="diagram-scroll" tabindex="0" role="region" aria-label="${escape(token.content)}">${original(tokens,index,options,env,self)}</div></figure>`;
    };
  }
  return { md, headings };
}
export async function readRecords(root, base, additional = []) {
  const files = [];
  for (const category of categories) {
    const folder = path.join(root, category.id);
    try { for (const name of (await fs.readdir(folder)).sort()) if (name.endsWith('.md')) files.push(`${category.id}/${name}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  const extra = new Map(additional.map(record => [record.file, record.raw]));
  for (const file of extra.keys()) {
    if (files.includes(file)) throw new Error(`Duplicate canonical record: ${file}`);
    files.push(file);
  }
  const recordFiles = new Set(files);
  const records = [];
  for (const file of files) {
    if (!/^[a-z0-9-]+\/[a-z0-9-]+\.md$/.test(file)) throw new Error(`Invalid record path: ${file}`);
    const raw = extra.has(file) ? extra.get(file) : await fs.readFile(path.join(root, file), 'utf8');
    const { data, content } = matter(raw);
    for (const field of ['title', 'description', 'updated', 'status']) if (typeof data[field] !== 'string' || !data[field].trim()) throw new Error(`${file}: missing ${field}.`);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(data.updated) || Number.isNaN(Date.parse(data.updated)) || new Date(data.updated).toISOString().slice(0, 10) !== data.updated) throw new Error(`${file}: invalid date.`);
    if (data.tags && (!Array.isArray(data.tags) || data.tags.some(t => typeof t !== 'string'))) throw new Error(`${file}: tags must be strings.`);
    for (const source of data.sources ?? []) { if (!source.title) throw new Error(`${file}: source needs a title.`); sourceURL(source.url); }
    const { md, headings } = renderer(recordFiles, file, base, data.tutorial);
    const html = md.render(content);
    records.push({ ...data, tags: data.tags ?? [], sources: data.sources ?? [], category: file.split('/')[0], id: file.replace(/\.md$/, ''), file, raw, html, headings, url: `${base}records/${file.replace(/\.md$/, '')}/`, search: `${data.title} ${data.description} ${content}`.toLowerCase() });
  }
  return records;
}
