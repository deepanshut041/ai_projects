import fs from 'node:fs/promises';
import path from 'node:path';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import anchor from 'markdown-it-anchor';
import { katex } from '@mdit/plugin-katex';
import { highlightCode } from './syntax-highlight.mjs';

export const categories = [
  { id: 'implementations', title: 'Implementations', description: 'Runnable local projects, contracts, tests and scope.' },
  { id: 'guides', title: 'Workspace guides', description: 'Use and extend the CLI, repositories and reproducible workflows.' },
  { id: 'tutorials', title: 'Project tutorials', description: 'Learn each model with diagrams, C++/Python examples, worked cases and whiteboard practice.' },
  { id: 'experiments', title: 'Learning experiments', description: 'Inspect known-model behavior; hand focused research questions to Research.' },
  { id: 'questions', title: 'Project questions', description: 'Next steps and implementation decisions requiring evidence.' },
];

export const escape = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
export function videoDefinition(text) {
  let data;
  try { data = JSON.parse(text); } catch { throw new Error('Video blocks require a JSON object.'); }
  if (!data || Array.isArray(data) || typeof data !== 'object') throw new Error('Invalid video block.');
  const fields = ['title', 'src', 'poster', 'captions', 'transcript'];
  if (fields.some(field => typeof data[field] !== 'string' || !data[field].trim()) || typeof data.duration !== 'number' || !Number.isFinite(data.duration) || data.duration <= 0) throw new Error('A video needs a title, duration, movie, poster, captions and transcript.');
  if (Object.keys(data).some(field => ![...fields, 'duration'].includes(field)) || !/\.mp4$/i.test(data.src) || !/\.vtt$/i.test(data.captions) || !/\.(?:jpg|jpeg|png|webp)$/i.test(data.poster) || !/\.(?:md|txt)$/i.test(data.transcript)) throw new Error('Unsupported video field or media format.');
  for (const field of fields.slice(1)) if (/^(?:[a-z]+:|\/|\\)/i.test(data[field])) throw new Error('Use relative, local media paths in video blocks.');
  return data;
}
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
export function renderer(recordFiles, relativeFile, base, tutorial = null, options = {}) {
  const headings = [];
  const md = new MarkdownIt({ html: false, linkify: true, typographer: true, highlight: highlightCode })
    .use(anchor, { level: [2, 3], callback: (token, info) => headings.push({ level: Number(token.tag.slice(1)), ...info }) })
    .use(katex, { throwOnError: true, trust: false, macros: { '\\ket': '\\left|#1\\right\\rangle', '\\bra': '\\left\\langle#1\\right|' } });
  // The reader supplies the chapter h1 after removing the source's h2 title.
  // Promote its remaining sections before anchors are collected; keep slugs.
  if (tutorial?.kind === 'chapter') md.core.ruler.before('anchor', 'chapter-heading-levels', state => {
    for (const token of state.tokens) {
      if (/^heading_(?:open|close)$/.test(token.type) && /^h[3-6]$/.test(token.tag)) {
        token.tag = `h${Number(token.tag.slice(1)) - 1}`;
      }
    }
  });
  md.renderer.rules.table_open = () => '<div class="markdown-table" tabindex="0" role="region" aria-label="Table; scroll horizontally if needed"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';
  const fence = md.renderer.rules.fence;
  md.renderer.rules.fence = (...args) => {
    const token = args[0][args[1]];
    if (token.info.trim() === 'video' && (options.resolveLink || tutorial?.links)) {
      const data = videoDefinition(token.content);
      const resolve = options.resolveLink ?? (href => tutorial.links[href] ? base + tutorial.links[href] : null);
      const urls = Object.fromEntries(['src', 'poster', 'captions', 'transcript'].map(field => [field, resolve(data[field])]));
      if (['poster', 'captions', 'transcript'].some(field => !urls[field]) || (!urls.src && !tutorial?.localVideos?.includes(data.src))) throw new Error(`Missing or disallowed media for video: ${data.title}`);
      const seconds = Math.round(data.duration);
      const duration = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
      if (!urls.src) return `<figure class="section-video"><figcaption><strong>${escape(data.title)}</strong><span>${duration}</span></figcaption><img src="${escape(urls.poster)}" alt="${escape(data.title)}"><p>The video plays in the local workspace reader. <a href="${escape(urls.transcript)}">Read the complete transcript</a> · <a href="${escape(urls.captions)}" download>Captions (VTT)</a></p></figure>\n`;
      return `<figure class="section-video"><figcaption><strong>${escape(data.title)}</strong><span>${duration}</span></figcaption><video controls playsinline preload="none" poster="${escape(urls.poster)}" aria-label="${escape(data.title)}"><source src="${escape(urls.src)}" type="video/mp4"><track kind="captions" src="${escape(urls.captions)}" srclang="en" label="English">Your browser cannot play this video. Use the download below.</video><details class="video-downloads"><summary>Transcript and downloads</summary><a href="${escape(urls.transcript)}" download>Transcript (Markdown)</a><a href="${escape(urls.src)}" download>Download video</a><a href="${escape(urls.captions)}" download>Captions (VTT)</a></details></figure>\n`;
    }
    return fence(...args).replace('<pre>', '<pre tabindex="0" role="region" aria-label="Code example; scroll horizontally if needed">');
  };
  if (tutorial) md.block.ruler.before('paragraph','tutorial-control',(state,startLine,endLine,silent)=>{
    const line=state.src.slice(state.bMarks[startLine]+state.tShift[startLine],state.eMarks[startLine]).trim();
    const match=line.match(/^\[\[tutorial-(practice|scenario):([a-z0-9-]+)\]\]$/);
    if(!match) return false;
    if(silent) return true;
    const token=state.push('html_block','',0);
    if(match[1]==='practice') {
      const item=tutorial.practice.find(item=>item.id===match[2]);
      if(!item) throw new Error('Unknown tutorial practice ID');
      const questionTag = tutorial.kind === 'chapter' ? 'h2' : 'h3';
      token.content=`<section class="tutorial-practice"><${questionTag} class="practice-question">${escape(item.question)}</${questionTag}><p>Try it yourself, then compare with the worked answer.</p><button type="button" class="practice-reveal" aria-expanded="false">Reveal worked answer and rubric</button><div class="practice-answer" hidden>${md.render(item.answer)}<p><strong>Check your reasoning:</strong> ${escape(item.rubric)}</p></div></section>`;
    } else {
      const item=tutorial.scenarios.find(item=>item.id===match[2]);
      if(!item) throw new Error('Unknown tutorial scenario ID');
      token.content=`<section class="tutorial-scenario" data-scenario="${escape(JSON.stringify(item))}"><h3>Explore: ${escape(item.title)}</h3><div class="scenario-inputs">${item.variables.map(variable=>`<label>${escape(variable.label)}<select data-variable="${escape(variable.id)}">${variable.options.map(option=>`<option>${escape(option)}</option>`).join('')}</select></label>`).join('')}</div><p><strong data-outcome></strong></p><p data-narrative></p></section>`;
    }
    state.line=startLine+1;return true;
  });
  md.core.ruler.after('inline', 'record-links', state => {
    for (const token of state.tokens) for (const child of token.children ?? []) {
      if (options.resolveLink && ['image', 'link_open'].includes(child.type)) {
        const attribute = child.type === 'image' ? 'src' : 'href';
        const href = child.attrGet(attribute);
        if (!/^(?:https?:|mailto:|#)/i.test(href)) {
          const target = options.resolveLink(href);
          if (target) child.attrSet(attribute, target);
          else {
            child.attrs = child.attrs.filter(([name]) => name !== attribute);
            child.attrSet('title', `Workspace file: ${href}`);
          }
          continue;
        }
      }
      if (tutorial?.links && ['image', 'link_open'].includes(child.type)) {
        const attribute = child.type === 'image' ? 'src' : 'href';
        const href = child.attrGet(attribute);
        if (!/^(?:https?:|mailto:|#|\/)/i.test(href)) {
          const split = href.indexOf('#');
          const file = split < 0 ? href : href.slice(0, split);
          const fragment = split < 0 ? '' : href.slice(split);
          if (!Object.hasOwn(tutorial.links, file) || !tutorial.links[file]) throw new Error(`Tutorial link not in validated export: ${href}`);
          const target = tutorial.links[file];
          child.attrSet(attribute, (/^https?:/i.test(target) ? target : base + target) + fragment);
          continue;
        }
      }
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
    if (!/^[a-z0-9-]+\/(?:[a-z0-9-]+\/)*[a-z0-9-]+\.md$/.test(file)) throw new Error(`Invalid record path: ${file}`);
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
