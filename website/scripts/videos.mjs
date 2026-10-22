// Owner-local reader adapter. The CLI supplies validated data, never executable content.
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import matter from 'gray-matter';
import { renderer, escape as e } from './content.mjs';

const exec = promisify(execFile);
const duration = value => { const seconds = Math.round(value); return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`; };
const escapePattern = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export function player(item, base) {
  const urls = Object.fromEntries(Object.entries(item.media).map(([key, value]) => [key, base + value]));
  const language = /^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$/i.test(item.language) ? item.language : 'en';
  let label = language;
  try { label = new Intl.DisplayNames(['en'], { type: 'language' }).of(language); } catch { /* Keep the explicit language code. */ }
  return `<figure class="section-video" data-video-id="${e(item.video_id)}"><figcaption><strong>${e(item.title)}</strong><span>${duration(item.duration)}</span></figcaption><video controls playsinline preload="none" poster="${e(urls.poster)}" aria-label="${e(item.title)}"><source src="${e(urls.src)}" type="video/mp4"><track kind="captions" src="${e(urls.captions)}" srclang="${e(language)}" label="${e(label)}">Your browser cannot play this video. Use the download below.</video><p class="video-error" role="status" hidden>The video could not load. <a href="${e(urls.transcript)}">Read the transcript</a> or try the download below.</p><details class="video-downloads"><summary>Transcript and downloads</summary><a href="${e(urls.transcript)}" download>Transcript (Markdown)</a><a href="${e(urls.src)}" download>Download video</a><a href="${e(urls.captions)}" download>Captions (VTT)</a></details></figure>`;
}

function insert(page, item, base) {
  const heading = item.target.heading;
  const selected = heading && page.headings?.find(value => value.title === heading);
  const url = page.url + (selected ? '#' + encodeURIComponent(selected.slug) : '');
  // Legacy course/notes players can already expose this exact finished file.
  if (page.html.includes(`src="${e(base + item.media.src)}"`)) return url;
  const title = item.source?.text.match(/^#{1,6}[ \t]+(.+?)\s*$/m)?.[1];
  if (!heading || !selected && heading === title) { page.html = player(item, base) + page.html; return url; }
  if (selected) {
    const pattern = new RegExp(`(<h[1-6]\\b[^>]*\\bid="${escapePattern(e(selected.slug))}"[^>]*>[\\s\\S]*?<\\/h[1-6]>)`);
    if (pattern.test(page.html)) { page.html = page.html.replace(pattern, match => match + player(item, base)); return url; }
  }
  return null;
}

export async function videoLibrary({ workspace, site, base, routes = new Map(), pages = [], catalogData = null }) {
  let data = catalogData;
  if (!data) {
    const keys = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'XDG_CACHE_HOME', 'UV_CACHE_DIR'];
    const env = Object.fromEntries(keys.filter(key => process.env[key]).map(key => [key, process.env[key]]));
    const result = await exec(path.join(workspace, 'ai'), ['--root', workspace, 'video', 'catalog', '--site', site, '--json'], { cwd: workspace, env, timeout: 60000, maxBuffer: 64 * 1024 * 1024 });
    data = JSON.parse(result.stdout);
  }
  if (data.schema_version !== 1 || data.site !== site || !Array.isArray(data.items) || !data.files || typeof data.files !== 'object') throw new Error('Invalid Workbench video catalog');
  const output = { files: data.files, pages: [], bindings: [], items: [], watches: [path.join(workspace, '.ai/video-library'), ...(data.watches ?? [])] };
  const groups = new Map();
  for (const item of data.items) {
    if (!/^video-[a-f0-9]{24}$/.test(item.id) || !/^[a-z0-9-]+\/[a-z0-9-]+$/.test(item.video_id)) throw new Error('Invalid video identity');
    if (item.status !== 'ready') { output.items.push(item); continue; }
    if (!Number.isFinite(item.duration) || item.duration <= 0 || ['src', 'poster', 'captions', 'transcript'].some(key => !/^local-files\/[a-f0-9]{24}\/[a-zA-Z0-9._%~-]+$/.test(item.media?.[key]) || !data.files[item.media[key].slice('local-files/'.length)])) throw new Error('Invalid video artifact mapping');
    const videoURL = `${base}videos/${item.video_id}/`;
    const existingURL = item.source && routes.get(path.resolve(workspace, item.source.document));
    const page = pages.find(value => value.url === existingURL);
    let targetURL = page ? insert(page, item, base) : null;
    if (item.source) {
      output.watches.push(path.dirname(path.resolve(workspace, item.source.document)));
      const previewURL = `${videoURL}source/${item.id}/`;
      const content = matter(item.source.text).content.replace(/^# [^\n]+\n/, '')
        .replace(/^```video\s*\n[\s\S]*?^```/gm, 'An existing video companion is available in the original tutorial.\n');
      const { md, headings } = renderer(new Set(), '', base, null, { resolveLink: href => {
        const [file, fragment] = href.split('#');
        const url = routes.get(path.resolve(workspace, path.dirname(item.source.document), file));
        return url ? url + (fragment ? '#' + fragment : '') : null;
      } });
      const preview = { url: previewURL, html: md.render(content), headings };
      let previewTarget = insert(preview, item, base);
      if (!previewTarget) { preview.html = player(item, base) + preview.html; previewTarget = previewURL; }
      output.pages.push({ url: previewURL, title: item.title + ' — tutorial source', body: `<nav class="breadcrumbs" aria-label="Breadcrumb"><a href="${videoURL}">Video</a><span>/</span><span>Tutorial source</span></nav><header class="page-heading"><h1>Tutorial source</h1><p>${e(item.source.document)}</p></header><article class="markdown">${preview.html}</article>` });
      targetURL ??= previewTarget;
    }
    targetURL ??= videoURL;
    const normalized = { ...item, video_url: videoURL, target_url: targetURL };
    output.bindings.push({ id: item.id, video_id: item.video_id, video_url: videoURL, target_url: targetURL });
    if (!groups.has(item.video_id)) groups.set(item.video_id, []);
    groups.get(item.video_id).push(normalized);
    output.watches.push(path.join(workspace, '.ai/videos', item.video_id));
  }
  for (const [id, bindings] of groups) {
    const item = bindings[0];
    output.items.push(item);
    const transcript = item.media.transcript.slice('local-files/'.length);
    // Only a verified transcript is rendered; its relative local links are inert.
    const fs = await import('node:fs/promises');
    const text = await fs.readFile(data.files[transcript], 'utf8');
    const { md } = renderer(new Set(), '', base, null, { resolveLink: () => null });
    const sourceLinks = bindings.filter(value => value.source).map(value => `<li><a href="${e(value.target_url)}">${e(value.target.heading || value.source.document)}</a></li>`).join('');
    output.pages.push({ url: item.video_url, title: item.title, body: `<nav class="breadcrumbs" aria-label="Breadcrumb"><a href="${base}videos/">Videos</a></nav><header class="page-heading"><h1>${e(item.title)}</h1><p>${duration(item.duration)} · ${e(item.language || 'en')}</p></header>${player(item, base)}${sourceLinks ? `<section><h2>Read the tutorial</h2><ul>${sourceLinks}</ul></section>` : ''}<section class="markdown"><h2>Transcript</h2>${md.render(text)}</section>` });
  }
  return output;
}

export function videoHome(items, base) {
  return `<header class="page-heading"><h1>Videos</h1><p>Watch an explanation, then follow it in the written tutorial.</p></header><label class="search-label">Search videos<input id="video-search" type="search" placeholder="Title or tutorial…"></label><p id="video-status" class="result-status" aria-live="polite">${items.length} videos</p><div class="record-grid">${items.map(item => `<article class="record-card video-card" data-search="${e((item.title + ' ' + (item.source?.document || '') + ' ' + item.video_id).toLowerCase())}"><div class="card-meta"><span>${e(item.video_id.split('/')[0])}</span><span>${item.status === 'ready' ? duration(item.duration) : 'Needs attention'}</span></div><h2>${e(item.title)}</h2>${item.status === 'ready' ? `<p>${e(item.target?.heading || item.source?.document || 'Local video')}</p><div class="card-bottom"><a class="button" href="${e(item.video_url)}">Play video</a>${item.source ? `<a class="text-link" href="${e(item.target_url)}">Open tutorial</a>` : ''}</div>` : `<p role="status">${e(item.problem || 'Video unavailable. Inspect video audit in the CLI.')}</p>`}</article>`).join('')}</div><p id="video-empty" class="empty" ${items.length ? 'hidden' : ''}>No videos match. Add a verified video with the tutorial-video skill or the CLI.</p>`;
}
