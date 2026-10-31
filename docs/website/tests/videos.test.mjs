import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { videoLibrary, videoHome } from '../scripts/videos.mjs';
import { serve } from '../scripts/serve.mjs';

const site = process.cwd().includes('research_docs') ? 'research-docs' : 'project-docs';
const id = 'video-' + 'a'.repeat(24);
const media = Object.fromEntries(['src','poster','captions','transcript'].map((key, i) => [key, `local-files/${String(i).repeat(24)}/${['video.mp4','poster.jpg','captions.vtt','transcript.md'][i]}`]));
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(),'reader-video-'));
  const transcript = path.join(root,'transcript.md');
  await fs.writeFile(transcript,'# Transcript\n\nA worked explanation with [unavailable file](../../secret.env).');
  const files = Object.fromEntries(Object.values(media).map(value => [value.slice('local-files/'.length),transcript]));
  const item = {id,video_id:'projects/example',title:'A worked example',status:'ready',duration:120.5,language:'fr-CA',media,
    target:{heading:'Example'},source:{document:'ai_projects/docs/tutorials/example.md',text:'# Topic\n\n## Example\nWritten reasoning.\n\n[Unknown](../../secret.env)\n\n<script>alert(1)</script>'}};
  return {root, files,item, data:{schema_version:1,site,files,items:[item]}};
}

test('local overlays precede prose, preserve sources, resolve real routes and keep captions', async () => {
  const f=await fixture();
  try {
    const page={url:'/nested/records/tutorial/',html:'<h2 id="example">Example</h2><p>Written reasoning.</p>',headings:[{title:'Example',slug:'example'}]};
    const before=f.item.source.text;
    const output=await videoLibrary({workspace:f.root,site,base:'/nested/',catalogData:f.data,routes:new Map([[path.resolve(f.root,f.item.source.document),page.url]]),pages:[page]});
    assert.match(page.html,/<\/h2><figure/);
    assert.ok(page.html.indexOf('<video')<page.html.indexOf('<p>Written reasoning'));
    assert.equal(f.item.source.text,before);
    assert.equal(output.bindings[0].target_url,'/nested/records/tutorial/#example');
    assert.match(page.html,/srclang="fr-CA"/);
    assert.match(page.html,/preload="none"/);
    assert.doesNotMatch(page.html,/autoplay/);
    assert.match(output.pages.at(-1).body,/Transcript/);
    assert.doesNotMatch(output.pages.at(-1).body,/href="\.\.\/\.\.\/secret.env"/);
    const again=await videoLibrary({workspace:f.root,site,base:'/nested/',catalogData:f.data,routes:new Map([[path.resolve(f.root,f.item.source.document),page.url]]),pages:[page]});
    assert.equal((page.html.match(/<video /g)||[]).length,1,'reused legacy/overlay movie is not duplicated');
    assert.equal(again.bindings.length,1);
  } finally {await fs.rm(f.root,{recursive:true,force:true});}
});

test('unregistered tutorials get safe local source previews and unavailable videos get diagnostics', async () => {
  const f=await fixture();
  try {
    const output=await videoLibrary({workspace:f.root,site,base:'/',catalogData:f.data});
    assert.match(output.bindings[0].target_url,/\/videos\/projects\/example\/source\/video-/);
    const source=output.pages.find(page=>page.url.includes('/source/'));
    assert.match(source.body,/<video controls/);
    assert.doesNotMatch(source.body,/<script>alert|href="\.\.\/\.\.\/secret.env"/);
    assert.match(source.body,/Written reasoning/);
    f.data.items=[{id,video_id:'projects/example',title:'<script>bad</script>',status:'unavailable',problem:'Source changed'}];
    const invalid=await videoLibrary({workspace:f.root,site,base:'/',catalogData:f.data});
    const html=videoHome(invalid.items,'/');
    assert.match(html,/Source changed/);assert.doesNotMatch(html,/<video|<script>/);
    assert.equal(invalid.bindings.length,0);
    assert.match(videoHome([],'/'),/No videos match/);
    assert.doesNotMatch(videoHome([],'/'),/class="empty" hidden/);
  } finally {await fs.rm(f.root,{recursive:true,force:true});}
});

test('malformed artifact URLs and unsupported identities never become players', async () => {
  const f=await fixture();
  try {
    f.item.media.src='file:///tmp/private';
    await assert.rejects(videoLibrary({workspace:f.root,site,base:'/',catalogData:f.data}),/artifact mapping/);
    f.item.video_id='../outside';
    await assert.rejects(videoLibrary({workspace:f.root,site,base:'/',catalogData:f.data}),/identity/);
  } finally {await fs.rm(f.root,{recursive:true,force:true});}
});

test('local streaming supports HEAD/ranges and hides runtime maps; ordinary serving excludes media', async () => {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'video-transport-'));
  let server, publicServer;
  try {
    await fs.writeFile(path.join(root,'build.json'),JSON.stringify({base:'/nested/'}));
    await fs.writeFile(path.join(root,'404.html'),'Missing');
    const movie=path.join(root,'movie.mp4');
    await fs.writeFile(movie,'0123456789');
    await fs.writeFile(path.join(root,'local-files.json'),JSON.stringify({'movie/video.mp4':movie}));
    const local=await serve(0,{directory:root,local:true}); server=local.server;
    const url=local.origin+'/nested/local-files/movie/video.mp4';
    const head=await fetch(url,{method:'HEAD'});assert.equal(head.status,200);assert.equal(head.headers.get('content-length'),'10');assert.equal(await head.text(),'');
    for(const [range,status,body] of [['bytes=2-5',206,'2345'],['bytes=-3',206,'789'],['bytes=7-',206,'789'],['bytes=99-',416,''],['bytes=3-1',416,''],['bytes=1-2,4-5',416,'']]) {
      const response=await fetch(url,{headers:{Range:range}});assert.equal(response.status,status,range);assert.equal(await response.text(),body);
      if(status===206) assert.ok(response.headers.get('content-range'));
    }
    assert.equal((await fetch(url,{method:'POST'})).status,405);
    assert.equal((await fetch(local.origin+'/nested/local-files.json')).status,404);
    assert.equal((await fetch(local.origin+'/nested/%2e%2e%2fpackage.json')).status,404);
    const ordinary=await serve(0,{directory:root}); publicServer=ordinary.server;
    assert.equal((await fetch(ordinary.origin+'/nested/local-files/movie/video.mp4')).status,404);
  } finally {
    await Promise.all([server,publicServer].filter(Boolean).map(value=>new Promise(resolve=>value.close(resolve))));
    await fs.rm(root,{recursive:true,force:true});
  }
});
