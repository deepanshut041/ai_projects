// Local browser acceptance uses generated, labelled codec fixtures; no provider requests.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { build, root } from '../scripts/build.mjs';
import { serve } from '../scripts/serve.mjs';

const projects=root.includes('ai_projects');
const site=projects?'project-docs':'research-docs';
const workspace=path.resolve(root,projects?'../../..':'../..');
const temp=await fs.mkdtemp(path.join(os.tmpdir(),'video-browser-'));
const managed=path.join(workspace,'.ai/tools/bin/ffmpeg');
const ffmpeg=await fs.access(managed).then(()=>managed).catch(()=> 'ffmpeg');
let browser,server;
try {
  execFileSync(ffmpeg,['-v','error','-y','-f','lavfi','-i','color=c=0x173b4a:s=640x360:r=24:d=3','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=3','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart','-shortest',path.join(temp,'video.mp4')]);
  execFileSync(ffmpeg,['-v','error','-y','-i',path.join(temp,'video.mp4'),'-frames:v','1','-threads','1',path.join(temp,'poster.jpg')]);
  await fs.writeFile(path.join(temp,'captions.vtt'),'WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nPlayback fixture, not a tutorial result.\n\n2\n00:00:01.000 --> 00:00:03.000\nRead the selected tutorial below.\n');
  await fs.writeFile(path.join(temp,'transcript.md'),'# Playback fixture\n\nThis synthetic clip checks playback, seeking and captions. It is not a narrated explanation of the tutorial.');
  const filenames={src:'video.mp4',poster:'poster.jpg',captions:'captions.vtt',transcript:'transcript.md'};
  const media=Object.fromEntries(Object.entries(filenames).map(([key,name],i)=>[key,`local-files/${String(i).repeat(24)}/${name}`]));
  const files=Object.fromEntries(Object.entries(filenames).map(([key,name])=>[media[key].slice('local-files/'.length),path.join(temp,name)]));
  const sources=projects?[
    'ai_projects/docs/tutorials/projects/p00/transformer-foundations/v1/lessons/03-numbers.md',
    'ai_projects/docs/tutorials/projects/p00/transformer-course/v1/lessons/04-learning.md',
    'cli/docs/tutorial-video.md',
  ]:['ai_research/research_docs/content/questions/retrieval-evaluation.md'];
  const items=await Promise.all(sources.map(async(document,i)=>{
    const text=await fs.readFile(path.join(workspace,document),'utf8');
    const heading=text.match(/^###? (.+)$/m)?.[1];
    return {id:'video-'+String(i+1).repeat(24),video_id:'cli/playback-fixture',title:'Playback QA fixture — existing docs',status:'ready',duration:3,language:'en',media,target:{document,heading},source:{document,text}};
  }));
  const output=await build({local:true,directory:path.join(root,'test-results',process.env.SITE_BASE==='/'?'video-root':'video-prefixed'),videoCatalog:{schema_version:1,site,files,items}});
  const local=await serve(0,{directory:output.out,local:true}); server=local.server;
  browser=await chromium.launch();
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  const index=await (await fetch(local.origin+local.base+'videos/index.json')).json();
  assert.equal(index.bindings.length,sources.length);
  for(const width of [1440,390]) {
    await page.setViewportSize({width,height:1000});
    await page.goto(local.origin+local.base+'videos/');await ready();
    assert.equal(await page.locator('.video-card').count(),1,'one movie may have several tutorial targets');
    await page.getByRole('searchbox',{name:'Search videos'}).fill('missing-qa-000');
    assert.ok(await page.locator('#video-empty').isVisible());
    await page.getByRole('searchbox',{name:'Search videos'}).fill('');
    await page.getByRole('link',{name:'Play video',exact:true}).click();await ready();
    assert.ok(await page.getByRole('heading',{name:'Transcript',exact:true}).isVisible());
    for(const [i,binding] of index.bindings.entries()) {
      await page.goto(local.origin+binding.target_url);await ready();
      if(projects && i<2) assert.ok(new URL(page.url()).pathname.includes('/records/tutorials/'),'known course uses its actual chapter route');
      if(projects && i===2) assert.ok(new URL(page.url()).pathname.includes('/source/'),'ordinary Markdown needs no course manifest');
      const video=page.locator('figure[data-video-id="cli/playback-fixture"] video');
      assert.equal(await video.count(),1);
      const check=await video.evaluate(async element=>{
        element.muted=true;element.load();
        if(element.readyState<1) await new Promise((resolve,reject)=>{element.onloadedmetadata=resolve;element.onerror=()=>reject(new Error('metadata failed'));});
        await element.play();await new Promise(resolve=>setTimeout(resolve,200));element.pause();
        const played=element.currentTime>0;
        const seek=new Promise(resolve=>element.addEventListener('seeked',resolve,{once:true}));element.currentTime=2;await seek;
        element.textTracks[0].mode='hidden';
        for(let n=0;n<20 && !element.textTracks[0].cues?.length;n++) await new Promise(resolve=>setTimeout(resolve,50));
        element.playbackRate=1.5;element.focus();
        return {played,seeked:Math.abs(element.currentTime-2)<.2,cues:element.textTracks[0].cues.length,focused:document.activeElement===element,rate:element.playbackRate,autoplay:element.autoplay,preload:element.preload};
      });
      assert.ok(check.played&&check.seeked&&check.focused);assert.equal(check.cues,2);assert.equal(check.rate,1.5);assert.equal(check.autoplay,false);assert.equal(check.preload,'none');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'responsive page');
      await video.scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(root,'test-results',`video-${projects?'projects':'research'}-${width}-${i}-${local.base==='/'?'root':'prefix'}.png`)});
    }
  }
  await page.emulateMedia({media:'print'});
  assert.equal(await page.locator('figure[data-video-id="cli/playback-fixture"] video').evaluate(element=>getComputedStyle(element).display),'none');
  await page.emulateMedia({media:'screen'});
  await page.setViewportSize({width:720,height:900});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'reflow at effective 200% desktop zoom');
  assert.equal((await fetch(local.origin+local.base+'local-files.json')).status,404);
  assert.deepEqual(errors,[]);
  console.log(`Video browser acceptance passed: ${site}, ${sources.length} source types, ${local.base}, desktop/phone, play/pause/seek/rate/focus/captions, search, source routes and print.`);
} finally {await browser?.close();if(server) await new Promise(resolve=>server.close(resolve));await fs.rm(temp,{recursive:true,force:true});}
