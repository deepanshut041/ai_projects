import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const here=path.dirname(fileURLToPath(import.meta.url));
let workspace=here;
while(!await fs.stat(path.join(workspace,'ai')).catch(()=>null)) workspace=path.dirname(workspace);
const website=path.join(workspace,'ai_projects/docs/website');
const require=createRequire(path.join(website,'package.json'));
const {chromium}=require('playwright');
const {build}=await import(pathToFileURL(path.join(website,'scripts/build.mjs')));
const {serve}=await import(pathToFileURL(path.join(website,'scripts/serve.mjs')));
const course=path.join(workspace,'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1');
const manifest=JSON.parse(await fs.readFile(path.join(course,'lesson.json'),'utf8'));
const root=path.join(workspace,'.ai/attention-text-rewrite/2026-10-24');
const suffix=process.env.SITE_BASE==='/'?'root':'prefix';
await fs.mkdir(path.join(root,'browser'),{recursive:true});
const built=await build({local:true,directory:path.join(root,`site-${suffix}`)});
const {server,origin,base}=await serve(0,{directory:built.out,local:true});
const notes='projects/transformer-labs/attention-is-all-you-need/docs/implementation/01_attention-mechanism/02-notes-to-understanding/';
const tutorial='records/tutorials/attention-is-all-you-need-attention-lab-v1/chapters/';
const programs=JSON.parse(await fs.readFile(path.join(course,'examples/walkthrough/route.json'),'utf8'));
const expected=await Promise.all(programs.map(p=>fs.readFile(path.join(course,p.path),'utf8')));
const report={base,widths:[1440,390],pages:[],notes:[],errors:[],approval:'pending',newVideos:0};
let browser;
try {
  browser=await chromium.launch();
  const page=await browser.newPage();
  page.on('pageerror',error=>report.errors.push(error.message));
  const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  for(const width of report.widths) {
    await page.setViewportSize({width,height:1000});
    await page.goto(origin+base+notes);await ready();
    assert.equal(await page.locator('video').count(),0,'Old Attention videos must be absent');
    assert.equal(await page.locator('.katex-error').count(),0);
    const python=page.locator('article code.language-python');
    assert.equal(await python.count(),10);
    for(let i=0;i<10;i++) {
      assert.equal(await python.nth(i).textContent(),expected[i]);
      assert.ok(await python.nth(i).isVisible());
      assert.ok(await python.nth(i).evaluate(e=>!e.closest('details,[hidden]')),'Code must not be collapsed');
      assert.ok(await python.nth(i).locator('..').evaluate(e=>e.scrollHeight<=e.clientHeight+2),'Code must not be vertically clipped');
    }
    assert.equal(await page.locator('article h2').count(),11);
    assert.equal(await page.locator('article .project-diagram svg').count(),2);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    const link=await page.locator('article a[href*="01_weighted.py"]').getAttribute('href');
    assert.equal(await (await fetch(new URL(link,page.url()))).text(),expected[0]);
    await page.locator('article h2').first().scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(root,`browser/${suffix}-notes-start-${width}.png`)});
    await python.nth(3).scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(root,`browser/${suffix}-complete-projection-code-${width}.png`)});
    report.notes.push({width,pythonBlocks:10,fullVisibleDefinitions:true,verticalClipping:false,players:0,downloadMatches:true});
    for(let i=0;i<manifest.chapters.length;i++) {
      const chapter=manifest.chapters[i];
      await page.goto(origin+base+tutorial+chapter.id+'/');await ready();
      assert.equal(await page.locator('video').count(),0);
      assert.equal(await page.locator('.katex-error').count(),0);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),chapter.id);
      if(i<manifest.chapters.length-1) assert.ok((await page.locator('.tutorial-pager a[rel="next"]').getAttribute('href')).endsWith('/'+manifest.chapters[i+1].id+'/'));
      if(i>0) assert.ok((await page.locator('.tutorial-pager a[rel="prev"]').getAttribute('href')).endsWith('/'+manifest.chapters[i-1].id+'/'));
      if(i<10) {
        const code=page.locator('article code.language-python');assert.equal(await code.count(),1);
        assert.equal(await code.textContent(),expected[i]);
        assert.ok(await code.evaluate(e=>!e.closest('details,[hidden]')));
      }
      if(chapter.id==='weighted') {
        const reveal=page.locator('.practice-reveal');await reveal.click();
        assert.equal(await reveal.getAttribute('aria-expanded'),'true');
        assert.ok(await page.locator('article code.language-python').isVisible());
        await page.screenshot({path:path.join(root,`browser/${suffix}-first-chapter-${width}.png`)});
      }
      report.pages.push({chapter:chapter.id,width,order:i+1,players:0});
    }
  }
  await page.goto(origin+base+notes);await ready();await page.emulateMedia({media:'print'});
  assert.equal(await page.locator('video').count(),0);
  assert.ok(await page.locator('article code.language-python').first().isVisible());
  await page.screenshot({path:path.join(root,`browser/${suffix}-print.png`)});
  assert.deepEqual(report.errors,[]);report.passed=true;
  await fs.writeFile(path.join(root,`browser-${suffix}.json`),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({base,chapterChecks:report.pages.length,notesWidths:report.notes.length,codeBlocks:10,players:0,passed:true}));
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
