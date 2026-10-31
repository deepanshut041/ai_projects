import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from '/mnt/projects/ai/ai_projects/docs/website/node_modules/playwright/index.mjs';
import { build, root } from '/mnt/projects/ai/ai_projects/docs/website/scripts/build.mjs';
import { serve } from '/mnt/projects/ai/ai_projects/docs/website/scripts/serve.mjs';
const label=process.env.SITE_BASE==='/'?'root':'prefix';
const folder=path.join(root,'test-results',`document-assistant-${label}`);
await fs.mkdir(folder,{recursive:true});
const out=label==='root'?(await build({directory:path.join(folder,'site')})).out:path.join(root,'dist');
const {server,origin,base}=await serve(0,{directory:out});
const browser=await chromium.launch();
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`);});
const records=JSON.parse(await fs.readFile(path.join(out,'search.json'),'utf8'));
const pages=records.filter(r=>r.id.startsWith('tutorials/document-assistant-quantization-lab-v1'));
const chapters=pages.filter(r=>r.tutorial.kind==='chapter');
assert.equal(chapters.length,24);
let math=0,practice=0,links=0,figures=0;
try {
 for(const record of pages){
  await page.goto(origin+record.url);await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  assert.equal(await page.locator('.katex-error').count(),0,record.id);
  assert.ok(!(await page.locator('article').innerText()).match(/\{\{example\}\}|\[\[practice:|tutorial-draft-template/));
  await page.locator('article img').evaluateAll(images=>Promise.all(images.map(i=>i.decode())));
  math+=await page.locator('.katex').count();figures+=await page.locator('.tutorial-board').count();
  if(record.tutorial.kind==='chapter'){
   const reveal=page.locator('.practice-reveal');assert.equal(await reveal.count(),1);
   await reveal.click();assert.equal(await reveal.getAttribute('aria-expanded'),'true');practice++;
   await reveal.click();assert.equal(await reveal.getAttribute('aria-expanded'),'false');
   const i=chapters.indexOf(record);const pager=page.locator('.tutorial-pager');
   if(i<23)assert.equal(await pager.locator('[rel="next"]').getAttribute('href'),chapters[i+1].url);
   else assert.equal(await pager.locator('[rel="next"]').count(),0);
  }
  const hrefs=await page.locator('article a').evaluateAll(as=>as.map(a=>a.getAttribute('href')).filter(h=>h&&h.startsWith('/')));
  for(const href of new Set(hrefs)){assert.equal((await fetch(origin+href)).status,200,href);links++;}
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:900});
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Overflow ${record.id} at ${width}`);
  }
  await page.setViewportSize({width:1440,height:1000});
 }
 const memory=chapters.find(c=>c.id.endsWith('/memory'));
 await page.goto(origin+memory.url);await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
 const scenario=page.locator('.tutorial-scenario');assert.equal(await scenario.count(),1);
 const controls=scenario.locator('select');assert.equal(await controls.count(),2);
 for(const w of ['BF16','toy Q4'])for(const c of ['2 GiB','8 GiB']){
  await controls.nth(0).selectOption({label:w});await controls.nth(1).selectOption({label:c});
  const visible=await scenario.innerText();assert.ok(visible.includes(w)&&visible.includes(c));
 }
 for(const id of ['contract','quantization','compensation','cache','handoff']){
  const record=chapters.find(r=>r.id.endsWith('/'+id));await page.goto(origin+record.url);await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  await page.screenshot({path:path.join(folder,id+'-desktop.png')});
  if(await page.locator('.tutorial-board').count())await page.locator('.tutorial-board').first().screenshot({path:path.join(folder,id+'-figure.png')});
  if(await page.locator('.katex-display').count())await page.locator('.katex-display').first().screenshot({path:path.join(folder,id+'-math.png')});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(folder,id+'-mobile.png')});
  if(await page.locator('.tutorial-board').count())await page.locator('.tutorial-board').first().screenshot({path:path.join(folder,id+'-figure-mobile.png')});
  await page.setViewportSize({width:1440,height:1000});
 }
 // Inspect text extents in original SVGs; no image-loading success claim substitutes.
 const figRoot=path.join(out,'tutorial-assets/document-assistant/quantization-lab/v1/figures');
 const svgIssues=[];
 for(const file of await fs.readdir(figRoot)){
  await page.goto(origin+base+'tutorial-assets/document-assistant/quantization-lab/v1/figures/'+file);
  const issues=await page.locator('svg').evaluate(svg=>{
   const boxes=[...svg.querySelectorAll('rect')].filter(r=>r.getAttribute('rx'));
   return [...svg.querySelectorAll('text')].flatMap(t=>{
    const b=t.getBBox();const x=+t.getAttribute('x'),y=+t.getAttribute('y');
    const box=boxes.find(r=>x===+r.getAttribute('x')+16&&y>=+r.getAttribute('y')&&y<=+r.getAttribute('y')+ +r.getAttribute('height'));
    if(box&&b.x+b.width>+box.getAttribute('x')+ +box.getAttribute('width')-8)return [t.textContent];
    return b.x<0||b.x+b.width>900?[t.textContent]:[];
   });
  });
  if(issues.length)svgIssues.push({file,issues});
  await page.locator('svg').screenshot({path:path.join(folder,file+'.png')});
 }
 // Print path: rendering the complete route must retain prose and reveal answers.
 const complete=pages.find(p=>p.tutorial.kind==='complete');await page.goto(origin+complete.url);await page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
 await page.emulateMedia({media:'print'});assert.ok((await page.locator('article').innerText()).includes('2,359,296'));
 await page.pdf({path:path.join(folder,'complete-print.pdf'),format:'A4',printBackground:true});
 assert.deepEqual(errors,[]);
 const result={label,base,pages:pages.length,chapters:chapters.length,practice,links,math,figures,svgIssues,errors,print:'generated'};
 await fs.writeFile(path.join(folder,'receipt.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
 assert.deepEqual(svgIssues,[],'SVG labels overflow their intended boxes');
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
