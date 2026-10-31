import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { build } from '../../../website/scripts/build.mjs';
import { serve } from '../../../website/scripts/serve.mjs';
const require=createRequire(new URL('../../../website/package.json',import.meta.url));
const {chromium}=require('playwright');
const website=fileURLToPath(new URL('../../../website/',import.meta.url));
const label=process.env.SITE_BASE==='/'?'root':'prefix';
const output=path.join(website,'test-results',`attention-inline-visuals-${label}`);
await fs.mkdir(output,{recursive:true});
const {out}=await build({directory:path.join(output,'site')});
const {server,origin,base}=await serve(0,{directory:out});
const browser=await chromium.launch({channel:'chrome'});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const review={base,widths:[1440,390,320],chapters:[],images:0};
const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
// The new teaching images must work with the external presentation hosts unavailable.
await page.route('https://**',route=>route.abort());
const paths=[['attention-foundations',[['map',1],['summary',3],['additive',4],['scale',3],['self',3],['projection',3],['causal',4]]],['attention-heads',[['heads',2],['layout',3],['boundary',1]]]];
try{
 for(const [slug,chapters] of paths){
  const source=JSON.parse(await fs.readFile(new URL(`../../../tutorials/projects/attention-labs/${slug}/v1/project/visual-provenance.json`,import.meta.url),'utf8'));
  for(const [chapter,count] of chapters){
   const route=`${base}records/tutorials/attention-labs-${slug}-v1/chapters/${chapter}/`;
   await page.goto(origin+route);await ready();
   const images=page.locator('.tutorial-board img[src*="/original-slides/"]');
   assert.equal(await images.count(),count,route);
   assert.equal(await page.locator('.katex-error').count(),0);
   assert.equal(await page.locator('iframe[src*="lecture9-"]').count(),0);
   for(const img of await images.all()){
    await img.evaluate(img=>img.decode());
    const url=await img.getAttribute('src');
    assert.ok(url.startsWith(base+'tutorial-assets/attention-labs/'),url);
    const filename=url.split('/original-slides/')[1];
    const expected=source.figures.find(f=>f.file.endsWith('/'+filename));assert.ok(expected);
    const actual=await img.evaluate(img=>({width:img.naturalWidth,height:img.naturalHeight}));
    assert.deepEqual(actual,{width:expected.width,height:expected.height});
    const response=await fetch(origin+url);assert.equal(response.status,200);
    assert.equal(crypto.createHash('sha256').update(Buffer.from(await response.arrayBuffer())).digest('hex'),expected.sha256);
    const board=img.locator('xpath=ancestor::figure');
    assert.ok(await board.getByRole('link',{name:'Open diagram at full size ↗',exact:true}).isVisible());
    assert.equal(await img.locator('xpath=ancestor::a').count(),0,'No invalid image/figure anchor nesting');
   }
   const reveal=page.locator('.practice-reveal').first();await reveal.click();assert.equal(await reveal.getAttribute('aria-expanded'),'true');
   const markdown=await page.getByRole('link',{name:'Download Markdown'}).getAttribute('href');const text=await(await fetch(origin+markdown)).text();
   assert.equal((text.match(/!\[[^\]]+\]\(images\/original-slides\//g)||[]).length,count);
   for(const width of review.widths){
    await page.setViewportSize({width,height:1000});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`Overflow ${width} ${route}`);
    for(const img of await images.all()){
     const region=img.locator('xpath=parent::*');
     assert.equal(await region.getAttribute('tabindex'),'0');
     assert.equal(await img.locator('xpath=ancestor::figure').getAttribute('data-source-capture'),'true');
     assert.ok(await img.evaluate(img=>img.getBoundingClientRect().width<=img.parentElement.clientWidth+1),'The complete source diagram must fit its preview');
     if(await region.evaluate(el=>el.scrollWidth>el.clientWidth)){
      await region.evaluate(el=>{el.scrollLeft=0});await region.focus();await page.keyboard.press('ArrowRight');
      await page.waitForFunction(el=>el.scrollLeft>0,await region.elementHandle());
      await region.evaluate(el=>{el.scrollLeft=el.scrollWidth});assert.ok(await region.evaluate(el=>el.scrollLeft>0));
      await region.evaluate(el=>{el.scrollLeft=0});
     }
    }
    // Capture the first and last actual diagram in its lesson context, including attribution.
    for(const [position,image] of [['first',images.first()],['last',images.last()]]){
     const board=image.locator('xpath=ancestor::figure');await board.scrollIntoViewIfNeeded();
     await page.screenshot({path:path.join(output,`${slug}-${chapter}-${width}-${position}.png`)});
    }
   }
   await page.setViewportSize({width:1440,height:1000});
   const first=images.first().locator('xpath=ancestor::figure');
   const [popup]=await Promise.all([page.waitForEvent('popup'),first.getByRole('link',{name:'Open diagram at full size ↗',exact:true}).click()]);
   await popup.waitForLoadState();assert.ok(popup.url().endsWith('.png'));await popup.locator('img').evaluate(img=>img.decode());await popup.close();
   await page.emulateMedia({media:'print'});
   for(const img of await images.all())assert.ok(await img.evaluate(img=>img.getBoundingClientRect().width<=img.parentElement.getBoundingClientRect().width+1));
   assert.equal(await page.locator('.practice-answer').first().evaluate(el=>getComputedStyle(el).display),'block');
   await page.screenshot({path:path.join(output,`${slug}-${chapter}-print.png`)});await page.emulateMedia({media:'screen'});
   review.chapters.push({slug,chapter,images:count,hashes:true,whole_diagram_preview:true,full_size_popup:true,markdown:true,print:true});review.images+=count;
  }
 }
 assert.equal(review.chapters.length,10);assert.equal(review.images,27);assert.deepEqual(errors,[]);
 review.external_presentation_hosts_blocked=true;review.passed=true;
 await fs.writeFile(path.join(output,'review.json'),JSON.stringify(review,null,2)+'\n');
 console.log(JSON.stringify({base,chapters:review.chapters.length,inline_source_images:review.images,passed:true}));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
