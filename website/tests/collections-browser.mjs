import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {build,root} from '../scripts/build.mjs';
import {serve} from '../scripts/serve.mjs';

const tag=process.env.SITE_BASE==='/'?'root':'prefix';
const output=path.join(root,'test-results',`section-collections-${tag}`);await fs.mkdir(output,{recursive:true});
const {out,library,tutorialLibrary}=await build({local:true,directory:path.join(output,'site')});
const {server,origin,base}=await serve(0,{local:true,directory:out});
const browser=await chromium.launch({channel:'chrome'});const page=await browser.newPage();const errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
const proof={base,widths:[1440,390,320],collections:[],tutorials:[],project_documents:[]};
try{
 for(const width of proof.widths){
  await page.setViewportSize({width,height:1000});await page.goto(origin+base);await ready();
  assert.equal(await page.getByRole('heading',{name:'Project Docs',exact:true}).count(),1);
  assert.equal(await page.getByRole('heading',{name:'Tutorials',exact:true}).count(),1);
  assert.equal(await page.locator('.project-collections .collection-card').count(),5);
  assert.equal(await page.locator('.tutorial-collections .collection-card').count(),5);
  assert.equal(await page.locator('#record-search,.record-grid').count(),0);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.screenshot({path:path.join(output,`home-${width}.png`),fullPage:true});
  for(const group of tutorialLibrary.groups){
   await page.goto(origin+base);await ready();
   await page.locator(`.project-collections .collection-card[href="${group.projectDocsURL}"]`).click();await ready();
   assert.equal(new URL(page.url()).pathname,group.projectDocsURL);
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   const docs=library.groups.find(g=>g.id===group.id);
   if(docs.projects.length){
    await page.locator(`.project-card[href="${docs.projects[0].url}"]`).click();await ready();
    assert.equal(new URL(page.url()).pathname,docs.projects[0].url);
    assert.equal(await page.getByRole('link',{name:'Download Markdown ↓',exact:true}).count(),1);
    if(width===1440)proof.project_documents.push(docs.projects[0].url);
   }
   await page.goto(origin+base);await ready();
   await page.locator(`.tutorial-collections .collection-card[href="${group.url}"]`).click();await ready();
   assert.equal(await page.locator('.tutorial-courses .record-card').count(),group.courses.length);
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   if(width===1440)proof.collections.push({id:group.id,tutorials:group.courses.length,project_docs:docs.projects.length});
   for(const course of group.courses){
    await page.goto(origin+group.url);await ready();
    await page.locator(`.tutorial-courses .record-card[href="${course.url}"]`).click();await ready();
    await page.getByRole('link',{name:/Begin Lesson 1/}).first().click();await ready();
    assert.ok(new URL(page.url()).pathname.startsWith(course.url+'chapters/'));
    assert.equal(await page.getByRole('navigation',{name:'Breadcrumb'}).getByRole('link',{name:group.title,exact:true}).getAttribute('href'),group.url);
    assert.equal(await page.locator('.katex-error').count(),0);
    await page.locator('.tutorial-board img').evaluateAll(images=>Promise.all(images.map(img=>img.decode())));
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    if(width===1440)proof.tutorials.push({url:course.url,chapter:new URL(page.url()).pathname,collection:group.id});
   }
  }
  await page.goto(origin+base);await ready();await page.keyboard.press('/');await page.waitForURL(origin+base+'search/');await ready();
  await page.getByRole('searchbox',{name:'Search records'}).fill('attention labs');
  assert.equal(await page.locator('.record-card:visible[href*="/records/tutorials/attention-labs-"]').count(),2);
  await page.getByRole('searchbox').fill('not-a-real-record-342233');assert.ok(await page.locator('#record-empty').isVisible());
 }
 await page.setViewportSize({width:1440,height:1000});await page.goto(origin+base+'tutorials/attention-labs/');await ready();
 assert.equal(await page.locator('.sidebar-tutorials [aria-current="page"]').count(),1);
 await page.screenshot({path:path.join(output,'attention-tutorial-collection-desktop.png')});
 await page.setViewportSize({width:390,height:1000});await page.screenshot({path:path.join(output,'attention-tutorial-collection-mobile.png')});
 await page.emulateMedia({media:'print'});assert.ok(await page.locator('h1').isVisible());await page.screenshot({path:path.join(output,'collection-print.png')});
 assert.equal(proof.tutorials.length,tutorialLibrary.groups.reduce((count,g)=>count+g.courses.length,0));assert.deepEqual(errors,[]);proof.passed=true;
 await fs.writeFile(path.join(output,'review.json'),JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify({base,collections:5,tutorials:proof.tutorials.length,widths:proof.widths,passed:true}));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
