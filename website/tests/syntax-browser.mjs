import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {build} from '../scripts/build.mjs';
import {serve} from '../scripts/serve.mjs';
import {renderer} from '../scripts/content.mjs';

const root=fileURLToPath(new URL('../',import.meta.url));
const projects=JSON.parse(await fs.readFile(path.join(root,'package.json'),'utf8')).name==='ai-projects-docs';
const site=projects?'projects':'research';
const suffix=process.env.SITE_BASE==='/'?'root':'prefix';
const directory=path.join(root,'test-results',`syntax-${site}-${suffix}`);
const built=await build({local:true,directory});
const {server,origin,base}=await serve(0,{directory:built.out,local:true});
const samples=[
 ['python','def attend(query):\n    # Keep each key visible\n    return "value" if query > 0 else None\n'],
 ['cpp','#include <vector>\nint main() { return 42; }\n'],
 ['typescript','const message: string = "ready";\n'],
 ['bash','# A reproducible local run\nexport DEVICE="cpu"\npython -m train --steps 42\n'],
 ['json','{"enabled": true, "name": "attention", "steps": 42}\n'],
 ['yaml','model: small\nsteps: 42\n'],
 ['sql','SELECT name FROM runs WHERE score > 0;\n'],
 ['html','<img src="sample.png" onerror="alert(1)">\n'],
 ['text','Raw output <tag> must remain unchanged.\n'],
 ['unknown-example','Unknown output <tag> stays literal.\n'],
 ['python','message = "'+ 'long-code-'.repeat(60)+'"\n'],
];
const {md}=renderer(new Set(),'guides/syntax-example.md',base);
const specimen=md.render(samples.map(([language,code])=>'```'+language+'\n'+code+'```\n').join('\n'));
const result={site,base,widths:[1440,390],actualPages:[],screenContrast:[],printContrast:[],languages:samples.map(([language])=>language)};
let browser;
async function htmlFiles(folder) {
 const output=[];
 for(const item of await fs.readdir(folder,{withFileTypes:true})) {
  const file=path.join(folder,item.name);
  if(item.isDirectory()&&!['assets','tutorial-assets','local-files'].includes(item.name))output.push(...await htmlFiles(file));
  else if(item.isFile()&&item.name.endsWith('.html'))output.push(file);
 }
 return output;
}
async function contrast(page) {
 return page.locator('article pre code [class*="hljs-"]').evaluateAll(spans=>{
  const luminance=color=>{
   const rgb=color.match(/[\d.]+/g).slice(0,3).map(Number).map(x=>x/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4);
   return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];
  };
  const seen=new Map();
  for(const span of spans){
   const fg=getComputedStyle(span).color,bg=getComputedStyle(span.closest('pre')).backgroundColor;
   const a=luminance(fg),b=luminance(bg),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
   seen.set(fg,{color:fg,background:bg,ratio});
  }
  return [...seen.values()];
 });
}
try {
 const pages=[];
 for(const file of await htmlFiles(built.out)) {
  const html=await fs.readFile(file,'utf8');
  const tokens=(html.match(/class="hljs-/g)||[]).length;
  if(tokens)pages.push({file,tokens,html});
 }
 assert.ok(pages.length>0,'Actual site must contain highlighted code');
 const representatives=[
  pages.find(p=>p.file.includes(projects?'02-notes-to-understanding':'implementation')),
  pages.find(p=>p.file.includes(projects?'chapters':'experiments')),
  pages.toSorted((a,b)=>b.tokens-a.tokens)[0],
 ].filter((p,i,all)=>p&&all.indexOf(p)===i);
 browser=await chromium.launch();
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
 for(const width of result.widths) {
  await page.setViewportSize({width,height:1000});
  for(const target of representatives) {
   const relative=path.relative(built.out,target.file).split(path.sep).join('/').replace(/index\.html$/,'');
   await page.goto(origin+base+relative);await ready();
   assert.ok(await page.locator('.markdown pre code [class*="hljs-"]').count()>0);
   assert.equal(await page.locator('.katex-error').count(),0);
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   const code=page.locator('.markdown pre code:has([class*="hljs-"])').first();
   await code.evaluate(e=>{for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;});
   await code.scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(root,'test-results',`syntax-${site}-${suffix}-actual-${width}-${result.actualPages.length}.png`)});
   result.actualPages.push({width,url:page.url(),tokens:target.tokens});
  }
  // Use the real built styles and layout; this is an explicit QA language specimen.
  const specimenPage=pages.find(p=>p.html.includes('<article class="markdown"'));
  const specimenURL=path.relative(built.out,specimenPage.file).split(path.sep).join('/').replace(/index\.html$/,'');
  await page.goto(origin+base+specimenURL);await ready();
  await page.locator('article').first().evaluate((article,html)=>{article.innerHTML='<h2>Syntax highlighting QA specimen</h2>'+html;},specimen);
  const code=page.locator('article pre code');assert.equal(await code.count(),samples.length);
  for(let i=0;i<samples.length;i++)assert.equal(await code.nth(i).textContent(),samples[i][1]);
  assert.equal(await code.nth(8).locator('span').count(),0);assert.equal(await code.nth(9).locator('span').count(),0);
  assert.equal(await page.locator('article img, article script').count(),0);
  const first=code.first();
  const copied=await first.evaluate(e=>{const range=document.createRange();range.selectNodeContents(e);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);const text=selection.toString();selection.removeAllRanges();return text;});
  // Browser text selection omits the final visual line break; DOM text above
  // checks all source bytes, including that trailing newline.
  assert.equal(copied,samples[0][1].replace(/\n$/,''));
  const pre=code.last().locator('..');
  const scroll=await pre.evaluate(e=>{e.focus();return {scrolls:e.scrollWidth>e.clientWidth,font:getComputedStyle(e.querySelector('code')).fontSize,focused:document.activeElement===e,outline:getComputedStyle(e).outlineWidth};});
  assert.ok(scroll.scrolls&&scroll.focused);assert.equal(scroll.font,'14px');assert.notEqual(scroll.outline,'0px');
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  const screen=await contrast(page);assert.ok(screen.length>=6);assert.ok(screen.every(c=>c.ratio>=4.5),JSON.stringify(screen));result.screenContrast.push({width,colors:screen});
  await page.locator('article h2').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'test-results',`syntax-${site}-${suffix}-specimen-${width}.png`)});
 }
 await page.emulateMedia({media:'print'});
 const print=await contrast(page);assert.ok(print.every(c=>c.ratio>=4.5),JSON.stringify(print));result.printContrast=print;
 assert.ok(await codeWraps(page));
 await page.screenshot({path:path.join(root,'test-results',`syntax-${site}-${suffix}-print.png`)});
 assert.deepEqual(errors,[]);
 result.highlightedPages=pages.length;result.copyPreservesSource=true;result.noClientHighlighter=true;result.passed=true;
 await fs.writeFile(path.join(root,'test-results',`syntax-${site}-${suffix}.json`),JSON.stringify(result,null,2)+'\n');
 console.log(`Syntax browser checks passed: ${site}, ${base}, ${pages.length} highlighted pages, desktop/mobile, exact source, focus/scroll and screen/print contrast.`);
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
async function codeWraps(page){return page.locator('article pre code').last().evaluate(e=>getComputedStyle(e).whiteSpace==='pre-wrap');}
