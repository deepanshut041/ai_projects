// Focused author audit; runs against a separately built local output at each base.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const workspace=path.resolve(here,'../../../../..');
const website=path.join(workspace,'ai_projects/docs/website');
const {chromium}=await import(path.join(website,'node_modules/playwright/index.mjs'));
const {build}=await import(path.join(website,'scripts/build.mjs'));
const {serve}=await import(path.join(website,'scripts/serve.mjs'));
const suffix=process.env.SITE_BASE==='/'?'root':'prefix';
const captured=JSON.parse(await fs.readFile(path.join(workspace,'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1/code/source-manifest.json'),'utf8'));
const expectedSourceFiles=Object.keys(captured).length;
const directory=path.join(website,'test-results/attention-layout-'+suffix);
const {out,library}=await build({local:true,directory});
const {server,origin,base}=await serve(0,{directory:out,local:true});
const records=JSON.parse(await fs.readFile(path.join(out,'search.json'),'utf8'));
const pages=records.filter(r=>r.tutorial?.project_id==='attention-is-all-you-need');
const chapters=pages.filter(r=>r.tutorial.kind==='chapter');
const project=library.groups.flatMap(g=>g.projects).find(p=>p.id==='attention-is-all-you-need');
const notes=project.pages.find(p=>p.file==='docs/implementation/01_attention-mechanism/02-notes-to-understanding.md');
assert.equal(chapters.length,12);assert.ok(notes);
const result={base,chapters:chapters.length,localProjectPages:project.pages.length,widths:[1440,768,390,320],checked:[]};
let browser;
try {
  browser=await chromium.launch();
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',response=>{if(response.status()>=400)errors.push(response.status()+' '+response.url());});
  const ready=()=>page.waitForFunction(()=>document.documentElement.dataset.ready==='true');
  // Validate every course/project page and local link/asset, not just its HTTP title.
  for(const item of [...pages,...project.pages]) {
    const response=await fetch(origin+item.url);assert.equal(response.status,200,item.url);
    const html=await response.text();
    assert.ok(!html.includes('tutorial-draft-template'));
    for(const [,href] of html.matchAll(/(?:href|src)="([^"]+)"/g))
      if(href.startsWith(base)&&!href.includes('#'))assert.equal((await fetch(origin+href,{method:'HEAD'})).status,200,href);
  }
  for(const width of result.widths) {
    await page.setViewportSize({width,height:1000});
    for(const item of [...chapters,...project.pages.filter(p=>/README|02-notes-to-understanding/.test(p.file))]) {
      await page.goto(origin+item.url);await ready();
      await page.locator('article img').evaluateAll(images=>Promise.all(images.map(img=>img.decode())));
      assert.equal(await page.locator('.katex-error').count(),0,item.url);
      assert.equal(await page.locator('pre.language-mermaid, pre:has(code.language-mermaid)').count(),0,'Unrendered diagram '+item.url);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Overflow '+width+' '+item.url);
      assert.ok(await page.locator('article img').evaluateAll(images=>images.every(i=>i.complete&&i.naturalWidth>0&&i.alt.length>0)),'Missing diagram '+item.url);
      if(item.tutorial?.kind==='chapter') {
        assert.equal(await page.locator('.practice-reveal').count(),1);
        await page.locator('.practice-reveal').click();
        assert.equal(await page.locator('.practice-reveal').getAttribute('aria-expanded'),'true');
        assert.ok(await page.locator('.practice-answer').isVisible());
      }
      result.checked.push({width,url:item.url});
    }
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(origin+project.url);await ready();
  const overview=page.locator('.collection-overview > summary');
  await overview.click();
  const notebookLink=page.locator('article a').filter({hasText:/^notebook$/});
  assert.equal(await notebookLink.count(),1);
  const notebook=await (await fetch(origin+await notebookLink.getAttribute('href'))).json();
  assert.equal(notebook.cells.filter(c=>c.cell_type==='code').length,6);
  assert.ok(JSON.stringify(notebook.cells).includes('from google.colab import files'));
  assert.ok(JSON.stringify(notebook.cells).includes('src/attention.py'));
  assert.ok(notebook.cells.filter(c=>c.cell_type==='code').every(c=>c.execution_count===null&&c.outputs.length===0));
  assert.ok(!JSON.stringify(notebook.cells).includes('from attention_is_all_you_need'));
  await overview.click();
  assert.equal(await page.getByRole('heading',{name:'Phases',exact:true}).count(),1);
  assert.equal(await page.getByRole('heading',{name:'Implementation',exact:true}).count(),1);
  assert.equal(await page.locator('.implementation-folder > summary').count(),2);
  await page.getByRole('link',{name:'Open project docs',exact:false}).click();await ready();
  await page.locator('article').getByRole('link',{name:'in-depth Attention tutorial',exact:true}).click();await ready();
  assert.equal(new URL(page.url()).pathname,chapters[0].url);
  for(const chapter of chapters.slice(0,-1)) {
    assert.equal(new URL(page.url()).pathname,chapter.url);
    await page.locator('.tutorial-pager [rel="next"]').click();await ready();
  }
  assert.equal(await page.locator('.tutorial-pager [rel="next"]').count(),0);
  const jsonLink=page.getByRole('link',{name:'captured source bundle',exact:true});
  const bundle=await (await fetch(origin+await jsonLink.getAttribute('href'))).json();
  assert.equal(Object.keys(bundle.files).length,expectedSourceFiles);
  assert.ok(bundle.files['src/core/attention.py']&&bundle.files['src/utils/attention_visuals.py']&&bundle.files['notebooks/attention_colab.ipynb']);
  assert.ok(!Object.keys(bundle.files).some(name=>name.includes('src/attention_is_all_you_need')));
  const practice=pages.find(p=>p.tutorial.sourcePath==='practice/explain-attention.md');
  await page.goto(origin+practice.url);await ready();assert.equal(await page.locator('.practice-reveal').count(),12);
  const targets=[['projections',chapters.find(p=>p.tutorial.sourcePath.includes('05-learned'))],['gradients',chapters.find(p=>p.tutorial.sourcePath.includes('11-gradients'))],['project',project],['notes',notes]];
  for(const [label,item] of targets) {
    for(const width of [1440,390]) {
      await page.setViewportSize({width,height:1000});await page.goto(origin+item.url);await ready();
      await page.screenshot({path:path.join(website,'test-results/attention-layout-'+suffix+'-'+label+'-'+width+'.png')});
      if(label==='projections'&&width===1440) {
        await page.locator('article .tutorial-board').first().screenshot({path:path.join(website,'test-results/attention-layout-'+suffix+'-matrix-detail.png')});
      }
      if(label==='gradients'&&width===1440) {
        await page.locator('.katex-display').last().scrollIntoViewIfNeeded();
        await page.screenshot({path:path.join(website,'test-results/attention-layout-'+suffix+'-equation-detail.png')});
      }
    }
  }
  await page.goto(origin+chapters[0].url);await ready();
  const markdown=page.getByRole('link',{name:'Download complete lesson',exact:false});
  const downloaded=await (await fetch(origin+await markdown.getAttribute('href'))).text();
  assert.ok(downloaded.includes('Turn comparisons into normalized attention weights')&&downloaded.includes('dL')&&downloaded.includes('readiness'));
  await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
  assert.ok(await page.locator('.practice-answer').isVisible());
  await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
  const plain=await browser.newPage({javaScriptEnabled:false});
  await plain.goto(origin+notes.url);assert.ok(await plain.getByRole('navigation',{name:'Project contents'}).isVisible());await plain.close();
  assert.deepEqual(errors,[]);
  result.passed=true;result.downloadSourceFiles=expectedSourceFiles;result.practicePageChecks=12;result.errors=errors;
  await fs.writeFile(path.join(here,'browser-'+suffix+'.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify({base,checks:result.checked.length,passed:true}));
} finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
