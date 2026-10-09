import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const hash=raw=>crypto.createHash('sha256').update(raw).digest('hex');
export async function artifact(root,relative) {
  if (path.isAbsolute(relative) || relative.includes('\\') || relative.split('/').some(part=>['.','..',''].includes(part))) throw new Error('Unsafe tutorial artifact path');
  const target=path.resolve(root,relative);
  if (!target.startsWith(path.resolve(root)+path.sep)) throw new Error('Tutorial artifact escapes package');
  for (let current=target;current!==path.resolve(root);current=path.dirname(current)) {
    if ((await fs.lstat(current)).isSymbolicLink()) throw new Error('Tutorial symlinks are not self-contained');
  }
  const raw=await fs.readFile(target);
  if (raw.length>10*1024*1024) throw new Error('Tutorial artifact exceeds 10 MiB');
  return raw;
}

export async function tutorialRecords(siteRoot) {
  const root=path.join(siteRoot,'tutorials');
  let index;
  try { index=JSON.parse(await fs.readFile(path.join(root,'index.json'),'utf8')); }
  catch(error) { if(error.code==='ENOENT') return {records:[],assets:[],index:null};throw error; }
  if (index.schema_version!==1 || !Array.isArray(index.projects) || index.languages?.join(',')!=='cpp,python') throw new Error('Invalid tutorial index/language policy');
  const ids=new Set();const records=[];const assets=[];
  for (const project of index.projects) {
    if (!/^P\d{2}$/.test(project.id) || ids.has(project.id)) throw new Error('Invalid/duplicate tutorial project ID');
    ids.add(project.id);
    for (const relative of project.lessons) {
      const raw=await artifact(root,relative+'/lesson.json');
      const data=JSON.parse(raw);
      const directory=path.join(root,relative);
      const frozen=JSON.parse(await artifact(directory,'render.json'));
      if (data.project_id!==project.id || frozen.project_id!==project.id || data.slug!==frozen.slug || data.revision!==frozen.revision) throw new Error('Tutorial index/manifest identity mismatch');
      if (!/^[a-z][a-z0-9-]*$/.test(data.slug) || !/^[a-z][a-z0-9-]*$/.test(data.revision)) throw new Error('Invalid tutorial slug/revision');
      const url=`tutorial-assets/${project.id.toLowerCase()}/${data.slug}/${data.revision}`;
      for (const [file,digest] of Object.entries(frozen.files)) {
        if (!/\.(?:md|json|txt|svg|png|jpe?g|webp|mmd|cpp|hpp|py)$/.test(file)) throw new Error('Unsupported tutorial artifact');
        const bytes=await artifact(directory,file);
        if (hash(bytes)!==digest) throw new Error(`Tutorial changed after rendering: ${relative}/${file}. Inspect and render it again.`);
        if(file.endsWith('.svg') && /<script|<foreignObject|\bon\w+\s*=|(?:href|src)\s*=|<!DOCTYPE|<!ENTITY/i.test(bytes.toString())) throw new Error('Tutorial SVG must be passive');
        assets.push({file:`${url}/${file}`,bytes});
      }
      if (frozen.files['lesson.json']!==hash(raw) || !frozen.files['lesson.md']) throw new Error('Tutorial render lacks its authoring manifest/lesson');
      let body=(await artifact(directory,'lesson.md')).toString().replace(/^# [^\n]*\n(?:\n)?/,'');
      const expected=(data.practice??[]);
      body=body.replace(/### Predict or explain: ([^\n]+)\n\n<details(?: data-practice-id="([a-z0-9-]+)")?><summary>Reveal worked answer and rubric<\/summary>\n\n[\s\S]*?<\/details>/g,(match,question,id)=>{
        const matches=expected.filter(item=>id ? item.id===id : item.question===question);
        const item=matches.length===1 ? matches[0] : null;
        if(!item) throw new Error('Unmapped tutorial practice');
        return `[[tutorial-practice:${item.id}]]`;
      });
      body=body.replace(/\[\[interactive:([a-z0-9-]+)\]\]/g,'[[tutorial-scenario:$1]]');
      body=body.replace(/\[([^\]]+)\]\(([^)]+)\)/g,(original,label,href)=>{
        if(/^(?:https?:|mailto:|#)/.test(href)) return original;
        if(!frozen.files[href]) throw new Error(`Tutorial link not in validated export: ${href}`);
        return `[${label}](/${url}/${href})`;
      });
      const metadata={practice:data.practice,scenarios:data.scenarios,project_id:project.id};
      const fields={title:data.title,description:data.question,updated:frozen.validated_at.slice(0,10),status:project.status.replaceAll('-',' ')+'; model progress separate',tags:['tutorials',project.id,'cpp','python'],sources:data.sources.filter(source=>source.kind==='external').map(source=>({title:source.title,url:source.url})),tutorial:metadata};
      const frontmatter=Object.entries(fields).map(([key,value])=>`${key}: ${JSON.stringify(value)}`).join('\n');
      records.push({file:`tutorials/${project.id.toLowerCase()}-${data.slug}-${data.revision}.md`,raw:`---\n${frontmatter}\n---\n\n${body}`});
    }
  }
  return {records,assets,index};
}
