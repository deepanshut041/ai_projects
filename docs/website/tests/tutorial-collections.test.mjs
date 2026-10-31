import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {tutorialRecords} from '../scripts/tutorials.mjs';
import {readRecords} from '../scripts/content.mjs';
import {projectLibrary} from '../scripts/projects.mjs';
import {tutorialCollectionLibrary,tutorialCollectionPage} from '../scripts/tutorial-collections.mjs';

const docs=fileURLToPath(new URL('../../',import.meta.url));
test('every authored tutorial has exactly one real collection and keeps its URL and course order',async()=>{
 const tutorials=await tutorialRecords(docs);
 for(const base of ['/','/project_docs/']){
  const records=await readRecords(path.join(docs,'website/content'),base,tutorials.records);
  const projects=await projectLibrary(path.join(docs,'../projects'),base);
  const before=new Map(records.map(r=>[r.id,r.url]));
  const catalog=tutorialCollectionLibrary(records,projects.groups,base);
  assert.deepEqual(catalog.groups.map(g=>g.id),['foundations','transformer-labs','language-models','quantization-lab','attention-labs']);
  const courses=catalog.groups.flatMap(g=>g.courses);
  const registered=records.filter(r=>r.tutorial?.kind==='overview');
  assert.equal(new Set(courses.map(r=>r.id)).size,registered.length);
  const expectedOwners={P00:'transformer-labs',P01:'language-models','bike-demand-planner':'foundations','attention-is-all-you-need':'transformer-labs','document-assistant':'quantization-lab','attention-labs':'attention-labs'};
  for(const group of catalog.groups)for(const course of group.courses)assert.equal(group.id,expectedOwners[course.tutorial.project_id]);
  for(const record of records){assert.equal(record.url,before.get(record.id));if(record.tutorial)assert.ok(record.tutorial.collection.url.startsWith(base+'tutorials/'));}
  const transformer=catalog.groups.find(g=>g.id==='transformer-labs');
  assert.deepEqual(transformer.courses.filter(c=>Number.isInteger(c.tutorial.courseNumber)).map(c=>c.tutorial.courseNumber),[0,1]);
  for(const group of catalog.groups){
   const html=tutorialCollectionPage(group,base);
   for(const course of group.courses)assert.ok(html.includes(`href="${course.url}"`));
  }
 }
});
test('new projects inherit their collection and unknown tutorial ownership cannot silently disappear',()=>{
 const groups=[{id:'foundations',title:'Foundations',description:'',url:'/projects/foundations/',projects:[{id:'new-project'}]}];
 const make=(id,kind='overview')=>({id:kind,url:'/old/'+kind+'/',search:'',tutorial:{project_id:id,kind}});
 const records=[make('new-project'),make('new-project','chapter')];
 const catalog=tutorialCollectionLibrary(records,groups,'/');assert.equal(catalog.groups[0].courses.length,1);
 assert.equal(records[1].tutorial.collection.id,'foundations');
 assert.throws(()=>tutorialCollectionLibrary([make('unknown')],groups,'/'),/Assign tutorial project unknown/);
});

test('standalone frozen tutorials retain their collections without dangling project links',()=>{
 const ids=['P00','P01','bike-demand-planner','attention-is-all-you-need','document-assistant','attention-labs'];
 const records=ids.map(id=>({id,url:'/records/'+id+'/',search:'',tutorial:{project_id:id,kind:'overview'}}));
 const catalog=tutorialCollectionLibrary(records,[],'/');
 assert.equal(catalog.groups.flatMap(g=>g.courses).length,ids.length);
 for(const group of catalog.groups){assert.equal(group.projectDocsURL,null);assert.ok(!tutorialCollectionPage(group,'/').includes('href="null"'));}
});
