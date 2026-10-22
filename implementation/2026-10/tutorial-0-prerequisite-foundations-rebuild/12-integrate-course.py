"""Reconcile the completed topic lessons with the existing course manifest."""
from pathlib import Path
import json,re
PACKAGE=Path(__file__).resolve().parent
COURSE=PACKAGE.parents[2]/'tutorials/projects/p00/transformer-foundations/v1'
LESSONS=[
('route','01-prediction.md'),('workspace','02-workspace.md'),('numbers','03-numbers.md'),
('probability','04-probability.md'),('model','08-model.md'),('initialization','06-initialization.md'),
('autograd','04-autograd.md'),('losses','08-losses.md'),('optimizers','09-optimizers.md'),
('steps','10-training-steps.md'),('data','06-data.md'),('tokenizer','05-tokenizer.md'),
('embeddings','13-embeddings.md'),('position','14-position.md'),('attention','07-attention.md'),
('masks','16-masks.md'),('multihead','17-multihead.md'),('ffn','18-positionwise-ffn.md'),
('residual','19-residual.md'),('normalization','20-normalization.md'),('regularization','21-regularization.md'),
('seq2seq','22-encoder-decoder.md'),('training','09-training.md'),('recovery','10-recovery.md'),
('handoff','11-colab-and-handoff.md')]
manifest=json.loads((COURSE/'lesson.json').read_text())
manifest['title']='Tutorial 0 — Deep learning foundations for Attention Is All You Need'
manifest['question']='Which mathematical, representation, architecture and training foundations explain every core operation in the original Transformer?'
manifest['prerequisites']=['Advanced Python programming is assumed; general Python syntax/classes/collections are omitted.','Mathematical notation, derivatives, probability and PyTorch contracts are developed through original worked examples and exercises.']
new_sources=json.loads((PACKAGE/'topic-sources.json').read_text())
new_sources += [('d2l-seq2seq','D2L1.0.3: sequence-to-sequence learning','https://d2l.ai/chapter_recurrent-modern/seq2seq.html','Teacher forcing and conditional sequence learning'),('d2l-beam-search','D2L1.0.3: beam search','https://d2l.ai/chapter_recurrent-modern/beam-search.html','Greedy/beam alternatives and stopping context')]
existing={s['id'] for s in manifest['sources']}
for sid,title,url,locator in new_sources:
 if sid not in existing:
  manifest['sources'].append({'id':sid,'kind':'external','title':title,'url':url,'accessed_at':'2026-10-12','locator':locator,'proof_limit':'Primary source/API/teaching context inspected; original calculations and independently executed probes are separate evidence. No learner mastery, benchmark or hosted-recovery claim.'})
  existing.add(sid)
# The original architecture and later comparisons have deliberately separate claims.
for item in [
 {'id':'original-transformer','text':'The original post-norm/ReLU/Adam encoder–decoder and its explicit attention/position/training equations are distinguished from later variants.','source_ids':['transformer-paper'],'status':'documented','proof_limit':'Original source design; small course fixtures do not reproduce its corpus/scale/results.'},
 {'id':'mask-contract','text':'SDPA allowed-True and MultiheadAttention blocked-True Boolean conventions are different; empty rows require a declared tested policy.','source_ids':['torch-sdpa','torch-mha'],'status':'documented','proof_limit':'API conventions verified; zero-empty-row behavior is this educational explicit-probe policy, not a universal backend claim.'},
 {'id':'later-variants','text':'AdamW and pre-norm are comparisons with different update/forward contracts.','source_ids':['adamw-paper','prenorm-paper'],'status':'documented','proof_limit':'Recognition/comparison; no new full variant model or training benchmark.'}]:
 if item['id'] not in {c['id'] for c in manifest['claims']}:manifest['claims'].append(item)
boards={
'attention-flow':('Attention matches positions, then transports values','Separate feature contraction, key-position normalization and value aggregation.', [('Q[T,d] and K[S,d]','Queries/keys can have different sequence lengths.'),('Scores[T,S] / sqrt(d)','Each cell compares query/key feature coordinates.'),('Row weights A[T,S]','Normalize across allowed key positions.'),('A @ V → O[T,dv]','Values supply transported information.')],[(0,1,'dot products'),(1,2,'softmax over S'),(2,3,'weighted sum')]),
'mask-contracts':('Three visibility contracts','Causality, source padding and query output gates have separate meanings.', [('Encoder key validity','Each real query sees real source keys.'),('Decoder prefix causality','Query t sees keys s≤t in shifted input.'),('Cross attention[T,S]','Decoder queries see real encoder memory.'),('Empty/padded queries','Zero policy/gating is explicit, not implicit.')],[]),
'head-shapes':('Split and merge head axes','Projection width stays D=H×d; transpose and reshape are distinct.', [('Projected[B,T,D]','Learned Q/K/V maps combine original features.'),('Split[B,T,H,d]','Divide projected feature coordinates into heads.'),('Heads[B,H,T,d]','Transpose position/head axes for attention.'),('Scores[B,H,T,S]','Normalize over S, transport values.'),('Merge[B,T,D] → Wo','Inverse transpose then merge features.')],[(0,1,'reshape'),(1,2,'transpose'),(2,3,'attention'),(3,4,'values / merge')]),
'residual-path':('Residual identity and learned branches','Both branches have the same decoder/encoder stream shape.', [('Input x[B,T,D]','Identity branch preserves the input.'),('Sublayer F(x)[B,T,D]','Learned attention or FFN output.'),('Sum x+F(x)','Upstream derivative has direct and sublayer paths.'),('Post-norm LN(sum)','Original order normalizes after residual addition.')],[(0,1,'learned branch'),(0,2,'identity branch'),(1,2,'residual addition'),(2,3,'original order')]),
'seq2seq-contract':('Parallel training, autoregressive prediction','Source memory, shifted target inputs and next-token labels are different streams.', [('Source IDs[B,S]','Encoder sees valid source tokens.'),('Encoder memory[B,S,D]','Cross-attention keys/values come from here.'),('Shifted prefix[B,T]','BOS and previous target tokens supply decoder input.'),('Decoder states[B,T,D]','Causal self-attention and source retrieval.'),('Logits[B,T,V]','Supervise next tokens; ignore padded labels.')],[(0,1,'encode'),(1,3,'K / V'),(2,3,'Q / causal prefix'),(3,4,'vocabulary map')])}
for did,(title,description,nodes,edges) in boards.items():
 if did in {d['id'] for d in manifest['diagrams']}:continue
 entity_ids=[];edge_ids=[]
 for i,(label,desc) in enumerate(nodes):
  eid=f'{did}-{i+1}';entity_ids.append(eid);manifest['entities'].append({'id':eid,'label':label,'description':desc,'claim_ids':['numbers']})
 for i,(src,dst,label) in enumerate(edges):
  eid=f'{did}-edge-{i+1}';edge_ids.append(eid);manifest['relationships'].append({'id':eid,'from':entity_ids[src],'to':entity_ids[dst],'label':label,'claim_ids':['numbers']})
 manifest['diagrams'].append({'id':did,'title':title,'description':description,'entity_ids':entity_ids,'relationship_ids':edge_ids,'layout':'grid'})
practices=json.loads((PACKAGE/'topic_practice.json').read_text())
manifest['practice']=[{'id':pid,'question':q,'answer':a,'rubric':rubric,'objective_ids':[oid],'claim_ids':['numbers']} for oid,pid,q,a,rubric in practices]
manifest['objectives']=[];manifest['chapters']=[]
for position,(oid,name) in enumerate(LESSONS,1):
 source=(COURSE/'lessons'/name).read_text()
 title=source.splitlines()[0].removeprefix('## ')
 if not title.startswith(f'Lesson {position} —'):raise ValueError((name,title,position))
 objective=title.split(' — ',1)[1]
 manifest['objectives'].append({'id':oid,'text':f'Explain, calculate and check {objective.lower()}.'})
 ids={kind:re.findall(r'\[\['+kind+r':([a-z0-9-]+)\]\]',source) for kind in ['diagram','table','scenario','practice']}
 if not ids['practice']:raise ValueError(f'Missing practice: {name}')
 manifest['chapters'].append({'id':oid,'title':title,'content_path':'lessons/'+name,'diagram_ids':ids['diagram'],'table_ids':ids['table'],'scenario_ids':ids['scenario'],'practice_ids':ids['practice'],'objective_ids':[oid]})
used_diagrams={i for c in manifest['chapters'] for i in c['diagram_ids']}
manifest['diagrams']=[d for d in manifest['diagrams'] if d['id'] in used_diagrams]
entity_ids={i for d in manifest['diagrams'] for i in d['entity_ids']};edge_ids={i for d in manifest['diagrams'] for i in d['relationship_ids']}
manifest['entities']=[e for e in manifest['entities'] if e['id'] in entity_ids]
manifest['relationships']=[e for e in manifest['relationships'] if e['id'] in edge_ids]
used_scenarios={i for c in manifest['chapters'] for i in c['scenario_ids']}
manifest['scenarios']=[s for s in manifest['scenarios'] if s['id'] in used_scenarios]
(COURSE/'lesson.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'lessons':len(manifest['chapters']),'sources':len(manifest['sources']),'practices':len(manifest['practice']),'diagrams':len(manifest['diagrams'])}))
