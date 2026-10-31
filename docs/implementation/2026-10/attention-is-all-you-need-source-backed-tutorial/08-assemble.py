"""Assemble authored course metadata and derivative local reader records.

This does not implement the model. Teaching prose lives in the lesson sources.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OWNER = ROOT/'ai_projects'
DOCS = OWNER/'docs'
PROJECT = OWNER/'projects/transformer-labs/attention-is-all-you-need'
COURSE = DOCS/'tutorials/projects/attention-is-all-you-need/attention-lab/v1'


def save(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text if text.endswith('\n') else text+'\n')


def write_json(path,value): save(path,json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def git(repo,*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


# Preserve implementation, reference/notebook/captions and unrelated Bike source.
baseline=HERE/'preservation-before.json'
if not baseline.exists():
    selected=list(PROJECT.rglob('*.py'))
    selected=[p for p in selected if not any(x in p.parts for x in ('.venv','__pycache__'))]
    selected += [PROJECT/'notebooks/attention_colab.ipynb',PROJECT/'uv.lock',PROJECT/'pyproject.toml',PROJECT/'docs/sources.json']
    selected += list((PROJECT/'.sources/subtitles').glob('*'))
    bike=git(OWNER,'ls-files','projects/foundations/bike-demand-planner').splitlines()
    selected += [OWNER/name for name in bike if (OWNER/name).is_file()]
    write_json(baseline,{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(selected)) if p.is_file()})

# Capture a runnable source archive; exclude environments, raw captions and runs.
names=['README.md','LICENSE','pyproject.toml','uv.lock','.python-version','.gitignore','.ai-project.toml']
names += [str(p.relative_to(PROJECT)) for folder in ('src','tests','configs','notebooks') for p in sorted((PROJECT/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
prior_manifest=COURSE/'code/source-manifest.json'
if prior_manifest.exists():
    obsolete=set(json.loads(prior_manifest.read_text()))-set(names)
    for name in obsolete:
        for folder in (COURSE/'code',PROJECT/'docs/tutorial-assets/code'):
            file=folder/name
            if file.is_file() and file.resolve().is_relative_to(folder.resolve()):file.unlink()
    for folder in (COURSE/'code',PROJECT/'docs/tutorial-assets/code'):
        if folder.exists():
            for child in sorted(folder.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if child.is_dir() and not any(child.iterdir()):child.rmdir()
for name in names:
    dest=COURSE/'code'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PROJECT/name,dest)
write_json(COURSE/'code/source-manifest.json',{name:sha(COURSE/'code'/name) for name in names})
write_json(COURSE/'code/source-bundle.json',{'schema_version':1,'files':{name:{'sha256':sha(COURSE/'code'/name),'text':(COURSE/'code'/name).read_bytes().decode('utf-8')} for name in names}})
(COURSE/'code/attention-source.zip').unlink(missing_ok=True)

image_names=set()
for lesson in (COURSE/'lessons').glob('*.md'):
    image_names.update(re.findall(r'\(images/([^)]*)\)',lesson.read_text()))
for name in image_names:
    dest=COURSE/'images'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PROJECT/'runs/attention-visuals'/name,dest)

outline=[
('weighted','01-weighted-information.md','Explain a query-weighted value mixture and its proof boundary.','If weights change from [0.5,0.5] to [0.75,0.25] for values [2,4] and [6,8], what is retrieved?','The context becomes [3,5]: compute each coordinate with the same position weights. This is a toy retrieved vector, not a token prediction.','Keep values fixed and show both coordinate sums.'),
('setup','02-project-setup.md','Reproduce the existing environment and distinguish runnable components from placeholders.','The train.py placeholder exits successfully. Has the translation model been trained?','No. Its source is comment-only. Process success without model work is not a training result; inspect source and recorded outputs.','Separate environment readiness, executed component tests and task results.'),
('shapes','03-tensors-and-products.md','Trace score and context axes through rectangular batched attention.','For B=2, H=3, T=3, S=5 and d=4, what are score and context shapes?','Scores are (2,3,3,5). Context is (2,3,3,4), and merged output width is H*d=12. Keys/values share S; output follows T.','Name the contracted axes and output ownership.'),
('softmax','04-scores-and-softmax.md','Derive stable softmax and head-width scaling.','What changes if [0,ln(3)] becomes [1000,1000+ln(3)]? What if it becomes [0,2ln(3)]?','The common shift preserves [0.25,0.75]. Doubling score differences produces [0.1,0.9]. Subtract a maximum for stable exponentiation.','Separate additive invariance from multiplicative concentration.'),
('projections','05-learned-projections.md','Work learned Q/K/V maps, bias orientation and value mixing.','Hold Q/K fixed and add [1,-2] to every value row. What changes in a normalized nonempty row?','Weights are unchanged; context shifts by [1,-2] because each weight row sums to one. Output projection may transform that shift.','State the normalized/pre-dropout/nonempty conditions and distinguish context from final output.'),
('heads','06-multiple-heads.md','Explain separate/vectorized heads, correct reassembly and parameter counts.','Can head-major context be flattened directly to recover query-major features?','No. Transpose (B,H,T,d) to (B,T,H,d), then flatten H and d. Preserving element count alone does not preserve token ownership.','Trace one query row across all heads.'),
('padding','07-padding-and-empty-rows.md','Explain key masks, all-blocked behavior and perturbation isolation.','All keys are blocked. Why can the module return a nonzero vector?','The context is zero, but a biased output projection returns its bias. Also, a source key mask does not normally zero padded query outputs.','Separate visibility, context, output projection and loss masking.'),
('causal','08-causal-information.md','Intersect causal/padding restrictions and diagnose future leakage.','A query is position two and valid keys are [1,1,0,1]. Which keys may it read causally?','Keys zero and one only: [1,1,0,0]. The current key is padding and key three is future. Intersect validity and j<=i.','Use AND and identify both reasons for exclusion.'),
('cross','09-cross-attention.md','Trace source keys/values and target queries through rectangular attention.','Source length is five and target length three. Which length controls output, and which controls the mask?','Output has three query positions. The source key mask has length five. Queries come from tgt, keys/values from src.','Distinguish representation role from argument order.'),
('interface','10-reusable-attention.md','Explain the final API and explicit/SDPA/dropout execution branches.','Does fused_attn=True prove that GPU FlashAttention was used? What changes with need_weights=True?','No. SDPA selects a backend for actual device/inputs; current evidence is CPU. Requesting weights selects the explicit calculation in this wrapper.','Separate code path, backend identity and measured performance.'),
('gradient','11-gradients-and-learning.md','Derive and independently verify a query gradient and update.','At q=0 with keys [1,-1], values [2,6] and desired context five, which query direction decreases half squared error?','Decrease q: c=4, dL/dq=2, so gradient descent moves q negative and increases weight on value six. A step of 0.1 gives q=-0.2.','Explain all three slopes and separate this probe from real translation training.'),
('handoff','12-evidence-and-next-phase.md','State the evidence boundaries and concrete next data-phase requirements.','Do passing attention tests establish a trained translator or a completed learner review?','No. They establish exercised local component properties. Tokenizer/data, full model/training/evaluation and actual human explanations remain separate evidence.','Name at least one missing implementation artifact and one missing evidence class.'),
]
manifest=json.loads((COURSE/'lesson.json').read_text())
manifest['objectives']=[];manifest['chapters']=[];manifest['practice']=[]
for identifier,filename,objective,question,answer,rubric in outline:
    path=COURSE/'lessons'/filename
    text=path.read_text().replace('2.104449','2.104448').replace('1.296663','1.296664').replace('0.183164','0.183163')
    save(path,text)
    title=text.splitlines()[0].removeprefix('## ')
    manifest['objectives'].append({'id':identifier,'text':objective})
    manifest['chapters'].append({'id':identifier,'title':title,'content_path':'lessons/'+filename,'objective_ids':[identifier],'diagram_ids':[],'table_ids':[],'scenario_ids':[],'practice_ids':[identifier+'-check']})
    manifest['practice'].append({'id':identifier+'-check','question':question,'answer':answer,'rubric':rubric,'objective_ids':[identifier],'claim_ids':['component' if identifier not in ('setup','handoff') else 'scope']})
manifest['sources']=[]
code_revision=git(OWNER,'rev-parse','HEAD')
for identifier,name,title in [('attention-source','src/core/attention.py','Implemented attention module'),('attention-tests','tests/test_attention.py','Existing independent attention checks'),('visual-source','src/utils/attention_visuals.py','Original matrix fixtures and drawing code'),('visual-tests','tests/test_attention_visuals.py','Known answers for visual examples')]:
    file=PROJECT/name
    excerpt='evidence/'+identifier+'.txt'
    save(COURSE/excerpt,file.read_text())
    manifest['sources'].append({'id':identifier,'kind':'local','title':title,'path':str(file.relative_to(OWNER)),'start':1,'end':len(file.read_text().splitlines()),'sha256':sha(file),'excerpt_sha256':sha(COURSE/excerpt),'excerpt_path':excerpt,'base_sha':code_revision,'working_tree':bool(git(OWNER,'status','--porcelain','--',str(file.relative_to(OWNER)))),'repository_path':'ai_projects'})
for identifier,title,url,locator in [('paper','Attention Is All You Need, v7','https://arxiv.org/html/1706.03762v7','Sections 3.2.1-3.2.3; component mathematics and roles'),('sdpa','PyTorch 2.14 SDPA API','https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html','Mask semantics, dropout argument and backend selection')]:
    manifest['sources'].append({'id':identifier,'kind':'external','title':title,'url':url,'accessed_at':'2026-10-23','locator':locator,'proof_limit':'Primary reference; actual local behavior is checked separately. No task quality or hardware speed claim.'})
manifest['claims']=[{'id':'component','text':'The existing attention module implements the taught projection, head, mask and retrieval paths.','source_ids':['attention-source','attention-tests','visual-source','visual-tests'],'status':'verified','proof_limit':'Local source and exercised CPU checks; not trained translation, learner understanding or GPU performance.'},{'id':'scope','text':'The course covers setup and implemented attention; later translation data/model/training remain pending.','source_ids':['attention-source'],'status':'documented','proof_limit':'Source audit and explicit user scope; no later placeholder is implemented by this authoring work.'}]
manifest['examples']=[{'id':'arithmetic','language':'python','path':'examples/arithmetic.py','purpose':'Independent scalar/list arithmetic and finite differences for the worked examples','expected_output':{'weighted_context':[4,6],'rectangular_product':[[58,64],[139,154]],'projected_first_context':[2.104448,1.296664],'query_gradient':2.0,'post_step_loss':.183163,'parameter_count':360,'passed':True}}]
write_json(COURSE/'lesson.json',manifest)
(COURSE/'lessons/01-start.md').unlink(missing_ok=True)

# Direct lesson links are required by the existing frozen-asset discovery.
last=COURSE/'lessons/12-evidence-and-next-phase.md'
text=last.read_text().split('\n### Source and review downloads')[0]
text+='\n### Source and review downloads\n\n'
text+='The [captured source bundle](code/source-bundle.json) contains exact UTF-8 file contents and hashes for the implementation, tests, notebook, lock and honest later placeholders. Its [file hashes](code/source-manifest.json) identify each captured file. Save that JSON file and the [restoration helper](examples/restore_source.py) into the same folder, then run `python restore_source.py /path/to/a/new-directory` there to restore a runnable project. The helper refuses an existing destination and verifies every hash before writing. Do not restore over a working project. The [drawing module](code/src/utils/attention_visuals.py) is available separately.\n\n'
text+='Review: [setup learning points](project/learning-points/00-setup.md), [attention learning points](project/learning-points/01-attention.md), [setup comparisons](project/comparisons/00-setup.md), [attention comparisons](project/comparisons/01-attention.md), [implementation](project/implementation.md), [reference runs](project/reference-runs.md), [readiness](project/readiness.md), [practice](practice/explain-attention.md), [source research](research.md), [arithmetic probe](examples/arithmetic.py) and [component probe](examples/verify_component.py).\n'
save(last,text)
save(COURSE/'research.md',f'''# Sources, teaching choices and evidence boundaries

Inspected 23 October 2026. The learner requested the depth and reader organization of Bike Demand Planner, scoped to setup and already implemented attention. Later text/data/model/training material will be added with implementation evidence. The application source belongs to ai_projects at {code_revision}; this course is a separately authored Projects Docs artifact.

## Source hierarchy

The captured attention module and existing tests are the authority for current behavior. src/utils/attention_visuals.py supplies the controlled Q/K/V and output-projection fixtures plus the original drawing code. The Colab execution notebook lives in notebooks/ outside the core/data/training/utils packages, with entry scripts directly under src/. Its six operational code cells verify/extract source, install the locked environment, record runtime identity, run demo/tests and download logs. Tutorial prose and figures replace the former concept notebook. Local runner verification is separate from an actual hosted run. The runnable project source archive excludes raw captions, environments, data, weights and run trees.

The [original paper](https://arxiv.org/html/1706.03762v7) supplies the component definition and self/cross-attention roles; our explanations and figures use original toy tensors and the actual local implementation. The [PyTorch 2.14 SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) was checked for API semantics. No paper performance numbers are presented as our results.

The project follows Priyam Mazumdar's three supplied videos. Its reference code is pinned to PyTorch-Adventures revision 2f18a5470c4ef0a3d413dd7b18d07941f45779e9 under MIT. The [attention source video](https://www.youtube.com/watch?v=JXY5CmiK3LI), [data video](https://www.youtube.com/watch?v=UkBggwQsbF8) and [model/training video](https://www.youtube.com/watch?v=DOqg_QxInd8) retain their original roles. Videos 2-3 are future implementation inputs; no new media is generated here.

## Important local choices

The final API preserves src/tgt and q_proj/k_proj/v_proj/out_proj names. True/1 means an allowed key. Source padding and causality combine explicitly; cross-attention rejects the causal flag. Entirely blocked rows have zero context; output bias can remain. Returned inspection weights precede dropout. Efficient mode uses SDPA with evaluation dropout disabled, while requesting weights selects explicit calculations.

These details are source-specific contracts. Q/K/V notation, normalized weighted sums and chain-rule arithmetic have mathematical meaning beyond the implementation, but they do not establish useful language representations or training success. The head-width scaling rationale names its independence/variance assumptions rather than treating them as measured properties.

## Authored evidence and departures

The twelve-lesson route is a new source-backed explanation of the existing component. Standard-library scalar/list probes check selected numbers independently; PyTorch probes compare them with frozen actual module behavior. Existing tests include an independent PyTorch MultiheadAttention reference. The former concept notebook was replaced with a Colab project runner; its operational cells are checked locally with simulated upload/download and real commands. Attention arithmetic and fixed-seed demo outputs remain unchanged; selected figures are copied with source provenance.

The route begins with a weighted result, then setup, axes, normalization, projections, heads, padding, causality, cross-attention, public API, gradients and evidence/handoff. The later full model is described only to explain the component boundary. Learner understanding, hosted recovery and translation quality remain unassessed. Runtime figures are synthetic/random demonstrations, not trained alignments or hardware benchmark plots.
''')

# Integrate the named application tutorial without altering P00-P19 sequence.
index_path=DOCS/'tutorials/index.json'
index=json.loads(index_path.read_text())
entry={'id':'attention-is-all-you-need','title':'Attention Is All You Need','status':'setup-and-attention-tutorial-available','lessons':['projects/attention-is-all-you-need/attention-lab/v1']}
prior=next((p for p in index['projects'] if p['id']==entry['id']),None)
if prior:prior.update(entry)
else:index['projects'].append(entry)
write_json(index_path,index)

# Private project records use one generated continuous view of the authored course.
doc=PROJECT/'docs'
source_for=lambda folder,target:Path(os.path.relpath(COURSE/target,folder)).as_posix()
def rebase(body,folder):
    def link(match):
        target=match[1]
        if target.startswith(('http:','https:','#')):return match.group(0)
        if target.startswith('source:'):
            source=next(s for s in manifest['sources'] if s['id']==target[7:])
            if source['kind']=='external':return ']('+source['url']+')'
            target=source['excerpt_path']
        if (COURSE/target).suffix!='.md':
            private=doc/'tutorial-assets'/target
            private.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(COURSE/target,private)
            return ']('+Path(os.path.relpath(private,folder)).as_posix()+')'
        return ']('+source_for(folder,target)+')'
    body=re.sub(r'\]\(([^)]+)\)',link,body)
    for practice in manifest['practice']:
        block='\n### Predict or explain: '+practice['question']+'\n\n**Worked answer:** '+practice['answer']+'\n\nCheck: '+practice['rubric']+'\n'
        body=body.replace('[[practice:'+practice['id']+']]',block)
    return body

records=[('00_project-setup',[outline[1]],'Setup and source boundaries'),('01_attention-mechanism',[outline[0],*outline[2:]],'Attention: from weighted values to gradients')]
for slug,chapters,title in records:
    folder=doc/'implementation'/slug
    folder.mkdir(parents=True,exist_ok=True)
    setup=slug.startswith('00')
    # The attention record is now independently authored with code videos.
    # Reassembling the shared course must preserve its teaching and media pins.
    if not setup:
        continue
    key='00-setup' if setup else '01-attention'
    start=chapters[0][1]
    save(folder/'00-resume.md',f'# {title} — current checkpoint\n\nReviewed 23 October 2026. Setup and attention are implemented locally; later data/model/training remain pending. The current 35 existing tests and CPU demo pass. Understanding review remains open.\n\nRead the [understanding notes](02-notes-to-understanding.md), [learning points](01-learning-points.md), [comparisons](04-comparisons.md) and [tracker](05-tracker.md). The [ordered tutorial]({source_for(folder,"lessons/"+start)}) is the canonical reading route.\n')
    for name,source in [('01-learning-points.md','project/learning-points/'+key+'.md'),('04-comparisons.md','project/comparisons/'+key+'.md')]:
        save(folder/name,rebase((COURSE/source).read_text(),folder))
    body=f'# {title} — notes to understanding\n\nThis continuous local view uses the same authored explanations as the [ordered Attention tutorial]({source_for(folder,"lessons/"+start)}). It covers existing code and synthetic component probes. Later data/model/training remain pending; no learner result is inferred.\n\n'
    for row in chapters:body+=rebase((COURSE/'lessons'/row[1]).read_text(),folder)+'\n\n'
    save(folder/'02-notes-to-understanding.md',body)
    save(folder/'03-tasks.md',f'# Review the {"setup" if setup else "attention component"}\n\n1. Open the existing source and the matching worked notes; do not recreate the project.\n2. Reproduce the documented example/test command in the locked environment.\n3. Explain one result and predict a changed case before revealing the answer.\n4. Compare the actual output with that prediction and record only evidence that really exists.\n\n{"Next: trace the first weighted result and Q/K/V roles." if setup else "Next implementation, when continuing: define and implement the Part 2 tokenizer/data/batch contract. No later model code is implemented by this record."}\n')
    save(folder/'05-tracker.md','# Implementation and understanding tracker\n\n| Item | State | Evidence boundary |\n| --- | --- | --- |\n| Existing setup | Available | Locked local package |\n| Attention implementation | Implemented and tested | 35 current CPU checks and demo |\n| Written worked tutorial | Authored; reader verification recorded in the delivery package | No new model implementation |\n| Learner explanation | Open | No inferred answer/mastery |\n| Tokenizer/data and full model/training | Pending | Comment-only files |\n| Hosted recovery | Pending | No hosted execution |\n')

save(doc/'implementation/README.md','# Attention working records\n\n[Setup](00_project-setup/00-resume.md) and [implemented attention](01_attention-mechanism/00-resume.md) have separate records. The worked maths/code route is in the understanding notes. Later phases get expanded records as they are implemented.\n')
phases=[('00-project-setup','Local setup and source boundaries','Implemented locally','00_project-setup'),('01-attention','Attention, masks, heads and gradients','Implemented and checked locally','01_attention-mechanism'),('02-tokenizer-and-data','English/French tokenizers and batches','Planned; not implemented',None),('03-transformer-and-training','Full Transformer, training, translation and recovery','Planned; not implemented',None)]
lines=['# Attention Is All You Need — phase route','','| Phase | State | Handoff |','| --- | --- | --- |']
for slug,title,state,record in phases:
    folder=doc/'phases'/slug
    handoff='Reproduce the existing environment, then inspect attention.' if slug.startswith('00') else 'Review one complete attention trace, then define the data contract.' if slug.startswith('01') else 'Pin data/tokenizers, print shifted target batches and verify masks.' if slug.startswith('02') else 'Assemble the model only after data contracts; retain training/evaluation/recovery evidence.'
    text=f'# Phase {slug[:2]} — {title}\n\nState: **{state}**.\n\n{handoff}\n\n'
    if record:text+=f'The [current implementation record](../../implementation/{record}/00-resume.md) contains learning points, understanding notes, tasks, comparisons and tracker.\n'
    else:text+='This is an overview of future work. Source files are placeholders; there is no detailed implemented walkthrough or claimed result for this phase yet.\n'
    save(folder/'README.md',text)
    lines.append(f'| [{title}]({slug}/README.md) | {state} | {handoff} |')
save(doc/'phases/README.md','\n'.join(lines)+'\n')
save(doc/'README.md',f'''# Attention Is All You Need — learning and working records

Start with the [in-depth Attention tutorial]({source_for(doc,'lessons/01-weighted-information.md')}). Its twelve ordered lessons explain the setup and already implemented attention component through worked maths, original figures, actual source traces, independent checks, changed cases and revealable practice. Later tokenizer/data/model/training lessons are added with implementation evidence.

The [phase route](phases/README.md) is separate from the [working records](implementation/README.md). Open [setup](implementation/00_project-setup/02-notes-to-understanding.md) or [the complete attention understanding notes](implementation/01_attention-mechanism/02-notes-to-understanding.md) for seven focused code/value walkthroughs, each preceded by its captioned code video. The learning points revisit the reasoning separately. Each record has separate learning points, tasks, comparisons and tracker.

Current software: locked local environment, reusable attention, demo and tests. All 35 existing CPU tests pass. The six-cell Colab runner lives in notebooks/attention_colab.ipynb; local verification runs the exported source, demo and tests through its operational cells. Hosted execution remains unverified. Tokenizer/data/model/trainer/checkpoint/inference/BLEU remain placeholders. Learner understanding and actual hosted recovery remain open.

Original [source references](SOURCES.md) and [Colab requirements](COLAB.md) remain available. This project-local study material stays ignored; the deliberately authored shared tutorial is tracked by Projects Docs.
''')
ignore=PROJECT/'.gitignore'
text=ignore.read_text()
for pattern in ['/docs/README.md','/docs/implementation/','/docs/phases/','/docs/tutorial-assets/']:
    if pattern not in text.splitlines():text+='\n'+pattern
save(ignore,text)
print(json.dumps({'course':str(COURSE),'lessons':len(outline),'images':len(image_names),'local_records':2,'source_files':len(names)},indent=2))
