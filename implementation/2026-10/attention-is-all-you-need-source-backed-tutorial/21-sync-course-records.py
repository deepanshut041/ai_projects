"""Keep the revised chapters, review pages and private notes in agreement."""
from pathlib import Path
import json
import os
import re
import shutil

ROOT = Path('/mnt/projects/ai')
COURSE = ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
NOTES = ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need/docs/implementation/01_attention-mechanism'
BEFORE = ROOT/'.ai/attention-course-review/2026-10-26/before/course'
data=json.loads((COURSE/'lesson.json').read_text())
for chapter in data['chapters']:
    chapter['title']=(COURSE/chapter['content_path']).read_text().splitlines()[0][3:]
data['title']='Attention: from the translation problem to the actual computation'
data['question']='How does a translation model share information between positions, why does it use queries, keys, values, heads and masks, and how does the actual attention code implement those decisions?'
data['prerequisites']=['Read vectors and matrix products; recognize a model parameter, gradient and loss. Python syntax is not taught here.']
data['objectives']=[{'id':c['id'],'text':c['title']} for c in data['chapters']]
data['media']={};data['scenes']=[];data['approval']=None
sources=[
 ('illustrated-transformer','The Illustrated Transformer — Jay Alammar','https://jalammar.github.io/illustrated-transformer/','High-level model, tensors, scalar/matrix attention, heads, decoder and training; editorial comparison only'),
 ('annotated-transformer','The Annotated Transformer — Harvard NLP, inspected 2018 version','https://nlp.seas.harvard.edu/2018/04/03/attention.html','Model Architecture, Attention, Applications of Attention, Training and decoding; historical implementation, not current install instructions'),
 ('d2l-attention','Dive into Deep Learning — Q/K/V','https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html','Attention pooling roles and visualization; editorial comparison'),
 ('d2l-bahdanau','Dive into Deep Learning — Bahdanau attention','https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html','Encoder/decoder bottleneck and adaptive source retrieval; editorial comparison'),
 ('bahdanau','Neural Machine Translation by Jointly Learning to Align and Translate','https://arxiv.org/abs/1409.0473','Abstract and proposed source retrieval versus fixed-length source representation')]
existing={s['id'] for s in data['sources']}
for ident,title,url,locator in sources:
    if ident not in existing:data['sources'].append({'id':ident,'kind':'external','title':title,'url':url,'accessed_at':'2026-10-26','locator':locator,'proof_limit':'Primary authored source for motivation or teaching comparison; no local task result or educational-effectiveness measurement.'})
for s in data['sources']:
    if s['id']=='paper':s['accessed_at']='2026-10-26';s['locator']='Sections 1–4: motivation, architecture, attention roles, positions and complexity'

checks={
'weighted':(
 'Keep input row 1 fixed and change row 0. Can h_1=f(x_1) change? What new path does attention provide?',
 'The row-independent h_1 cannot change because its expression does not use x_0. Attention can make output row 1 depend on value 0 and its score against key 0, subject to the allowed mask. This provides a possible dependency, not a guarantee that every input perturbation changes the result.',
 'Identify the actual input-to-output path, rather than saying only that attention adds context.'),
'softmax':(
 'Hold K and V fixed but replace q=[1,0] with q=[0,0]. Explain the entire consequence.',
 'All three dot products become zero, so softmax assigns 1/3 to each key and the context becomes [4/3,4/3]. The available content is unchanged; the receiving comparison now treats every key equally.',
 'Follow query → scores → normalized weights → both value features.'),
'shapes':(
 'You reorder K rows but leave V in its old order. Shapes still match. Why is the calculation wrong?',
 'A weight calculated against key j now multiplies content from a different source row. Keys and values must move together. With paired K/V permutations the summed result for fixed Q is unchanged; permuting Q as well reorders output receivers. Positional information is a separate input.',
 'Explain source pairing and receiver ownership; shape agreement alone is insufficient.'),
'projections':(
 'Halve W_V for every input. Which quantities change, and is this the same kind of state as an attention-weight row?',
 'For fixed X and Q/K maps, V and C halve while scores and A stay the same. W_V is a persistent shared parameter; A is recomputed from each input. The parameter change therefore applies to subsequent examples as well.',
 'Separate content from selection, and shared model parameters from input-dependent activations.'),
'heads':(
 'Would giving head 0 only the first half of a sentence implement the head split shown here?',
 'No. The shown split assigns projected feature groups to heads while each head retains every permitted source position. Different heads can normalize different distributions over the same positions. Splitting sentence positions would remove information paths instead.',
 'Name the split axis, retained source axis and independently normalized distributions.'),
'padding':(
 'If a padded value is [0,0], can it still change real-token outputs when its key is not masked?',
 'Yes. Its score still contributes an exponential to the softmax denominator, reducing the weights on real keys. Exclude it before normalization. The key mask does not delete query rows, and target-loss masking remains separate.',
 'Trace denominator contamination instead of looking only for a nonzero padded contribution.'),
'causal':(
 'At decoder input position 1 containing Nous, which token is the label, and why must input position 2 be blocked?',
 'The label is lisons; input position 2 already contains lisons during teacher-forced training. Reading it would reveal the answer unavailable at generation time. Allow input keys 0 and 1, and shift labels separately. Padding is a different restriction.',
 'Identify the exact cheating path and reconcile the current-position allowance with next-token labels.'),
'cross':(
 'A decoder has five receiving rows and the encoder has three memory rows. What are the weight/output position axes, and may the last source row be read by target row 0?',
 'Weights have shape (B,H,5,3); output has shape (B,5,d_model) in this class. All valid source rows may be read even at target row 0, because the source was available before generation. The target query must already obey its own prefix boundary.',
 'Trace Q versus K/V origin and avoid imposing target time on source indices.'),
'gradient':(
 'Does translation training need a hand-written desired context vector or an attention-map label?',
 'Ordinary next-token training uses reference token labels. Cross-entropy gradients flow from vocabulary logits through decoder representations into attention parameters. The course’s [2,0] desired context is an isolated derivative probe. A decrease in that probe loss does not establish useful translation.',
 'Connect actual task supervision to shared projections and distinguish the local probe objective.'),
'interface':(
 'The class is configured with fused_attn=True, but the call requests need_weights=True. Which path runs, and what must be matched to compare paths?',
 'Requesting weights selects the explicit helper. A valid comparison copies the same parameters and fixes input, masks, dtype and evaluation mode. Evaluation supplies zero dropout to the library path. Two random layer initializations are not the same function.',
 'Follow the actual branch condition and identify controlled comparison inputs.'),
'handoff':(
 'Why does the Transformer retain a position-wise feed-forward layer after adding attention, and where does order come from?',
 'Attention shares information between permitted positions. The feed-forward layer applies a nonlinear feature transformation within each row, while residual/norm sublayers organize updates. Positional input supplies order information that unrestricted attention alone lacks. These components have distinct responsibilities.',
 'Explain the separate operations in the architecture rather than treating attention as the complete model.'),
'setup':(
 'The demo and 35 tests pass. What has been checked, and what would still be needed to claim translation quality?',
 'Those runs check the implemented component, known arithmetic and exercised interfaces. Translation quality needs actual tokenizer/data, full encoder/decoder, trained parameters, decoding and fixed evaluation. A placeholder script exiting successfully or local Colab simulation supplies none of those results.',
 'Separate component evidence from task execution and held-out evaluation.')}
for p in data['practice']:
    key=p['id'].removesuffix('-check')
    p['question'],p['answer'],p['rubric']=checks[key]
(COURSE/'lesson.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

(COURSE/'project/implementation.md').write_text('''# Read the model problem, then trace the computation

Start with [the whole picture](lessons/01-weighted-information.md): the translation process, its two jobs and attention's place in them. The opening map shows how each pass adds detail. Then zoom into one source-side information-sharing calculation and build on it through calculated weights, all-position matrices, learned maps, heads, availability boundaries, target/source relationships, task gradients and the actual class. The final chapter returns those understood operations to the Transformer.

The conceptual route is chapters 1–11; the last setup page is a run reference. Each numerical program is complete and downloadable. The three-row fixture is deliberately small so every contribution can be inspected; its row labels track ownership without pretending the coordinates were learned from text.

The [whole-course comparison](project/teaching-review.md) explains the structural problems found in the previous version and how this route addresses them. Its assessment covers all chapters, not just the opening paragraph.

All Attention companions were removed from this course and its project understanding notes on 26 October 2026. Recoverable original media remains local. No replacement video production is part of this revision.

The existing implementation remains the attention component. Tokenizer/data, complete model, training, decoding/evaluation and hosted recovery remain future work. [Readiness](project/readiness.md) records that boundary.
''')

(COURSE/'project/learning-points/01-attention.md').write_text('''# Reconstruct the attention design from the translation task

A translation model receives source tokens and predicts each next target token from that source and its available target prefix. Token IDs select embedding rows; positional input supplies order. Encoder and decoder layers transform those rows into representations that support vocabulary predictions.

Applying f(x_i) independently to each row cannot make output i depend on a changed neighboring input. The missing capability is communication between positions. A common average communicates one shared summary, while attention gives each receiver its own input-dependent mixture of available source values. This explains the operation before naming its parts.

The query is derived from the receiving representation. Each available representation supplies a key for comparison and a paired value for contribution. Their separate projection matrices can adapt selection and content independently. The matrices are shared model parameters; the calculated Q/K/V, scores and attention weights are activations for the current input. Useful comparison features must be learned rather than assigned by names.

In the fixture, query [1,0] compares with keys [1,0], [0,1], [1,1]. The scaled scores normalize to [0.401112,0.197776,0.401112]. Applying them to [2,0], [0,2], [2,2] gives [1.604448,1.197776]. That result is a gathered feature row. Its two numbers are not a token prediction, and the labels on the rows do not give their coordinates linguistic meaning.

Stacked queries preserve one result per receiver. Score cell (i,j) is a comparison from receiver i to source j. Row-wise softmax gives each receiver a distribution; multiplying by V sums out the source axis. Batch examples remain independent. Paired permutations of source keys and values preserve a fixed receiver’s result; unrestricted self-attention needs positional input to represent order.

One head uses one weight per source row across all its value features. Several heads let projected feature groups gather independently weighted mixtures. Every head still sees all permitted positions. Restoring the position axis before joining heads is essential to preserve row ownership.

Masks are information boundaries. Padding removes storage-only source rows before normalization; even a zero padded value otherwise consumes denominator weight. Decoder causality blocks future reference tokens that would reveal the next-token label during training. A causal mask requires the correct shifted input/label contract. Cross-attention instead lets target receivers read all valid encoded source rows, whose indices do not represent future target time.

Reference-token cross-entropy supplies gradients through the vocabulary projection and decoder into attention. The course’s explicit squared-context probe isolates one part of that derivative path. Its falling loss explains a local update, rather than establishing a trained translator.

The actual class packages projection, head layout, allowed masks, explicit/library computation and output mapping. The surrounding Transformer still needs positional input, residual/norm/FFN sublayers and vocabulary prediction. Return to [the execution trace](lessons/10-reusable-attention.md) for exact source behavior and [the architecture](lessons/12-evidence-and-next-phase.md) for component boundaries.
''')

(COURSE/'project/comparisons/01-attention.md').write_text('''# Compare designs by their information paths

Use these comparisons after [the main route](lessons/01-weighted-information.md). The important difference is which inputs can influence which outputs, through which calculation.

| Decision | What it permits or constrains | Consequence to check |
| --- | --- | --- |
| Independent row transform | Output i reads only input i | A neighbor-only change cannot alter that row |
| Common average | Every receiver gets the same mixture | Communication exists, but the mixture is not receiver-dependent |
| Query-dependent attention | Each receiver derives weights against available keys | A changed query can redistribute the same available values |
| Recurrent communication | State passes sequentially through positions | Context is possible, with a position-dependent computation chain |
| Separate Q/K/V maps | Selection and contributed content can differ | Halve only W_V: context halves with identical weights |
| Multiple heads | Projected feature groups have separate distributions | All heads keep every allowed position; joining must preserve receivers |
| Source padding mask | Storage-only key rows are absent from normalization | A padded-row perturbation cannot affect real receivers |
| Target causal mask | Future target inputs cannot reveal labels | Test future perturbation while keeping those keys real |
| Cross-attention | Target states retrieve from a separate source memory | Rectangular weights; source mask length follows memory |
| Explicit/library execution | Two implementations of the same formula | Match exact state and evaluation inputs before comparing |

A mask’s appearance does not establish isolation, and matching tensor shapes do not establish correct key/value pairing. Perturbations and row ownership make those properties concrete.

Parallel attention still has n² pair comparisons for a length-n sequence. Parameter sharing across positions does not remove activation work, and increasing heads at fixed model width narrows each head. These are engineering tradeoffs, not automatic quality improvements.
''')

(COURSE/'project/teaching-review.md').write_text('''# Why the whole tutorial was rebuilt

On 26 October 2026, the learner rejected the tutorial and explicitly asked for an entire-course comparison with good human-authored attention tutorials. The review read all twelve chapters, their code, practice, supporting pages and the rendered opening.

[Jay Alammar’s Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) establishes the translation system and successively opens its components. [Harvard NLP’s Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), inspected in its historical 2018 version, connects architectural responsibilities to implementation. [D2L’s attention chapters](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) motivate retrieval through a specific encoder/decoder limitation. These are useful teaching patterns, not text or figures copied into this course.

The previous route mostly formed an arithmetic ladder: supplied values, supplied query, matrices, projections, heads and masks. Its local numbers could be correct while the reader still lacked the task, input ownership, inadequate design and reason for each operation. The full model context appeared too late. A large exact source block did not itself explain execution, and figures available on disk were mostly absent from the pages.

The replacement establishes translation and token-to-vector input first, demonstrates the missing dependency of independent row processing, then derives input-dependent gathering. It introduces Q/K/V as transformations of owned input rows, interprets matrix axes, distinguishes parameter state from activations, motivates heads and information boundaries, connects token loss to attention, and follows an actual class call before the full source.

In the subsequent review, the learner said these detailed explanations were good and requested an overview-to-depth progression. The opening now shows the whole translation process before opening a component, introduces encoder/decoder names after explaining their jobs, locates attention in plain language and maps the successive zooms through the course. The accepted detailed calculations and programs are retained.

The numerical fixture and exact programs remain useful as inspectable arithmetic. They are explicitly separated from learned linguistic representations. New diagrams show actual data paths and axes. Practice asks about influences, pairing, leakage and supervision rather than merely repeating names. Setup remains a reference.

All Attention video attachments were removed. Prior delivery checks remain historical records and do not establish that the teaching was effective. The revised text’s teaching acceptance remains for the learner to judge.
''')

(COURSE/'project/readiness.md').write_text('''# Implementation and understanding remain separate

The project has implemented attention, a locked CPU environment, an executable demo and 35 existing component/visualization checks. This course explains the actual module through complete inspectable programs. The current authoring record records which checks were rerun for the 26 October 2026 revision.

The learner rejected the preceding tutorial and videos. The whole route was compared with external authored tutorials and rewritten; every Attention video attachment was removed. Earlier approval and playback receipts describe their historical date and do not establish current teaching acceptance.

The next source implementation still needs tokenizer/data/batch contracts, then a complete encoder/decoder, vocabulary objective, training, decoding/evaluation and saved recovery. There is no new dataset, trained model, translation result or hosted execution from this tutorial revision.

Understanding would require the learner’s explanation of information paths and changed conditions. Authored answers, runnable code and browser checks do not record that explanation or move the current learner checkpoint.
''')

ref=(COURSE/'project/reference-runs.md').read_text()
ref=ref.replace('The current deliverable is text for approval; no replacement videos are produced during this phase.','The 26 October 2026 deliverable is the complete revised written route. No replacement videos are produced or attached.')
(COURSE/'project/reference-runs.md').write_text(ref)
research=(BEFORE/'research.md').read_text()
research='''# Current research and whole-course comparison — 26 October 2026

The latest learner instruction rejects a paragraph-level repair: compare the complete course with good attention tutorials, diagnose its teaching structure and remove its videos. The [full comparison](project/teaching-review.md) summarizes the review; the owning implementation record retains the twelve-chapter audit. The revised route starts from translation input/output and the missing cross-row dependency, then derives selection/content, layouts, masks, learning and the actual implementation.

The learner subsequently described the detailed explanations as good and requested a big overview followed by progressively deeper passes. The entry and opening diagram now establish the whole process and explain the two jobs before naming their components. A chapter map makes the successive zooms explicit; the detailed calculations remain intact.

Read the primary authored Illustrated Transformer, the historical 2018 Annotated Transformer, D2L Q/K/V and Bahdanau chapters, and the original Bahdanau/Transformer papers on 26 October 2026. External examples and figures are not copied. Numerical calculations and exact source remain independently checked local material. The separate Q/K/V choices and component semantics are supported by actual source; claims about teaching organization are editorial assessment. New SVGs are original diagrams of this route’s data and arithmetic.

All Attention video bindings are removed; original local media/provenance is recoverable. Previous approvals, byte checks and playback receipts are historical evidence, not current teaching acceptance.

## Earlier inspection record

'''+research
(COURSE/'research.md').write_text(research)
practice='# Explain what can influence an output\n\nThese optional questions examine the decisions developed in the [main route](lessons/01-weighted-information.md). Try a prediction before revealing the answer. The stored answers are reference explanations, not recorded learner responses.\n\n'
practice+='\n\n'.join('[[tutorial-practice:'+p['id']+']]' for p in data['practice'])+'\n'
(COURSE/'practice/explain-attention.md').write_text(practice)

prefix=os.path.relpath(COURSE,NOTES)
assets=NOTES.parent.parent/'tutorial-assets'
for source in (COURSE/'images').glob('2[0-7]-*.svg'):
    target=assets/'figures'/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
for source in (COURSE/'examples/walkthrough').glob('*.py'):
    target=assets/'walkthrough'/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
target=assets/'evidence/attention-source.txt'
target.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(COURSE/'evidence/attention-source.txt',target)
def private_link(match):
    name=match[1]
    if name.startswith('images/'):
        return ']('+os.path.relpath(assets/'figures'/Path(name).name,NOTES)+')'
    if name.startswith('examples/walkthrough/'):
        return ']('+os.path.relpath(assets/'walkthrough'/Path(name).name,NOTES)+')'
    return ']('+prefix+'/'+name+')'
parts=['# Attention: from the translation task to the actual computation\n\nRead sections 1–11 in order. The [separate chapter route]('+prefix+'/lessons/01-weighted-information.md) contains the same explanation and revealable practice; [setup]('+prefix+'/lessons/02-project-setup.md) is a run reference. The videos have been removed.\n\n']
for c in data['chapters']:
    if c['id']=='setup':continue
    text=(COURSE/c['content_path']).read_text()
    text=re.sub(r'\[\[practice:[^\]]+\]\]','',text)
    text=re.sub(r'\]\((?!https?://|#|source:)([^)]+)\)',private_link,text)
    text=text.replace('(source:attention-source)','('+os.path.relpath(assets/'evidence/attention-source.txt',NOTES)+')')
    parts.append(text+'\n')
(NOTES/'02-notes-to-understanding.md').write_text(''.join(parts))
for target,source in [('01-learning-points.md','project/learning-points/01-attention.md'),('04-comparisons.md','project/comparisons/01-attention.md')]:
    text=(COURSE/source).read_text()
    text=re.sub(r'\]\((?!https?://|#)([^)]+)\)',lambda m:']('+prefix+'/'+m[1]+')',text)
    (NOTES/target).write_text(text)
(NOTES/'03-tasks.md').write_text('''# Next understanding step

Read the revised route from its translation task. Identify the first transition whose purpose or information path remains unclear, then revise that explanation. No learner answer or acceptance is inferred from the author’s checks.

Attention implementation remains the current source scope. Tokenizer/data and later model/training phases have not started. Videos are removed and no replacement production is requested.
''')
current='''# Whole attention course revised after comparative review — 26 October 2026

The learner positively reviewed the detailed explanations and requested one structural refinement: start with the whole picture, then progressively open the mechanism while building on earlier sections. The entry now shows that overview and chapter progression before token layouts or internal architecture names. The complete numerical programs remain unchanged.

The learner rejected the explanation and videos and required a complete-course comparison. All twelve chapters and supporting pages were reviewed against primary authored attention tutorials, then rewritten around translation inputs, the missing cross-row dependency, design choices, results and actual code execution. Eight original diagrams expose those data paths. All 22 bindings for the eleven Attention videos are detached from the course and project notes; recoverable media is retained locally.

The ten complete runnable programs retain their exact previous bytes; actual Attention/Bike source, tests and the Attention Colab notebook are protected separately. Verification is recorded in the shared owning tutorial package’s 26 October review. Source implementation, trained/hosted results and learner progress have not advanced.

Next: review [the revised understanding notes](02-notes-to-understanding.md), beginning with the actual translation problem. Teaching acceptance remains pending. No new videos are authorized by the rejection.
'''
for name in ['00-resume.md','05-tracker.md']:(NOTES/name).write_text(current)
print(json.dumps({'chapters':len(data['chapters']),'new_external_sources':len(sources),'practice':len(data['practice']),'private_notes_synced':True}))
