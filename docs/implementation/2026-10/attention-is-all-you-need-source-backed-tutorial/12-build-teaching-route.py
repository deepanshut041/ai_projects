"""Assemble the revised teaching route without changing model source or media."""
from pathlib import Path
import json
import re
import os
import shutil

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'ai').is_file() and (p/'cli').is_dir())
COURSE = ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
PROJECT = ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
NOTES = PROJECT/'docs/implementation/01_attention-mechanism'
ROUTE = ['weighted','softmax','shapes','projections','heads','padding','causal','cross','gradient','interface','handoff','setup']
manifest = json.loads((COURSE/'lesson.json').read_text())
chapters = {c['id']: c for c in manifest['chapters']}
checks = {
 'weighted': ('If the weights become [1,0,0] for our same three values, what is the context? Is it a vocabulary prediction?', 'It is [2,0], exactly value 0. This is a retrieved feature vector; no vocabulary output layer or trained token prediction has been introduced.', 'Apply the weight to both coordinates and distinguish context from a predicted token.'),
 'softmax': ('Keep the same keys and values but use query [0,0]. What happens, and why?', 'Every dot product is zero. All exponentials are equal, so each position has weight 1/3 and context is [4/3,4/3]. The values did not change.', 'Connect the changed query through scores, normalization and both context coordinates.'),
 'shapes': ('For two queries, three keys and two value features, what do the weights and context shapes mean?', 'Weights are (2,3): a row for each query and a column for each key. Context is (2,2): one two-feature result per query. The key axis is summed out.', 'Name what is retained and what is contracted, rather than only listing dimensions.'),
 'projections': ('Halve only W_V in the identity/doubling example. Which quantities change?', 'Q, K, scores and weights stay the same. V and context are halved. Changing X would generally affect all three roles and would not isolate this cause.', 'Keep the comparison inputs fixed and explain the linear value sum.'),
 'heads': ('Why does head 1 return 4/3 for query position 0 in the two one-feature-head example?', 'Its query coordinate is zero, so its three scores are equal. Its values are [0,2,2]; their average is 4/3. It still reads all three positions.', 'Trace the actual scalar inputs and reject a sentence-splitting interpretation of heads.'),
 'padding': ('What context does a completely blocked query receive in our masked function? Must the project output be zero too?', 'The function explicitly returns zero weights and context for that row. A later biased output projection can return its bias. A key mask also does not itself remove query rows.', 'Separate key visibility, retrieved context, final output and target-loss masking.'),
 'causal': ('At query position 1, with all three keys real, which keys are allowed for causal next-token computation?', 'Keys 0 and 1 only. Key 2 is future information, even though it is real. Padding is a separate restriction and would combine with this rule using AND.', 'Explain the unavailable-answer boundary before applying the triangle.'),
 'cross': ('Our source has three positions and target has two. Which length determines output rows, and which determines key-mask entries?', 'The two target queries determine output rows. The three source keys determine mask entries. Q comes from target representations while paired K/V come from source representations.', 'Trace vector origins through the rectangular comparison and value sum.'),
 'gradient': ('For the supplied target [2,0], the first query gradient is [-0.223132,0.451290]. In which direction does gradient descent move that query?', 'It increases the first coordinate and decreases the second. A 0.1 step gives about [1.022313,-0.045129] and reduces the illustrated loss from 0.795564 to 0.770391. This is a controlled query-parameter probe, not translation training.', 'Use subtraction of the gradient and connect the changed mixture to the chosen target.'),
 'interface': ('How can you inspect the project calculation without relying on the optimized library attention call?', 'Use fused_attn=False or request weights. The shown _context then calls the fully defined explicit scaled_dot_product_attention. The full block includes its mask, layout and validation dependencies.', 'Follow the branch and identify where scores, normalized weights and the value sum are visible.'),
 'handoff': ('What have we explained, and what is still missing before the project can translate?', 'We have explained and exercised the existing attention component. A translation system still needs token/data contracts, complete encoder/decoder computation, vocabulary loss, training/evaluation and recovery. Text approval, learner understanding and hosted results are separate.', 'State the component boundary and a concrete next implementation need.'),
 'setup': ('If a comment-only training script exits successfully, what has it established?', 'Only that the process exited. It has not fitted a model, saved learned state or measured translation quality. Inspect implemented source and actual outputs.', 'Separate executable setup from actual model work.'),
}

# Follow the existing source-title convention without touching code comments.
for c in chapters.values():
    p = COURSE/c['content_path']; inside = False; first_heading = True; lines=[]
    for line in p.read_text().splitlines():
        if line.startswith('```'): inside = not inside
        if not inside and re.match(r'^#{1,3} ', line):
            line = ('## ' if first_heading else '### ') + line.lstrip('#').strip()
            first_heading = False
        lines.append(line)
    p.write_text('\n'.join(lines)+'\n')
    c['title'] = next(l[3:] for l in lines if l.startswith('## '))
manifest['chapters'] = [chapters[k] for k in ROUTE]
manifest['title'] = 'Attention Is All You Need: build the calculation from its purpose'
manifest['question'] = 'Why does a position need attention, how is its information gathered, and how does the complete project code perform that calculation?'
manifest['level'] = 'intermediate'
manifest['practice'] = [{**next(p for p in manifest['practice'] if p['id']==k+'-check'), 'question':checks[k][0], 'answer':checks[k][1], 'rubric':checks[k][2]} for k in ROUTE]
manifest['objectives'] = [{'id':k,'text':chapters[k]['title']} for k in ROUTE]
manifest['examples'][0]['purpose'] = 'Independent standard-library arithmetic for the single three-position teaching fixture and its query-gradient update.'
manifest['examples'][0]['expected_output'] = {'supplied_context':[1.5,1.0], 'query_context':[1.604448,1.197776], 'query_gradient':[-0.223132,0.451290], 'post_step_loss':0.770391, 'passed':True}
(COURSE/'lesson.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')

# Complete program downloads are derived from the actual visible Python fences.
programs=[];folder=COURSE/'examples/walkthrough';folder.mkdir(parents=True,exist_ok=True)
for number,key in enumerate(ROUTE[:-2],1):
    chapter=chapters[key];text=(COURSE/chapter['content_path']).read_text()
    blocks=re.findall(r'^```python\n(.*?)^```\s*$',text,re.M|re.S)
    assert len(blocks)==1,(key,len(blocks))
    name=f'{number:02d}_{key}.py';(folder/name).write_text(blocks[0])
    link=f'**Runnable copy:** [step {number} program](examples/walkthrough/{name}). The code above is complete; the download contains the same definitions and inputs.\n\n'
    if '**Runnable copy:**' not in text:
        text=text.replace('[[practice:',link+'[[practice:',1)
        (COURSE/chapter['content_path']).write_text(text)
    programs.append({'step':number,'chapter':key,'path':str((folder/name).relative_to(COURSE))})
(COURSE/'examples/walkthrough/route.json').write_text(json.dumps(programs,indent=2)+'\n')

# One primary local reading page follows the same conceptual order, without setup.
practice = {p['id']:p for p in manifest['practice']}
header = '''# Understand attention from the problem to the complete code\n\n**Start at section 1 and read through section 11.** We build one example from the need for context to the complete project code. Every Python block includes its inputs and the helper functions it calls.\n\n[Separate tutorial chapters](COURSE/lessons/01-weighted-information.md) and a [setup reference](COURSE/lessons/02-project-setup.md) are available when helpful.\n\n'''
prefix=os.path.relpath(COURSE,NOTES)
header=header.replace('COURSE/',prefix+'/')
parts=[header]
for key in ROUTE[:-1]:
    text=(COURSE/chapters[key]['content_path']).read_text()
    if key=='weighted':
        text=re.sub(r'Read this route in the displayed order\..*?Code is visible on the page\.\n\n', '', text, flags=re.S)
    text=re.sub(r'\[\[practice:[^\]]+\]\]', '', text)
    text=re.sub(r'\]\((?!https?://|#)([^)]+)\)',lambda m:']('+(os.path.relpath(PROJECT/'docs/tutorial-assets/walkthrough'/Path(m[1]).name,NOTES) if m[1].startswith('examples/walkthrough/') else prefix+'/'+m[1])+')',text)
    parts.append(text+'\n')
(NOTES/'02-notes-to-understanding.md').write_text(''.join(parts))
for item in programs:
    dest=PROJECT/'docs/tutorial-assets/walkthrough'/Path(item['path']).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(COURSE/item['path'],dest)
print(json.dumps({'chapters':ROUTE,'complete_programs':len(programs),'primary_notes_words':len((NOTES/'02-notes-to-understanding.md').read_text().split()),'approval':'pending','new_video_operations':0}))
