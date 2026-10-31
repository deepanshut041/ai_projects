"""Run existing meaningful numerical checks and verify this edit's boundaries."""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path('/mnt/projects/ai')
RECORD=Path(__file__).parent
OUT=ROOT/'.ai/attention-course-review/2026-10-26'
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'

# Reuse the existing standalone/known-answer/finite-difference/source-AST checks.
# Its dated preservation and delivery section belongs to the old run, so replace
# only those run-specific checks with this review's independently saved baseline.
checker=(RECORD/'13-check-visible-programs.py').read_text().split('\nprotected=',1)[0]
scope={'__file__':str(RECORD/'13-check-visible-programs.py'),'__name__':'__main__'}
exec(compile(checker,str(RECORD/'13-check-visible-programs.py'),'exec'),scope)
report=scope['report']
before=json.loads((OUT/'programs-before.json').read_text())
actual={}
for path in (COURSE/'lessons').glob('*.md'):
    for i,source in enumerate(re.findall(r'^```python\n(.*?)^```\s*$',path.read_text(),re.M|re.S)):
        actual[f'{path.name}:{i}']=hashlib.sha256(source.encode()).hexdigest()
assert actual==before,'A previously checked standalone program changed'
protected=json.loads((OUT/'protected.json').read_text())
for path,digest in protected.items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
old=json.loads((OUT/'before/bindings.json').read_text())['bindings']
now=json.loads((ROOT/'.ai/video-library/bindings.json').read_text())['bindings']
removed=json.loads((OUT/'removed-bindings.json').read_text())
assert [b for b in old if b not in removed]==now,'An unrelated binding changed'
assert not [b for b in now if b['video_id'].startswith('projects/attention-') or 'attention-is-all-you-need/' in b['target'].get('document','')]
manifest=json.loads((COURSE/'lesson.json').read_text())
assert len(manifest['chapters'])==12 and not manifest['media']
for c in manifest['chapters']:
    text=(COURSE/c['content_path']).read_text()
    assert text.startswith('## '+c['title']+'\n'),c['id']
    assert 'beside the river' not in text and 'approved this text' not in text
images=[p for p in (COURSE/'images').glob('2[0-7]-*.svg')]
assert len(images)==8

# Exercise newly explained invariants using the retained attention function.
import torch
base={'__name__':'probe'}
program=(COURSE/'examples/walkthrough/03_shapes.py').read_text()
exec(compile(program,'shapes.py','exec'),base)
attend,x=base['attend'],base['x']
perm=torch.tensor([2,0,1])
original,_=attend(x,x,2*x)
permuted,_=attend(x[perm],x[perm],2*x[perm])
torch.testing.assert_close(permuted,original[perm])
paired,_=attend(x,x[perm],(2*x)[perm])
torch.testing.assert_close(paired,original)
unpaired,_=attend(x,x[perm],2*x)
assert not torch.allclose(unpaired,original)
q=x[:1]
with_zero,_=attend(q,torch.tensor([[1.,0.],[0.,1.],[0.,0.]],dtype=x.dtype),torch.tensor([[2.,0.],[0.,2.],[0.,0.]],dtype=x.dtype))
without_zero,_=attend(q,x[:2],2*x[:2])
assert not torch.allclose(with_zero,without_zero)
result={'passed':True,'chapters':12,'complete_programs':len(report),'program_bytes_preserved':len(before),'protected_files_unchanged':len(protected),'source_AST_parity':True,'finite_difference_gradient':True,'permutation_and_key_value_pairing':True,'zero_value_denominator_contamination':True,'detached_bindings':len(removed),'removed_videos':len({b['video_id'] for b in removed}),'unrelated_bindings_preserved':len(now),'original_figures':len(images),'teaching_acceptance':'pending','new_videos':0}
(OUT/'numerical-and-preservation.json').write_text(json.dumps({'summary':result,'programs':report},indent=2)+'\n')
print(json.dumps(result))
