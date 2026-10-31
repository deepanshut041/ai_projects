"""Execute every displayed Python program in an independent namespace."""
from pathlib import Path
import ast
import builtins
import contextlib
import hashlib
import io
import json
import math
import re
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'ai').is_file() and (p/'cli').is_dir())
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
manifest=json.loads((COURSE/'lesson.json').read_text());report=[]
for chapter in manifest['chapters']:
    text=(COURSE/chapter['content_path']).read_text()
    for index,source in enumerate(re.findall(r'^```python\n(.*?)^```\s*$',text,re.M|re.S)):
        tree=ast.parse(source)
        definitions={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        imports={a.asname or a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
        assigned={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        missing=calls-definitions-imports-assigned-set(dir(builtins))
        assert not missing,(chapter['id'],missing)
        scope={'__name__':'__main__'};output=io.StringIO()
        with contextlib.redirect_stdout(output):exec(compile(source,chapter['content_path'],'exec'),scope)
        key=chapter['id'];tensor=lambda value:torch.tensor(value,dtype=torch.float64)
        if key=='weighted':torch.testing.assert_close(scope['context'],tensor([1.5,1.0]))
        if key in ['softmax','shapes','projections','cross']:
            actual=scope['context'];actual=actual if actual.ndim==1 else actual[0]
            torch.testing.assert_close(actual,tensor([1.6044483707191437,1.1977758146404283]))
        if key=='heads':torch.testing.assert_close(scope['output'][0,0],tensor([1.6892751930060728,4/3]))
        if key=='padding':
            torch.testing.assert_close(scope['context'][0],tensor([1.3395230986533138,.6604769013466861]))
            assert scope['empty_context'].count_nonzero()==0
        if key=='causal':torch.testing.assert_close(scope['context'][0],tensor([2.,0.]))
        if key=='gradient':
            torch.testing.assert_close(scope['w_q'].grad[0],tensor([-.22313223383820946,.45129017016991646]))
            assert math.isclose(scope['new_loss'].item(),.770391167435663,abs_tol=1e-12)
            # Independent finite differences of the displayed scalar objective.
            w=scope['w_q'].detach();eps=1e-6
            for axis in range(2):
                losses=[]
                for sign in [1,-1]:
                    probe=w.clone();probe[0,axis]+=sign*eps
                    c,_=scope['attend'](scope['x'][:1]@probe,scope['k'],scope['v'])
                    losses.append((.5*((c-scope['target'])**2).sum()).item())
                assert math.isclose((losses[0]-losses[1])/(2*eps),scope['w_q'].grad[0,axis].item(),abs_tol=1e-8)
        if key=='interface':
            torch.testing.assert_close(scope['output'][0,0],tensor([1.6044483707191437,1.1977758146404283]))
            actual=ast.parse((PROJECT/'src/core/attention.py').read_text())
            reference={n.name:ast.dump(n,include_attributes=False) for n in actual.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            for n in tree.body:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(n,include_attributes=False)==reference[n.name],n.name
        report.append({'chapter':key,'block':index,'independent_execution':True,'missing_local_callees':sorted(missing),'sha256':hashlib.sha256(source.encode()).hexdigest(),'stdout':output.getvalue()})
assert len(report)==10
protected=json.loads((ROOT/'.ai/attention-text-rewrite/2026-10-24/protected.json').read_text())
for name,digest in protected.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
bindings=json.loads((ROOT/'.ai/video-library/bindings.json').read_text())['bindings']
assert not [b for b in bindings if 'attention-is-all-you-need/' in b['target']['document']]
result={'programs':report,'all_passed':True,'protected_files_unchanged':len(protected),'attention_video_bindings':0,'text_approval':'pending','video_generation_operations':0}
(ROOT/'.ai/attention-text-rewrite/2026-10-24/visible-programs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='programs'}))
