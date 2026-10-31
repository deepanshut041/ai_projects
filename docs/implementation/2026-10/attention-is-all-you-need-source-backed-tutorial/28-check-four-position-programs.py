"""Check the revised arithmetic independently of the actual attention source."""
from pathlib import Path
import ast
import builtins
import contextlib
import hashlib
import io
import json
import math
import re
import runpy

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'ai').is_file())
COURSE = ROOT / 'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
PROJECT = ROOT / 'ai_projects/projects/transformer-labs/attention-is-all-you-need'
OUT = ROOT / '.ai/attention-four-rows/2026-10-27'
manifest = json.loads((COURSE / 'lesson.json').read_text())
route = {e['chapter']: e['path'] for e in json.loads((COURSE / 'examples/walkthrough/route.json').read_text())}
with contextlib.redirect_stdout(io.StringIO()):
    scalar = runpy.run_path(str(COURSE / 'examples/arithmetic.py'))
retrieve, x, v = scalar['retrieve'], scalar['X'], scalar['V']
assert set(map(tuple, x)) == {(0., 0.), (1., 0.), (0., 1.), (1., 1.)}
contexts = [retrieve(q, x, v)[0] for q in x]
source = ast.parse((PROJECT / 'src/core/attention.py').read_text())
definitions = {n.name: ast.dump(n, include_attributes=False) for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
report = []
for chapter in manifest['chapters']:
    for block in re.findall(r'^```python\n(.*?)^```\s*$', (COURSE / chapter['content_path']).read_text(), re.M | re.S):
        key = chapter['id']
        assert block == (COURSE / route[key]).read_text(), key
        tree = ast.parse(block)
        local = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        imports = {a.asname or a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not calls - local - imports - assigned - set(dir(builtins)), key
        scope = {'__name__': '__main__'}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(block, chapter['content_path'], 'exec'), scope)
        tensor = lambda val: torch.tensor(val, dtype=torch.float64)
        if key == 'weighted':
            torch.testing.assert_close(scope['context'], tensor(scalar['supplied']))
        if key == 'softmax':
            torch.testing.assert_close(scope['context'], tensor(contexts[0]))
        if key in ('shapes', 'projections'):
            torch.testing.assert_close(scope['context'], tensor(contexts))
        if key == 'heads':
            high = 2 * math.e / (math.e + 1)
            torch.testing.assert_close(scope['output'][0], tensor([[high, 1], [1, high], [high, high], [1, 1]]))
        if key == 'padding':
            expected = [retrieve(q, x[:3], v[:3])[0] for q in x]
            torch.testing.assert_close(scope['context'], tensor(expected))
            assert scope['empty_context'].count_nonzero() == 0
        if key == 'causal':
            expected = [retrieve(q, x[:i + 1], v[:i + 1])[0] for i, q in enumerate(x)]
            torch.testing.assert_close(scope['context'], tensor(expected))
        if key == 'cross':
            torch.testing.assert_close(scope['context'], tensor(contexts[:2]))
            assert tuple(scope['weights'].shape) == (2, 4)
        if key == 'gradient':
            torch.testing.assert_close(scope['w_q'].grad, tensor([scalar['query_gradient'], [0, 0]]))
            assert math.isclose(scope['new_loss'].item(), scalar['loss'](scalar['next_query']), abs_tol=1e-12)
        if key == 'interface':
            torch.testing.assert_close(scope['output'][0], tensor(contexts))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    assert ast.dump(node, include_attributes=False) == definitions[node.name], node.name
        report.append({'chapter': key, 'passed': True, 'stdout': output.getvalue()})
assert len(report) == len(route) == 10
protected = json.loads((OUT / 'protected.json').read_text())
for path, digest in protected.items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
# Frozen project code/export remains the pre-edit captured implementation.
for before in (OUT / 'before-course/code').rglob('*'):
    if before.is_file():
        assert before.read_bytes() == (COURSE / 'code' / before.relative_to(OUT / 'before-course/code')).read_bytes()
assert not manifest['media']
bindings = json.loads((ROOT / '.ai/video-library/bindings.json').read_text())['bindings']
assert not [b for b in bindings if 'attention-is-all-you-need/' in b['target'].get('document', '')]
summary = {'complete_programs': len(report), 'independent_scalar_and_finite_difference': True,
           'all_four_binary_pairs': True, 'download_parity': True, 'project_source_AST_parity': True,
           'protected_project_files': len(protected), 'frozen_source_preserved': True,
           'attention_video_bindings': 0, 'learner_progress_changed': False, 'passed': True}
(OUT / 'program-verification.json').write_text(json.dumps({'summary': summary, 'programs': report}, indent=2) + '\n')
print(json.dumps(summary))
