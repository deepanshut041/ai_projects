"""Generate cleared personal notes and optional probes from the topic workbook."""

import argparse
import json
from pathlib import Path
import re


def notes_text():
    candidates = [Path(__file__).with_name('study-notes.md'),
                  Path(__file__).resolve().parents[1] / 'project' / 'study-notes.md']
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text()
    raise FileNotFoundError('Keep study-notes.md beside the exported sources or course project records')


def notebook(text=None):
    text = notes_text() if text is None else text
    sections = re.split(r'(?m)(?=^## Lesson \d+ — )', text)
    if len(sections) < 2:
        raise ValueError('The workbook needs numbered topic sections')
    cells = []

    def markdown(identity, source):
        cells.append({'cell_type': 'markdown', 'id': identity, 'metadata': {},
                      'source': source.splitlines(keepends=True)})

    def code(identity, source):
        cells.append({'cell_type': 'code', 'id': identity, 'metadata': {},
                      'source': source.splitlines(keepends=True),
                      'execution_count': None, 'outputs': []})

    markdown('intro', sections[0] + '\nSave your own copy before writing notes. Read the full lesson, '
             'derive and predict first, then run an optional probe. Run beside the exported sources '
             'or inside the course folder. Torch probes use the recorded compatible CPU environment; '
             'the notebook does not install dependencies or train a model.\n')
    code('setup', '''from pathlib import Path
import sys, math
candidates = [Path.cwd(), Path.cwd() / 'examples']
source_folder = next((p for p in candidates if (p / 'deep_learning_math.py').is_file()), None)
if source_folder is None:
    raise FileNotFoundError('Open beside the exported sources or from the course folder')
sys.path.insert(0, str(source_folder))
from prerequisite_math import linear, central_difference
from deep_learning_math import shared_parameter, binary_loss, mlp_trace, worked_fixture
print('Source probes ready. Write a prediction before executing a calculation.')
''')
    probes = {
        'linear algebra': '''print(linear([[1,2],[3,4]], [[1,0,2],[0,1,3]], [0,0,1]))
# Predict a zero row and a changed second row before modifying the inputs.
''',
        'derivatives': '''result = shared_parameter(1.)
measured = central_difference(lambda w: (2*w-1)**2 + (3*w-1)**2, 1.)
assert abs(measured-result['gradient']) < 1e-7
print(result)
''',
        'backpropagation': '''fixture = worked_fixture()
trace = mlp_trace(**fixture)
for name, value in trace.items():
    print(name, value)
# Change the supplied target only after deriving its backward effect.
''',
        'loss functions': '''result = binary_loss(-math.log(99), 1)
assert abs(result['cross_entropy_gradient']+.99) < 1e-12
print(result)
''',
        'embeddings': '''from transformer_foundation_probes import embedding_gradient_probe
print(embedding_gradient_probe())
# Which updates originate from lookup, and which from a tied output path?
''',
        'sequence order': '''from transformer_foundation_probes import sinusoidal_positions
print(sinusoidal_positions(3, 4))
# Predict the zero-position row and a shifted position before changing arguments.
''',
        'queries, keys': '''import torch
from transformer_foundation_probes import explicit_attention
q = torch.tensor([[1.,0.]], dtype=torch.float64)
k = torch.tensor([[0.,0.],[math.sqrt(2)*math.log(2),0.],
                  [math.sqrt(2)*math.log(3),0.]], dtype=q.dtype)
v = torch.tensor([[6.,0.],[0.,3.],[2.,2.]], dtype=q.dtype)
output, weights = explicit_attention(q,k,v)
assert torch.allclose(output, torch.tensor([[2.,2.]],dtype=q.dtype))
print('weights',weights,'output',output)
''',
        'padding masks': '''import torch
from transformer_foundation_probes import explicit_attention
q = torch.tensor([[1.,0.],[1.,0.]], dtype=torch.float64)
k = torch.tensor([[0.,0.],[math.sqrt(2)*math.log(2),0.],
                  [math.sqrt(2)*math.log(3),0.]], dtype=q.dtype)
v = torch.tensor([[6.,0.],[0.,3.],[2.,2.]], dtype=q.dtype)
allowed = torch.tensor([[True,False,True],[False,False,False]])
output, weights = explicit_attention(q,k,v,allowed)
assert torch.isfinite(output).all()
assert torch.equal(output[1],torch.zeros(2,dtype=q.dtype))
print('weights',weights,'output',output)
''',
    }
    for position, section in enumerate(sections[1:], 1):
        markdown(f'notes-{position:02}', section)
        title = section.splitlines()[0].lower()
        matched = [source for key, source in probes.items() if key in title]
        if len(matched) > 1:
            raise ValueError('A topic matched multiple probe definitions')
        if matched:
            code(f'probe-{position:02}', matched[0])
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'}},
            'cells': cells}


def write_notebook(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve existing study notes; choose a new output filename')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook(), indent=2) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    print(write_notebook(parser.parse_args().destination))
