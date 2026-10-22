"""Create cleared, editable topic notes with optional applied model probes."""

import argparse
import json
from pathlib import Path
import re


def notes_text():
    for path in (Path(__file__).with_name('study-notes.md'),
                 Path(__file__).resolve().parents[1] / 'project' / 'study-notes.md'):
        if path.is_file():
            return path.read_text(encoding='utf-8')
    raise FileNotFoundError('Keep study-notes.md beside the exported reference sources')


def notebook(text=None):
    sections = re.split(r'(?m)(?=^## Lesson \d+ — )', notes_text() if text is None else text)
    if len(sections) < 2:
        raise ValueError('Study notes need numbered topic sections')
    cells = []

    def markdown(identity, source):
        cells.append(dict(cell_type='markdown', id=identity, metadata={},
                          source=source.splitlines(keepends=True)))

    def code(identity, source):
        cells.append(dict(cell_type='code', id=identity, metadata={},
                          source=source.splitlines(keepends=True), execution_count=None, outputs=[]))

    markdown('intro', sections[0] + '\nSave your own copy before writing notes. Read and derive first, '
             'then run an optional probe. This notebook uses the supplied reference modules; it neither '
             'installs dependencies nor trains the full models. Model training and persistent Colab '
             'recovery use the separate exported Colab notebook and course_session.py.\n')
    code('setup', '''from pathlib import Path
import sys, math
candidates = [Path.cwd(), Path.cwd() / 'examples']
SOURCE = next((p for p in candidates if (p / 'model_learning_probes.py').is_file()), None)
if SOURCE is None:
    raise FileNotFoundError('Open beside the exported sources or from the course package')
sys.path.insert(0,str(SOURCE))
import torch
from model_learning_probes import (token_loss_and_gradient, learned_attention_fixture, projected_attention,
    corruption_expectations, controlled_mlm_probe, utf8_byte_character_map,
    qa_endpoint_loss_and_gradient, valid_span_candidates, multiset_f1,
    bleu_from_counts, model_control_probe)
print('Applied probes ready. Write a prediction before executing each calculation.')
''')
    probes = {
        4: '''z = torch.tensor([[0.,math.log(3.)],[20.,-20.]],dtype=torch.float64)
y = torch.tensor([1,-100])
print(token_loss_and_gradient(z,y))
# Predict the ignored-row gradient and a changed target before editing.
''',
        7: '''e = torch.nn.Embedding(4,2,padding_idx=0).double()
(e(torch.tensor([0,2,2,3])) * torch.tensor([[9.,9.],[1.,-2.],[4.,5.],[-3.,6.]])).sum().backward()
print(e.weight.grad)
# Separate repeated lookup contributions from a tied output-head gradient.
''',
        8: '''from translation_model import PositionalEncoding
print(PositionalEncoding(4,4).encodings)
# Which row would a newest-token-only call receive without an offset?
''',
        9: '''result = projected_attention(*learned_attention_fixture())
for key in ('weights','output'):
    print(key,result[key])
# Derive the row/column meaning and change only a value before changing keys.
''',
        12: '''from translation_data import collate_translation
batch = collate_translation([dict(src_ids=[2,4,3],tgt_ids=[2,4,3]),
    dict(src_ids=[2,5,6,3],tgt_ids=[2,5,6,7,3])],0,1)
for name,value in batch.items():
    print(name,value.tolist())
print('supervised targets',int(batch['tgt_outputs'].ne(-100).sum()))
''',
        14: '''u = torch.tensor([1.,2.,3.],dtype=torch.float64,requires_grad=True)
y = torch.nn.LayerNorm(3).double()(u)
y[0].backward()
print('normalized',y.detach(),'input gradient',u.grad)
# Explain why a zero post-norm sublayer does not preserve u.
''',
        20: '''print(bleu_from_counts([3,2],[4,3],4,5))
# Work clipped counts and brevity penalty before running; this is not corpus quality.
''',
        24: '''print(corruption_expectations(40,.15))
print(controlled_mlm_probe())
# Distinguish eligibility, selected targets and replacement roles; predict zero selection.
''',
        28: '''print(utf8_byte_character_map('café'))
# Original character coordinates differ from UTF-8 byte coordinates.
''',
        29: '''print(qa_endpoint_loss_and_gradient(
    torch.zeros(1,4,dtype=torch.float64),torch.zeros(1,4,dtype=torch.float64),
    torch.tensor([1]),torch.tensor([2])))
# Derive both endpoint gradients and identify the packed-position denominator.
''',
        30: '''print(valid_span_candidates([0,7,1],[0,2,8],[None,(0,3),(4,7)],max_tokens=2))
print('duplicate-aware F1',multiset_f1(['cat','cat'],['cat','dog']))
# Restrict joint spans before argmax; count token occurrences rather than sets.
''',
        31: '''print(model_control_probe())
# Name the controlled mode, positions, mask and one falsifiable information-flow claim.
''',
    }
    for section in sections[1:]:
        heading = section.splitlines()[0]
        number = int(re.match(r'## Lesson (\d+) — ',heading).group(1))
        markdown(f'notes-{number:02d}',section)
        if number in probes:
            code(f'probe-{number:02d}',probes[number])
    return dict(nbformat=4,nbformat_minor=5,metadata={
        'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'},
        'language_info': {'name':'python'}},cells=cells)


def write_notebook(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve personal notes; choose a new notebook filename')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(notebook(),indent=2)+'\n',encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    print(write_notebook(parser.parse_args().destination))
