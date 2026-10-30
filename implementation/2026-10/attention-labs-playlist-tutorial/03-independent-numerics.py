"""Compare original reference calculations with independent CPU tensor operations."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import math
import random
import runpy
import subprocess
import sys
import tempfile
import torch

DOCS = Path(__file__).resolve().parents[3]
BASE = DOCS / 'tutorials/projects/attention-labs'
FOUNDATIONS = BASE / 'attention-foundations/v1'
HEADS = BASE / 'attention-heads/v1'
sys.path.insert(0, str(FOUNDATIONS / 'examples'))
from attention_math import attention, X, WQ, WK, WV

torch.set_num_threads(1)
random.seed(173)
torch.manual_seed(173)
dtype = torch.float64
cases = 0
maximum_error = 0.0
for query_count, source_count in [(1, 3), (2, 4), (3, 3)]:
    for key_width in [1, 2, 5]:
        for value_width in [1, 3]:
            q = torch.randn(query_count, key_width, dtype=dtype)
            k = torch.randn(source_count, key_width, dtype=dtype)
            v = torch.randn(source_count, value_width, dtype=dtype)
            for padded in [False, True]:
                valid = torch.tensor([True] * (source_count - 1) + [not padded])
                raw = q @ k.T / math.sqrt(key_width)
                expected_weights = raw.masked_fill(~valid, -torch.inf).softmax(-1)
                expected = expected_weights @ v
                _, actual_weights, actual = attention(q.tolist(), k.tolist(), v.tolist(), key_valid=valid.tolist())
                actual = torch.tensor(actual, dtype=dtype)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(torch.tensor(actual_weights, dtype=dtype), expected_weights, atol=1e-12, rtol=1e-12)
                maximum_error = max(maximum_error, (actual - expected).abs().max().item())
                cases += 1

# Autograd independently checks the local projection gradient shown in Lesson 6.
theta = torch.tensor(1.0, dtype=dtype, requires_grad=True)
x = torch.tensor(X, dtype=dtype)
wq = torch.stack([torch.stack([theta, theta * 0]), torch.tensor([0., 1.], dtype=dtype)])
q = x @ wq
k = x @ torch.tensor(WK, dtype=dtype)
v = x @ torch.tensor(WV, dtype=dtype)
out = (q @ k.T / math.sqrt(2)).softmax(-1) @ v
loss = (out[-1, 0] - 2).square() / 2
loss.backward()
with contextlib.redirect_stdout(io.StringIO()) as captured:
    runpy.run_path(str(FOUNDATIONS / 'examples/06_projections.py'), run_name='__main__')
projection = json.loads(captured.getvalue())
assert abs(theta.grad.item() - projection['gradient']) < 1e-9

# Extract displayed complete files into a disposable directory and run each stage.
extracted = 0
with tempfile.TemporaryDirectory(prefix='attention-tutorial-extraction-') as tmp:
    temp = Path(tmp)
    for slug in ['attention-foundations', 'attention-heads']:
        target = temp / slug
        target.mkdir()
        lesson_root = BASE / slug / 'v1'
        for lesson in json.loads((lesson_root / 'lesson.json').read_text())['chapters']:
            text = (lesson_root / lesson['content_path']).read_text()
            import re
            for match in re.finditer(r'<!-- example:([^:]+):start -->.*?```(?:python|cpp)\n(.*?)\n```.*?<!-- example:\1:end -->', text, re.S):
                name, content = match.groups()
                assert content.rstrip() == (lesson_root / 'examples' / name).read_text().rstrip()
                (target / name).write_text(content + '\n')
                if name in ['row_math.py', 'attention_math.py']:
                    continue
                if name.endswith('.cpp'):
                    binary = target / 'cpp-reference'
                    subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', str(target / name), '-o', str(binary)], check=True)
                    argv = [str(binary)]
                else:
                    argv = [sys.executable, str(target / name)]
                result = subprocess.run(argv, cwd=target, capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                json.loads(result.stdout)
                extracted += 1

spec = importlib.util.spec_from_file_location('head_reference', HEADS / 'examples/multihead_torch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
packed_cases = 0
for width, heads in [(2, 1), (2, 2), (4, 2), (6, 3)]:
    model = module.PackedMHA(width, heads).double()
    inputs = torch.randn(2, 5, width, dtype=dtype, requires_grad=True)
    actual, _ = model(inputs)
    expected = module.loop_reference(inputs, model)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    grad_a = torch.autograd.grad(actual.square().sum(), model.query.weight, retain_graph=True)[0]
    grad_b = torch.autograd.grad(expected.square().sum(), model.query.weight)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-12, rtol=1e-12)
    packed_cases += 1

record = {'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu',
          'independent_attention_cases': cases, 'maximum_output_error': maximum_error,
          'autograd_projection_gradient': theta.grad.item(), 'displayed_programs_extracted_and_run': extracted,
          'packed_forward_and_gradient_cases': packed_cases, 'passed': True}
destination = Path(__file__).with_name('04-numerical-verification.json')
destination.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
