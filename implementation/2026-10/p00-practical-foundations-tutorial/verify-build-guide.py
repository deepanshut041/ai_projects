"""Extract the authored complete files into a NEW disposable project and test them.

Example from workspace home (never pass the learner repository as --output):
python3 ai_projects/docs/implementation/2026-10/p00-practical-foundations-tutorial/verify-build-guide.py \
  --lesson ai_projects/docs/tutorials/projects/p00/transformer-foundations/v1 \
  --output .ai/tutorial-runs/p00-build-check-new \
  --python .ai/envs/p00-tutorial/bin/python
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lesson', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--python', type=Path, required=True)
args = parser.parse_args()
lesson, output, python = args.lesson.resolve(), args.output.resolve(), args.python.absolute()
if output.exists():
    parser.error('Choose a new disposable output directory; existing files are preserved')
output.mkdir(parents=True)
package = output/'src/transformer_foundations_lab'
package.mkdir(parents=True)
(package/'__init__.py').write_text('"""Disposable author verification; not a learner implementation."""\n')
(output/'tests').mkdir()
env = {**os.environ, 'PYTHONPATH': str(output/'src'), 'PYTHONWARNINGS': 'ignore:Failed to initialize NumPy'}
report = {'files': [], 'stages': []}
for page in sorted((lesson/'lessons').glob('*.md')):
    tests = []
    for operation, relative, code in re.findall(r'```python\n# (File|Append to): ([^\n]+)\n(.*?)\n```', page.read_text(), re.S):
        target = output/relative
        if not target.resolve().is_relative_to(output):
            raise ValueError('Unsafe output path')
        ast.parse(code)
        target.parent.mkdir(parents=True, exist_ok=True)
        if operation == 'Append to':
            if not target.exists():
                raise ValueError('Append requires an earlier file: ' + relative)
            code = target.read_text().rstrip() + '\n\n\n' + code
        ast.parse(code)
        target.write_text(code+'\n')
        report['files'].append({'path': relative, 'page': page.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        if relative.startswith('tests/'):
            tests.append(target.name)
    for filename in tests:
        command = [str(python), '-m', 'unittest', 'discover', '-s', 'tests', '-p', filename, '-v']
        result = subprocess.run(command, cwd=output, env=env, text=True, capture_output=True, timeout=120)
        report['stages'].append({'page': page.name, 'test_file': filename, 'returncode': result.returncode, 'output': result.stdout+result.stderr})
        (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.stdout+result.stderr)
        print(page.name, filename, 'passed', flush=True)
# Verify document generation and syntax of each hosted cell; no Colab API calls.
shutil.copy2(lesson/'examples/make_colab_notebook.py', output/'tools/make_colab_notebook.py')
subprocess.run([str(python), 'tools/make_colab_notebook.py'], cwd=output, check=True, capture_output=True, text=True)
notebook = json.loads((output/'notebooks/colab.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        assert cell['execution_count'] is None and cell['outputs'] == []
report['notebook'] = {'cells': len(notebook['cells']), 'output_cleared': True, 'code_syntax': 'passed', 'hosted_execution': 'not run'}
command = [str(python), '-m', 'unittest', 'discover', '-s', 'tests', '-v']
result = subprocess.run(command, cwd=output, env=env, capture_output=True, text=True, timeout=120)
report['combined'] = {'returncode': result.returncode, 'output': result.stdout+result.stderr}
(output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(result.stderr)
raise SystemExit(result.returncode)
