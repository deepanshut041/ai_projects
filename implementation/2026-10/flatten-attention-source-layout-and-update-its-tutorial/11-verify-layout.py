"""Verify the grouped project, packaged imports, demo parity and source exports."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
BUILD=ROOT/'.ai/attention-layout/2026-10-23/dist'
before=json.loads((HERE/'preservation-before.json').read_text())['files']
mapping=json.loads((HERE/'path-mapping.json').read_text())
result={}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(argv,cwd=None,env=None):
    p=subprocess.run(list(map(str,argv)),cwd=cwd,capture_output=True,text=True,env=env)
    if p.returncode:raise RuntimeError(str(argv)+'\n'+p.stdout+'\n'+p.stderr)
    return p.stdout

assert digest(PROJECT/'src/core/attention.py')==before['src/attention_is_all_you_need/attention.py']
assert not (PROJECT/'src/attention_is_all_you_need').exists()
assert not list(PROJECT.glob('*.py'))
assert {p.name for p in (PROJECT/'src').glob('*.py')} == {'attention.py','tokenizer.py','prepare_data.py','train.py','inference.py','compute_bleu.py'}
assert not (PROJECT/'src/cli').exists()
for group in ('core','data','training','utils'):assert (PROJECT/'src'/group/'__init__.py').is_file()
for old,new in mapping.items():
    if old not in ('attention.py','attention_visuals.py','attention_mechanism.ipynb'):
        assert digest(PROJECT/new)==before[old],new
from attention import run_demo
with contextlib.redirect_stdout(io.StringIO()):demo=run_demo()
observed={'errors':demo['errors'],'output_sha256':hashlib.sha256(demo['output'].numpy().tobytes()).hexdigest(),'weights_sha256':hashlib.sha256(demo['decoder_weights'].numpy().tobytes()).hexdigest()}
assert observed==json.loads((HERE/'before-demo.json').read_text())
result.update(numerical_module_byte_identical=True,demo_bitwise_identical=True,groups=['core','data','training','utils'])

wheel=next(BUILD.glob('*.whl'))
with zipfile.ZipFile(wheel) as archive:
    entries=archive.namelist()
    for name in ('core/attention.py','data/dataset.py','data/tokenizer.py','training/trainer.py','attention.py','utils/attention_visuals.py'):assert name in entries,name
    assert all(not name.endswith('.ipynb') for name in entries)
    assert not any(name.startswith(('src/','cli/','attention_is_all_you_need/')) for name in entries)
    result['wheel_files']=entries
sdist=next(BUILD.glob('*.tar.gz'))
with tarfile.open(sdist) as archive:
    entries=archive.getnames()
    assert any(name.endswith('/notebooks/attention_colab.ipynb') for name in entries)
    assert any(name.endswith('/src/data/dataset.py') for name in entries)
    assert not any('/docs/implementation/' in name or '/docs/tutorial-assets/' in name or '/.sources/' in name for name in entries)
    result['sdist_keeps_notebook_excludes_private_material']=True

with tempfile.TemporaryDirectory(prefix='attention-layout-wheel-') as temporary:
    temp=Path(temporary);env=temp/'env'
    run([sys.executable,'-m','venv',env])
    python=env/'bin/python'
    run(['uv','pip','install','--python',python,'--no-deps','--reinstall',wheel])
    # Dependency libraries may come from the base environment; project source may not.
    script='''import contextlib,io,json,sys
from pathlib import Path
source=Path(sys.argv[1]).resolve()
sys.path[:]=[p for p in sys.path if not Path(p or '.').resolve().is_relative_to(source/'src')]
sys.path.append(sys.argv[2])
import core.attention,attention,utils.attention_visuals,data.dataset,training.checkpoint
for module in (core.attention,attention,utils.attention_visuals,data.dataset,training.checkpoint):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix)),module.__file__
utils.attention_visuals.configure(save_figures=False,output_dir=Path.cwd()/'figures')
with contextlib.redirect_stdout(io.StringIO()):attention.run_demo()
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null',str(source/'tests')]))
'''
    dependencies=sysconfig.get_paths()['purelib']
    output=run([python,'-I','-c',script,PROJECT,dependencies],cwd=temp)
    assert '35 passed' in output and 'skipped' not in output
    result['installed_wheel_tests']=output.strip()
    output=run([env/'bin/attention-demo'],cwd=temp,env={**os.environ,'PYTHONPATH':dependencies})
    assert 'PASS:' in output
    result['installed_console_entrypoint']=True
    result['installed_dependency_libraries']=dependencies

bundle=json.loads((COURSE/'code/source-bundle.json').read_text())
assert 'src/core/attention.py' in bundle['files'] and 'notebooks/attention_colab.ipynb' in bundle['files']
assert not any('src/attention_is_all_you_need' in name for name in bundle['files'])
with tempfile.TemporaryDirectory(prefix='attention-layout-source-') as temporary:
    restored=Path(temporary)/'project'
    run([sys.executable,COURSE/'examples/restore_source.py',restored])
    for name,item in bundle['files'].items():assert digest(restored/name)==item['sha256']
    code='import sys;sys.path.insert(0,'+repr(str(restored/'src'))+');import core.attention as a;assert a.__file__.startswith('+repr(str(restored))+');import pytest;raise SystemExit(pytest.main(["-q","tests"]))'
    output=run([sys.executable,'-I','-c',code],cwd=restored)
    assert '35 passed' in output and 'skipped' not in output
    result['restored_source_tests']=output.strip()
    result['restored_source_files']=len(bundle['files'])
result['passed']=True
(HERE/'layout-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
