"""Verify source restoration and complete teaching snippets in isolation."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
PYTHON=PROJECT/'.venv/bin/python'
result={}
with tempfile.TemporaryDirectory(prefix='attention-tutorial-source-') as temp:
    restored=Path(temp)/'restored-project'
    run=subprocess.run(['python3',str(COURSE/'examples/restore_source.py'),str(restored)],capture_output=True,text=True,check=True)
    result['restore']=json.loads(run.stdout)
    run=subprocess.run(['python3',str(COURSE/'examples/restore_source.py'),str(restored)],capture_output=True,text=True)
    assert run.returncode!=0 and 'existing paths are never overwritten' in run.stderr
    result['overwrite_rejected']=True
    downloads=Path(temp)/'downloads';downloads.mkdir()
    for name,source in [('restore_source.py','examples/restore_source.py'),('source-bundle.json','code/source-bundle.json')]:
        shutil.copyfile(COURSE/source,downloads/name)
    sibling=Path(temp)/'same-folder-project'
    run=subprocess.run(['python3',str(downloads/'restore_source.py'),str(sibling)],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)['files']==25
    result['two_downloads_same_folder_restore']=True
    prologue='import sys; from pathlib import Path; sys.path.insert(0,'+repr(str(restored/'src'))+'); sys.path.insert(0,'+repr(str(restored))+'); '
    script=prologue+'import attention_is_all_you_need.attention as module; assert Path(module.__file__).resolve().is_relative_to(Path('+repr(str(restored))+')); import pytest; raise SystemExit(pytest.main(["-q", "tests"]))'
    run=subprocess.run([str(PYTHON),'-I','-c',script],cwd=restored,capture_output=True,text=True,check=True)
    assert '35 passed' in run.stdout and 'skipped' not in run.stdout
    result['isolated_tests']=run.stdout.strip()
    result['snippet_outputs']={}
    for filename in ('01-weighted-information.md','10-reusable-attention.md','11-gradients-and-learning.md'):
        text=(COURSE/'lessons'/filename).read_text()
        blocks=re.findall(r'^```python\n(.*?)^```',text,re.M|re.S)
        assert len(blocks)==1
        run=subprocess.run([str(PYTHON),'-I','-c',prologue+'exec('+repr(blocks[0])+')'],cwd=restored,capture_output=True,text=True,check=True)
        result['snippet_outputs'][filename]=run.stdout.strip()
    result['syntax_checked_blocks']=0
    for lesson in (COURSE/'lessons').glob('*.md'):
        for index,block in enumerate(re.findall(r'^```python\n(.*?)^```',lesson.read_text(),re.M|re.S)):
            compile(block,str(lesson)+':'+str(index),'exec')
            result['syntax_checked_blocks']+=1
    # Public source restoration itself does not require any live project import.
    result['imports_from_restored_source']=True
before=json.loads((HERE/'preservation-before.json').read_text())
changed=[name for name,expected in before.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected]
assert not changed,changed
result['preserved_files']=len(before)
result['model_notebook_captions_bike_unchanged']=True
result['passed']=True
(HERE/'source-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
