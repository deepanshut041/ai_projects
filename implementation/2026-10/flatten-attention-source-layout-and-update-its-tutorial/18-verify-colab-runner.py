"""Execute the exact Colab notebook cells locally with only file UI simulated.

The source archive, uv bootstrap, locked install, subprocesses, tests and evidence
download are real. This intentionally does not claim Google-hosted execution.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROJECT = ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
EXPORT = ROOT/'.ai/exports/attention-is-all-you-need'
LOG = ROOT/'.ai/attention-layout/2026-10-23/colab-runner-local.log'
nb = json.loads((PROJECT/'notebooks/attention_colab.ipynb').read_text())
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
assert len(cells)==6
assert not (PROJECT/'notebooks/attention_mechanism.ipynb').exists()
assert all(c['execution_count'] is None and c['outputs']==[] for c in nb['cells'] if c['cell_type']=='code')
for index, cell in enumerate(cells): compile(cell,f'colab-cell-{index+1}','exec')
archive_name = 'attention-is-all-you-need-source.zip'
manifest_name = 'source-manifest.json'
upload = {name:(EXPORT/name).read_bytes() for name in (archive_name,manifest_name)}
downloads=[]
files=types.SimpleNamespace(upload=lambda:upload,download=lambda p:downloads.append(Path(p)))
google=types.ModuleType('google');colab=types.ModuleType('google.colab');colab.files=files;google.colab=colab
results={'interface':'simulated google.colab.files; local execution, not hosted', 'cells':len(cells)}
mkdtemp=tempfile.mkdtemp

with tempfile.TemporaryDirectory(prefix='attention-colab-audit-') as temporary:
    namespace={}
    def local_directory(*args,**kwargs):
        if kwargs.get('dir')=='/content':kwargs['dir']=temporary
        return mkdtemp(*args,**kwargs)
    with patch.dict(sys.modules,{'google':google,'google.colab':colab}), patch('tempfile.mkdtemp',local_directory), patch.dict(os.environ,{'UV_LINK_MODE':'copy'}):
        with LOG.open('w') as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            for index, cell in enumerate(cells):
                print(f'Executing exact operational cell {index+1}',flush=True)
                exec(compile(cell,f'colab-cell-{index+1}','exec'),namespace)
        assert len(downloads)==1 and downloads[0].is_file()
        with zipfile.ZipFile(downloads[0]) as downloaded:
            assert set(downloaded.namelist())=={'source-manifest.json','environment.json','attention-demo.txt','tests.txt','tests.xml','run-receipt.json'}
            receipt=json.loads(downloaded.read('run-receipt.json'))
            assert receipt['demo_passed'] and receipt['tests_passed']==35 and receipt['tests_skipped']==0
            assert b'PASS:' in downloaded.read('attention-demo.txt')
            assert json.loads(downloaded.read('source-manifest.json'))==json.loads(upload[manifest_name])
            results.update(receipt=receipt,download_members=downloaded.namelist())
        for name,digest in json.loads(upload[manifest_name])['files'].items():
            assert hashlib.sha256((namespace['SOURCE']/name).read_bytes()).hexdigest()==digest
        # Repeat source upload in the same runtime: a fresh directory, no mixed tree.
        earlier=namespace['SOURCE']
        with contextlib.redirect_stdout(io.StringIO()):exec(cells[0],namespace)
        assert namespace['SOURCE']!=earlier and earlier.is_dir()
        results['repeat_upload_uses_fresh_directory']=True

        def reject(pair):
            files.upload=lambda:pair
            try:
                with contextlib.redirect_stdout(io.StringIO()):exec(cells[0],{})
            except ValueError:return True
            raise AssertionError('Invalid source was accepted')
        assert reject({archive_name:upload[archive_name]})
        assert reject({**upload,archive_name:upload[archive_name]+b'corrupt'})
        unsafe=io.BytesIO()
        with zipfile.ZipFile(unsafe,'w') as archive:archive.writestr('attention-is-all-you-need/../escape.py','bad')
        payload=unsafe.getvalue()
        fake={'archive_sha256':hashlib.sha256(payload).hexdigest(),'files':{'../escape.py':hashlib.sha256(b'bad').hexdigest()}}
        assert reject({archive_name:payload,manifest_name:json.dumps(fake).encode()})
        # Consistent archive hash still cannot conceal an incorrect member digest.
        mismatch=json.loads(upload[manifest_name]);first=next(iter(mismatch['files']));mismatch['files'][first]='0'*64
        assert reject({archive_name:upload[archive_name],manifest_name:json.dumps(mismatch).encode()})
        results['rejections']=['missing upload pair','archive hash mismatch','path traversal','file hash mismatch']

results.update(all_cells_executed=True,source_member_hashes_preserved=True,notebook_outputs_cleared=True,passed=True)
(HERE/'colab-runner-verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
