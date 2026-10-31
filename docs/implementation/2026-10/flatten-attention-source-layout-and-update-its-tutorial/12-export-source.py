"""Refresh the documented source ZIP from the current captured project."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
EXPORT=ROOT/'.ai/exports/attention-is-all-you-need'
RECOVERY=ROOT/'.ai/attention-layout/2026-10-23'
EXPORT.mkdir(parents=True,exist_ok=True)
for name in ('attention-is-all-you-need-source.zip','source-manifest.json'):
    original=EXPORT/name;old=RECOVERY/('old-export-'+name)
    if original.is_file() and not old.exists():shutil.copyfile(original,old)
bundle=json.loads((COURSE/'code/source-bundle.json').read_text())
payload={name:item['text'].encode('utf-8') for name,item in bundle['files'].items()}
for name in ('docs/SOURCES.md','docs/COLAB.md','docs/sources.json'):payload[name]=(PROJECT/name).read_bytes()
shutil.copyfile(PROJECT/'notebooks/attention_colab.ipynb',EXPORT/'attention_colab.ipynb')
archive=EXPORT/'attention-is-all-you-need-source.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as output:
    for name,raw in payload.items():
        entry=zipfile.ZipInfo('attention-is-all-you-need/'+name,date_time=(2026,10,23,0,0,0))
        entry.compress_type=zipfile.ZIP_DEFLATED;output.writestr(entry,raw)
with zipfile.ZipFile(archive) as check:
    for name,raw in payload.items():assert check.read('attention-is-all-you-need/'+name)==raw
manifest={'source_revision':subprocess.check_output(['git','-C',str(ROOT/'ai_projects'),'rev-parse','HEAD'],text=True).strip(),'working_tree':True,'layout':'src/core,data,training,utils; entry scripts directly in src; Colab runner outside src','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in payload.items()},'excludes':['raw captions','environments','data','weights','runtime runs','private study notes']}
(EXPORT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'export-verification.json').write_text(json.dumps({'archive':str(archive),'files':len(payload),'archive_sha256':manifest['archive_sha256'],'member_bytes_verified':True,'passed':True},indent=2)+'\n')
print(json.dumps({'archive':str(archive),'files':len(payload),'passed':True}))
