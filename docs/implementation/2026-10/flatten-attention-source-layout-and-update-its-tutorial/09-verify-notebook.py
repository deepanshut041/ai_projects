"""Execute the complete notebook from both supported working directories."""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[4]/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
SOURCE=PROJECT/'notebooks/attention_mechanism.ipynb'
os.environ['JUPYTER_PATH']=str(PROJECT/'.venv/share/jupyter')
records=[]
for cwd in (PROJECT,PROJECT/'notebooks'):
    notebook=nbformat.read(SOURCE,as_version=4)
    notebook.metadata.pop('widgets',None)
    for cell in notebook.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    client=NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(cwd)}})
    client.execute()
    code=[c for c in notebook.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
    assert len(code)==37 and all(c.execution_count is not None for c in code)
    assert not errors and images==28,(errors,images)
    assert not (PROJECT/'src/utils/runs').exists()
    assert not (PROJECT/'notebooks/runs').exists()
    assert len(list((PROJECT/'runs/attention-visuals').glob('[0-9][0-9]-*.png')))==28
    records.append({'cwd':str(cwd),'code_cells':len(code),'png_outputs':images,'errors':0,'passed':True})
    # Retain fresh executed output from the actual notebooks-directory run.
    nbformat.write(notebook,SOURCE)
result={'python':sys.executable,'runs':records,'figure_directory':str(PROJECT/'runs/attention-visuals'),'passed':True}
(HERE/'notebook-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
