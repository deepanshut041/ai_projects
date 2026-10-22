"""Author checks in an ignored output directory. Run with the pinned Torch runtime."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--worker',choices=['full','part','resume'])
args=parser.parse_args();sys.path.insert(0,str(args.source.resolve()))
import torch
from test_gpt_training import fixture
from gpt_checkpoint import publish,recover

if args.worker:
    trainer,_=fixture()
    run=args.output/('full' if args.worker=='full' else 'split')
    if args.worker=='resume':
        state,_=recover(run,trainer.identity);trainer.restore(state)
    for _ in range(8 if args.worker=='full' else 4):trainer.update()
    trainer.validation();publish(run,trainer.state())
    raise SystemExit()
args.output.mkdir(parents=True,exist_ok=False)
for mode in ['full','part','resume']:
    subprocess.run([sys.executable,__file__,'--source',str(args.source),'--output',str(args.output),'--worker',mode],check=True)
trainer,_=fixture();a,_=recover(args.output/'full',trainer.identity);b,_=recover(args.output/'split',trainer.identity)
def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)
equal(a,b)
from gpt_config import Config
from gpt_model import GPT
from gpt_data import windows
from gpt_tokenizer import ByteBPE
from gpt_objective import nll_sum
from gpt_profile import profile
from export_source import export

from fit_tiny import fit
fitting = fit(300)
before, after = fitting['before_nll'], fitting['after_nll']
exports=export(args.source,args.output/'export')
notebook=json.loads(Path(exports['notebook']).read_text())
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        assert cell['outputs']==[] and cell['execution_count'] is None
        compile(''.join(cell['source']),'notebook-cell','exec')
report={'fresh_process_resume':{'full':8,'split':[4,4],'all_state_exact':True,'dropout':.1},
        'fixed_batch_overfit':{'updates':300,'before_nll':before,'after_nll':after,'target':.1,'passed':after<.1},
        'profile':[profile(32),profile(64)],'export':exports,'notebook_cells_compile':True,'hosted_run':False}
(args.output/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
