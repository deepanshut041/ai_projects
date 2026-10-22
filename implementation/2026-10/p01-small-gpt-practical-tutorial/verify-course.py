"""Rebuild the taught files in order, in a new disposable directory only."""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--lesson',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.lesson.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((source/'lesson.json').read_text())
pattern=r'<!-- file:(\S+) mode:(write|append) -->\n```\w+\n(.*?)\n```'
embedded=set(name for chapter in manifest['chapters'] for name,_,_ in re.findall(pattern,(source/chapter['content_path']).read_text(),re.S))
checks=[]
def run(arguments):
 result=subprocess.run([sys.executable,*arguments],cwd=out,capture_output=True,text=True,timeout=120)
 if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
 return result.stdout.strip()
for chapter in manifest['chapters']:
 text=(source/chapter['content_path']).read_text()
 for name,mode,code in re.findall(pattern,text,re.S):
  dest=out/name
  if mode=='append':code=dest.read_text()+'\n'+code
  dest.write_text(code+'\n')
  if name.endswith('.py'):ast.parse(code) # Every intermediate file must be syntactically complete.
 for name in re.findall(r'\]\(examples/([^)]*)\)',text):
  if name not in embedded and not (out/name).exists():shutil.copy2(source/'examples'/name,out/name)
 id=chapter['id'];output=[]
 if id=='next-token':output.append(run(['math_trace.py']))
 if id=='workspace':output.append(run(['-c','from gpt_config import Config; assert Config().width//Config().heads==16']))
 if id=='text':
  output.append(run(['-c',"from gpt_data import prepare,load_data; prepare('data/alice-byte'); s,t,m=load_data('data/alice-byte'); assert t.decode(t.encode('café 🙂'))=='café 🙂'"]))
 if id=='windows':output.append(run(['-m','unittest','test_gpt_data']))
 if id=='attention':output.append(run(['-c',"import torch; from gpt_attention import attend; q=torch.zeros(1,1,3,1); v=torch.tensor([2.,4.,8.]).view(1,1,3,1); torch.testing.assert_close(attend(q,q,v,torch.ones(3,3,dtype=torch.bool).tril()).flatten(),torch.tensor([2.,3.,14/3]))"]))
 if id=='model':output.append(run(['-c',"import torch; from gpt_model import GPT; m=GPT(); assert list(m(torch.tensor([[257,104,105]])).shape)==[1,3,260]; assert sum(p.numel() for p in m.parameters())==120832"]))
 if id=='learning':output.extend([run(['-m','unittest','test_gpt_data','test_gpt_model']),run(['fit_tiny.py'])])
 if id=='generation':output.append(run(['-c',"from gpt_model import GPT; from gpt_tokenizer import ByteBPE; from gpt_sample import generate; assert len(generate(GPT(),ByteBPE(),'hi',new_tokens=5)['generated_ids'])<=5"]))
 if id=='checkpoint':output.append(run(['-c','import gpt_checkpoint']))
 if id=='trainer':
  output.append(run(['-m','unittest','test_gpt_data','test_gpt_model','test_gpt_training']))
  for mode in ['new','resume']:output.append(run(['gpt_train.py','--data','data/alice-byte','--run','runs/staged','--config','cpu.json','--mode',mode,'--updates','2','--save-every','2']))
 if id=='colab':output.append(run(['export_source.py','exports/staged']))
 if id=='handoff':output.append(run(['gpt_profile.py']))
 checks.append({'lesson':id,'stage_passed':True,'commands':len(output)})
for name in embedded:
 if name.endswith('.py'):assert ast.dump(ast.parse((out/name).read_text()))==ast.dump(ast.parse((source/'examples'/name).read_text())),name
notebook=json.loads((out/'exports/staged/colab.ipynb').read_text())
for cell in notebook['cells']:
 if cell['cell_type']=='code':compile(''.join(cell['source']),'cell','exec');assert cell['outputs']==[]
report={'stages':checks,'embedded_files':sorted(embedded),'all_ast_match':True,'notebook_code_compiles':True,'learner_repository_created':False,'hosted_run':False}
(out/'course-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
