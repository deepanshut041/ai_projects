"""Execute current complete lesson snippets against the captured grouped source."""
import json,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
results={}
prologue='import sys;sys.path.insert(0,'+repr(str(COURSE/'code/src'))+');'
for name in ('01-weighted-information.md','10-reusable-attention.md','11-gradients-and-learning.md'):
 blocks=re.findall(r'^```python\n(.*?)^```',(COURSE/'lessons'/name).read_text(),re.M|re.S)
 assert len(blocks)==1
 run=subprocess.run([sys.executable,'-I','-c',prologue+'exec('+repr(blocks[0])+')'],capture_output=True,text=True,check=True)
 results[name]=run.stdout.strip()
count=0
for path in (COURSE/'lessons').glob('*.md'):
 for code in re.findall(r'^```python\n(.*?)^```',path.read_text(),re.M|re.S):compile(code,str(path),'exec');count+=1
output={'snippets':results,'compiled_blocks':count,'passed':True}
(HERE/'teaching-verification.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
