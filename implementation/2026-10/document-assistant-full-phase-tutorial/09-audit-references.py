"""Independent arithmetic, failure paths and teaching-order reconstruction."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,re,runpy,subprocess,tempfile,sys,math
ROOT=Path('/mnt/projects/ai/ai_projects/docs/tutorials/projects/document-assistant/quantization-lab/v1')
checks=[]
def check(name,condition,scope):
 if not condition:raise AssertionError(name)
 checks.append(dict(name=name,passed=True,scope=scope))
def load(n):
 return runpy.run_path(str(next((ROOT/'examples').glob(f'{n:02}_*.py'))))
# Rational arithmetic uses exact decimal fractions rather than copying float outputs.
w=[[F(24,100),F(102,100)],[-F(76,100),F(49,100)]]
q=[[F(0),F(1)],[-F(1),F(1,2)]]
x=[F(4),F(1)]
y=[sum(a*b for a,b in zip(row,x)) for row in w]
delta=[sum((a-b)*c for a,b,c in zip(qr,wr,x)) for qr,wr in zip(q,w)]
check('exact-layer-values',y==[F(198,100),-F(255,100)] and delta==[-F(98,100),-F(95,100)],'Independent rational matrix products')
check('rational-affine-endpoints',F(6,7)*(0-2)==-F(12,7) and F(6,7)*(7-2)==F(30,7),'Endpoint and zero-point arithmetic')
check('exact-packing',[((a&15)|((b&15)<<4)) for a,b in [(-8,-1),(0,7),(3,0)]]==[248,112,3],'Independent Python nibble reconstruction vs C++ output')
check('exact-scale-loss',F(96,100)**2==F(9216,10000) and F(6,100)**2==F(36,10000),'Exact squares for ternary scale example')
# A changed input, not the fixture's original vector.
check('changed-layer-input',[sum((a-b)*c for a,b,c in zip(qr,wr,[F(1),F(4)])) for qr,wr in zip(q,w)]==[-F(32,100),-F(20,100)],'Transfer answer independently computed')
check('compensation-cost',4*F(1,4)**2+4*F(1,4)*-F(1,4)+2*(-F(1,4))**2==F(1,8),'Exact constrained quadratic loss')
check('residual-cache-ratio',(F(896,4)+F(128*2))/(1024*2)==F(15,64),'Two-bit old region plus BF16 residual, excludes metadata')
check('cache-logical-sizes',2*2*512*64*2+2*2*2048*128*2==2359296,'Independent two-owner allocation')
# Finite differences check the softmax derivative used in the cache lesson.
sm=load(7)['softmax'];z=[0.0,math.log(3)];p=sm(z);eps=1e-5
for i in range(2):
 for j in range(2):
  plus=z.copy();minus=z.copy();plus[j]+=eps;minus[j]-=eps
  fd=(sm(plus)[i]-sm(minus)[i])/(2*eps)
  check(f'softmax-jacobian-{i}-{j}',abs(fd-p[i]*((i==j)-p[j]))<1e-9,'Independent numerical differentiation')
cal=load(12)
for invalid in [[],[[]],[[1,2],[1]]]:
 try:cal['gram'](invalid)
 except ValueError:pass
 else:raise AssertionError('Bad calibration matrix accepted')
check('calibration-input-validation',True,'Empty/ragged failures exercised')
samples=[[4,1],[4,-1]]
check('duplicated-mean-invariance',cal['gram'](samples)==cal['gram'](samples*2),'Denominator transfer case')
sc=load(8)
for args in [([1],1,1,4),([1],1,2,4)]:
 try:sc['affine'](*args)
 except ValueError:pass
 else:raise AssertionError('Invalid quantizer range accepted')
check('quantizer-range-validation',True,'Degenerate/range-not-containing-zero rejected')
select=load(23)
check('all-infeasible-empty',select['choose']([],8,.9) is None,'No invented candidate when the set is empty')
# Complete listing extraction into an entirely new directory, in teaching order.
manifest=json.loads((ROOT/'lesson.json').read_text());records=[]
with tempfile.TemporaryDirectory(prefix='document-assistant-from-lessons-') as temp:
 for chapter,example in zip(manifest['chapters'],manifest['examples']):
  lesson=(ROOT/chapter['content_path']).read_text();raw=(ROOT/example['path']).read_text()
  blocks=re.findall(r'^```(?:python|cpp)\n(.*?)^```',lesson,re.M|re.S)
  matching=[block for block in blocks if block.rstrip()==raw.rstrip()]
  check('listing-'+chapter['id'],len(matching)==1,'Displayed complete file exactly matches download')
  destination=Path(temp)/Path(example['path']).name;destination.write_text(matching[0])
  argv=[sys.executable,str(destination)]
  if example['language']=='cpp':
   binary=Path(temp)/'pack_int4'
   subprocess.run(['c++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-pedantic',str(destination),'-o',str(binary)],check=True)
   argv=[str(binary)]
  result=subprocess.run(argv,cwd=temp,check=True,text=True,capture_output=True,timeout=15)
  check('isolated-'+chapter['id'],json.loads(result.stdout)==example['expected_output'],'Teaching-order extraction, no sibling imports or checkout dependency')
  records.append(dict(chapter=chapter['id'],sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),passed=True))
 # Verify toy generation loading from a genuinely separate process.
 recovery=load(20);storage=Path(temp)/'storage';storage.mkdir();recovery['publish'](storage,'gen-0001',b'persisted fixture')
 script='import runpy,json,sys; m=runpy.run_path(sys.argv[1]); name,raw,rejected=m["load_latest"](__import__("pathlib").Path(sys.argv[2])); print(json.dumps([name,raw.decode(),rejected]))'
 probe=subprocess.run([sys.executable,'-c',script,str(ROOT/'examples/20_recovery.py'),str(storage)],text=True,capture_output=True,check=True)
 check('fresh-process-toy-reload',json.loads(probe.stdout)==['gen-0001','persisted fixture',[]],'Toy byte integrity only; not model recovery')
# Real read-only scaffold execution, separate from portable default mode.
probe=subprocess.run([sys.executable,str(ROOT/'examples/02_scaffold_trace.py'),'/mnt/projects/ai/ai_projects/projects/quantization-lab/document-assistant'],text=True,capture_output=True,check=True)
scaffold=json.loads(probe.stdout)
check('live-scaffold-actions',[v['exit'] for v in scaffold.values()]==[0,0,0,2],'Actual project status/check/config and pending-answer boundary')
record=dict(schema_version=1,scope='Authored tutorial references and existing read-only scaffold; no model weights/inference',checks=checks,extracted=records,live_scaffold=scaffold,passed=True)
(ROOT/'evidence/numerical-audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),extracted=len(records),live_scaffold=True)))
