"""Draw original passive, local figures from the authored numerical fixtures."""
from pathlib import Path
from html import escape
ROOT=Path('/mnt/projects/ai/ai_projects/docs/tutorials/projects/document-assistant/quantization-lab/v1/figures')
ROOT.mkdir(exist_ok=True)
def text(x,y,s,size=21,color='#18263c'):
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(s)}</text>'
def box(x,y,w,h,label,detail='',color='#e4efff'):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="#7892b5"/>'+text(x+16,y+34,label)+''.join(text(x+16,y+65+i*24,part,18) for i,part in enumerate(detail.split('\n')))
def line(x,y,x2,y2):
 return f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="#486687" stroke-width="3" marker-end="url(#arrow)"/>'
def draw(name,title,desc,body,h=410):
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{h}" viewBox="0 0 900 {h}" role="img" aria-labelledby="title description"><title id="title">{escape(title)}</title><desc id="description">{escape(desc)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#486687"/></marker></defs><rect width="900" height="{h}" fill="#f8fbff"/><g font-family="sans-serif">'+text(28,42,title,27)+body+'</g></svg>'
 (ROOT/(name+'.svg')).write_text(svg)
draw('evidence-flow','One answer, three separate checks','A correct answer can cite an existing passage that does not support the claim.',box(30,85,250,100,'Question','Full charge duration?')+line(285,135,325,135)+box(335,85,530,100,'Response: 3 hours; citation: lumen-2','lumen-2 describes the power button')+box(30,235,255,100,'Correct fact: yes','3 hours matches gold','#e0f5e9')+box(320,235,255,100,'Valid ID: yes','lumen-2 exists','#e0f5e9')+box(610,235,255,100,'Supported: no','Wrong evidence','#ffe7df')+text(30,382,'Acceptance requires the claim AND the evidence contract.',22))
b=''
items=[('weights',4.191,'#8cb6f5'),('cache',2,'#8fd2b1'),('workspace',1,'#f0cf7b'),('reserve',2,'#c5b4eb')]
x=35
for label,value,color in items:
 width=value/12*825
 b+=f'<rect x="{x}" y="100" width="{width}" height="72" fill="{color}" stroke="#fff"/>'
 x+=width
for i,(label,value,color) in enumerate(items):
 b+=box(35+(i%2)*430,215+(i//2)*88,400,75,label+f': {value:.3f} GiB','Illustrative allowance',color)
b+=text(35,193,'9.191 GiB including reserve; 12 GiB illustrative capacity',20)
draw('memory-budget','Weight storage does not equal request memory','A budget stacks weights, cache, workspace and reserve. A longer cache can exceed capacity.',b,h=420)
b=line(70,155,835,155)
for q in range(8):
 x=90+q*90
 b+=f'<circle cx="{x}" cy="155" r="7" fill="#386ab0"/>'+text(x-12,120,str(q),20)+text(x-32,200,f'{(6/7)*(q-2):.2f}',18)
b+=text(35,255,'Code q (top) → reconstructed value (bottom)',21)+text(35,297,'s = 6/7; z = 2; zero is exact, endpoints shift.',21)+text(35,343,'Input 7 clips to q=7 → 4.286; its original magnitude is lost.',21)
draw('quantizer-grid','Three-bit affine grid: eight possible reconstructions','Codes zero through seven reconstruct values from minus 1.714 to 4.286. Input seven clips to the largest code.',b)
draw('layer-error','The input determines which weight errors matter','The reconstructed weight error matrix times the input gives output errors minus 0.98 and minus 0.95.',box(30,95,300,110,'Weight error E','[−0.24, −0.02]\n[−0.24, +0.01]')+text(350,160,'×',32)+box(390,95,170,110,'x','[4, 1]')+text(580,160,'=',32)+box(630,95,235,110,'Output error','[−0.98, −0.95]','#ffe7df')+text(35,270,'Row 1: (−0.24 × 4) + (−0.02 × 1) = −0.98',23)+text(35,320,'Row 2: (−0.24 × 4) + ( 0.01 × 1) = −0.95',23)+text(35,375,'The same weight error has a larger effect on a larger input.',21))
draw('calibration','Equal weight MSE, unequal output error','Two equal-size row errors have different output mean squared error when calibration inputs amplify the first channel.',box(30,90,400,110,'Calibration inputs','[4, 1] and [4, −1]')+box(470,90,400,110,'Gram matrix G','[16, 0]; [0, 1]')+box(30,245,400,110,'Error A = [0.25, 0]','Output MSE = 1.0000','#ffe7df')+box(470,245,400,110,'Error B = [0, 0.25]','Output MSE = 0.0625','#e0f5e9')+text(30,390,'Both have weight MSE 0.03125. Inputs expose the difference.',20))
b=line(55,150,850,150)
for x,label,detail in [(70,'request','t0=0'),(250,'submit','ts=0.2'),(430,'first token','t1=0.7'),(810,'last token','tn=2.7')]:
 b+=f'<line x1="{x}" y1="136" x2="{x}" y2="165" stroke="#244866" stroke-width="3"/>'+text(x-40,95,label,20)+text(x-35,200,detail,18)
b+=text(35,265,'User TTFT = 0.7 s; runtime TTFT = 0.5 s',23)+text(35,315,'Five tokens create four intervals after the first token.',23)+text(35,365,'Decode rate = (5 − 1) / (2.7 − 0.7) = 2 tokens/s',23)
draw('timing','Measure a declared interval','An invented timeline distinguishes user and runtime time to first token from inter-token decode throughput.',b)
draw('cache-owners','Logical cache counts distinct owners','Two layers share a sliding cache owner, while a full-context owner grows with sequence length.',box(30,90,245,90,'Layer 1','uses owner s')+box(30,210,245,90,'Layer 2','uses owner s')+line(285,140,380,165)+line(285,255,380,200)+box(395,115,470,110,'Owner s: sliding window 512','262,144 bytes once T ≥ 512','#e0f5e9')+box(30,335,245,90,'Layer 3','uses owner f')+line(285,380,380,380)+box(395,325,470,110,'Owner f: full history','2,097,152 bytes at T=2048')+text(30,480,'At T=2048: total 2,359,296 bytes. This is not the Gemma map.',20),h=510)
draw('recovery','Choose a verified generation, disclose fallback','A loader ignores an incomplete staging directory, rejects a corrupt newest generation and loads the older verified one.',box(30,90,840,80,'gen-0003.partial — ignore','Not published')+box(30,195,840,80,'gen-0002 — reject','Saved bytes fail size/hash check','#ffe7df')+box(30,300,840,80,'gen-0001 — load and record fallback','Complete manifest and bytes verified','#e0f5e9'),h=420)
draw('scaling','Exact float transform; different quantized errors','Scaling weights and inversely scaling input preserves the floating output, but affects quantization grid loss.',box(30,95,400,110,'Original: w=[0.24,1.02]','x=[4,1] → output 1.98')+box(470,95,400,110,'Scale [4,1]: w=[0.96,1.02]','x=[1,1] → output 1.98')+text(35,275,'Ternary reconstruction, two calibration inputs:',22)+text(35,320,'Scale 1 → MSE 0.9216; scale 4 → MSE 0.0036',22)+text(35,365,'Scale 8 → MSE 0.81. Larger scaling can harm other channels.',21))
draw('compensation','Fix one error; optimize the remaining coordinate','For positive correlated channels, a negative second error reduces the constrained quadratic loss.',box(30,100,400,110,'Fixed e1 = +0.25','No compensation: e2=0; L=0.25','#ffe7df')+box(470,100,400,110,'Choose e2 = −0.25','Compensated loss L=0.125','#e0f5e9')+text(35,280,'L = 4 e1² + 4 e1 e2 + 2 e2²',24)+text(35,330,'∂L/∂e2 = 4 e1 + 4 e2 = 0 → e2 = −e1',24)+text(35,380,'Later quantizing e2 can lose part of this continuous optimum.',20))
print('Drew',len(list(ROOT.glob('*.svg'))),'original figures')
