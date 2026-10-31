"""Original, editable diagrams for the task-to-implementation attention route."""
from html import escape
from pathlib import Path

OUT = Path('/mnt/projects/ai/ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1/images')
INK = '#e8eef7'
BLUE = '#7cc8ff'
GREEN = '#84e2be'
RED = '#ffa3a3'

def txt(x, y, label, size=21, color=INK, anchor='middle'):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}">{escape(label)}</text>'

def box(x, y, w, h, lines, color=BLUE):
    value = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#18273b" stroke="{color}" stroke-width="2"/>'
    start = y + h/2 - (len(lines)-1)*14 + 7
    return value + ''.join(txt(x+w/2, start+28*i, s, color=color if i==0 else INK) for i,s in enumerate(lines))

def line(x1,y1,x2,y2,color=BLUE):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow)"/>'

def save(name, title, height, body):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="560" height="{height}" viewBox="0 0 560 {height}" role="img" aria-labelledby="title">
<title id="title">{escape(title)}</title>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="#7cc8ff"/></marker></defs>
<rect width="560" height="{height}" rx="12" fill="#101c2c"/>
<g font-family="DejaVu Sans, sans-serif">{body}</g></svg>''')

body=txt(280,32,'The whole translation process',23)
body+=box(40,54,480,60,['English source: We | read | books | daily'])
body+=line(280,114,280,140)
body+=box(40,142,480,70,['Build useful information from the source'])
body+=box(40,264,480,70,['French generated so far: Nous'],GREEN)
body+=line(280,334,280,370)
body+='<path d="M520,177 L540,177 L540,411 L520,411" fill="none" stroke="#7cc8ff" stroke-width="2.5" marker-end="url(#arrow)"/>'
body+=box(40,374,480,80,['Use source information + French so far','Predict the next token: lisons'])
body+=line(280,454,280,490)
body+=box(40,494,480,80,['Append that token to the output','French generated so far: Nous lisons'],GREEN)
body+='<path d="M40,534 L20,534 L20,299 L40,299" fill="none" stroke="#7cc8ff" stroke-width="2.5" marker-end="url(#arrow)"/>'
body+=txt(280,613,'Repeat using the growing French prefix.',21)
body+=txt(280,648,'Next we open one information-sharing operation.',19)
save('20-translation-task.svg','Read the English source and repeatedly choose and append a French token using the source and output so far',672,body)

body=txt(280,32,'A local transform has no cross-row path',22)
for x,i in [(15,0),(155,1),(295,2),(435,3)]:
    body+=box(x,58,110,52,[f'x{i}'])+line(x+55,110,x+55,147)+box(x,149,110,52,[f'f(x{i})'])
body+=txt(280,238,'Changing x0 cannot change f(x1).',21)
body+=txt(280,292,'Gathering creates the missing path',22)
for x,i,v in [(15,0,'[2,0]'),(155,1,'[0,2]'),(295,2,'[2,2]'),(435,3,'[0,0]')]:
    body+=box(x,316,110,70,[f'v{i}',v])+line(x+55,386,280,449)
body+=box(20,452,520,80,['c0 = a00 v0 + a01 v1 + a02 v2 + a03 v3','Each value can contribute to receiver 0'])
save('21-communication.svg','Independent row functions versus a gathered output with contributions from multiple rows',560,body)

body=txt(280,32,'One receiving row: position 0',23)
stages=[(['x0 = [1,0] → q0 = [1,0]'],70),
        (['K: [1,0] | [0,1] | [1,1] | [0,0]','q0 compared with each key: [1,0,1,0]'],90),
        (['Divide by sqrt(2)','Scaled scores: [0.707107,0,0.707107,0]'],90),
        (['Softmax across available positions','a0 ≈ [.334881,.165119,.334881,.165119]'],90),
        (['V: [2,0] | [0,2] | [2,2] | [0,0]','c0 = [1.339523,1.000000]'],90)]
y=55
for i,(labels,h) in enumerate(stages):
    body+=box(20,y,520,h,labels,GREEN if i==4 else BLUE)
    if i<len(stages)-1:body+=line(280,y+h,280,y+h+30)
    y+=h+32
save('22-one-query.svg','One source receiver through query key comparisons normalized weights and paired value contributions',y+4,body)

body=txt(280,32,'Every cell compares receiver i with source j',21)
body+=txt(310,70,'Source keys j = 0, 1, 2, 3',21)
for row,nums in enumerate([[1,0,1,0],[0,1,1,0],[1,1,2,0],[0,0,0,0]]):
    y=95+row*50
    body+=txt(76,y+30,f'q{row}',21)
    for col,n in enumerate(nums):
        x=145+col*90
        body+=f'<rect x="{x}" y="{y}" width="80" height="42" fill="{ "#284d69" if row==0 else "#18273b"}" stroke="#7cc8ff"/>'+txt(x+40,y+29,str(n))
body+=txt(280,326,'Scale and normalize across each row →',21)
body+=box(20,352,520,88,['A: (4 receivers, 4 available positions)','Row 0 ≈ [.335,.165,.335,.165]'])
body+=line(280,440,280,470)
body+=box(20,472,520,88,['C = A @ V: (4 receivers, 2 features)','Row 0: [1.339523,1.000000]'],GREEN)
body+=txt(280,598,'Source axis is summed. Receiver axis survives.',21)
save('23-matrix-axes.svg','Score cell ownership row wise normalization and retained output axes',625,body)

body=txt(280,32,'Split projected features; keep every position',22)
body+=box(20,58,520,70,['Projected Q/K/V: (B,T,2)','Reshape (B,T,2,1) → transpose (B,2,T,1)'])
body+=line(180,128,145,168)+line(380,128,415,168)
body+=box(20,172,250,170,['Head 0, receiver 0','q = [1]','[.366,.134]','repeated twice','context = 1.462117'])
body+=box(290,172,250,170,['Head 1, receiver 0','q = [0]','weights =','[1/4,1/4,1/4,1/4]','context = 1.000000'],GREEN)
body+=line(145,342,280,395)+line(415,342,280,395)
body+=box(20,398,520,80,['Transpose back, then join each token’s heads','Position 0: [1.462117,1.000000]'])
body+=line(280,478,280,508)
body+=box(20,510,520,62,['Apply output map W_O'])
save('24-head-layout.svg','Two one feature heads independently normalize weights then rejoin features within each position',600,body)

body=txt(280,32,'First four decoder training positions',23)
body+=txt(280,70,'Input: BOS | Nous | lisons | des',22)
body+=txt(280,105,'Label: Nous | lisons | des | livres',22,GREEN)
body+=txt(365,155,'Key input',21)
for col,label in enumerate(['BOS','Nous','lisons','des']):body+=txt(267+col*75,190,label,18)
for row,label in enumerate(['0: BOS','1: Nous','2: lisons','3: des']):
    y=208+row*62
    body+=txt(98,y+37,label,21)
    for col in range(4):
        x=232+col*75;ok=col<=row
        body+=f'<rect x="{x}" y="{y}" width="70" height="52" rx="5" fill="{ "#244c46" if ok else "#4c2938"}"/>'+txt(x+35,y+34,'read' if ok else 'block',18,GREEN if ok else RED)
body+=txt(280,510,'Row 1 must predict “lisons”.',22)
body+=txt(280,544,'Reading input key 2 would reveal that label.',20,RED)
body+=txt(280,582,'Allow j ≤ i; shift the labels separately.',21)
save('25-decoder-causality.svg','Shifted translation labels and allowed decoder pairs show the future token leakage path',612,body)

body=txt(280,32,'Cross-attention changes vector origins',23)
body+=box(20,58,520,80,['Encoder memory: 4 source rows','K: (4,2) and V: (4,2)'])
body+=box(20,172,520,80,['Causal decoder states: 2 receiving rows','Q: (2,2)'],GREEN)
body+='<path d="M540,97 L550,97 L550,315 L540,315" fill="none" stroke="#7cc8ff" stroke-width="2.5" marker-end="url(#arrow)"/>'
body+=line(280,252,280,283)
body+=box(20,286,520,90,['Q @ K.T → weights (2,4)','Each target row compares all 4 source keys'])
body+=line(280,376,280,405)
body+=box(20,408,520,90,['weights @ V → context (2,2)','2 receivers each get 2 gathered features'],GREEN)
body+=txt(280,538,'Output length follows Q, not source memory.',21)
body+=txt(280,574,'Key validity has 4 entries: one per source row.',20)
save('26-cross-attention.svg','Two target receivers read four source memory positions producing two output rows',600,body)

body=txt(280,32,'Full-model learning path',23)
stages=[['Q/K/V parameters → gathered context C'],['Decoder computation → h_t'],['Vocabulary projection → logits z_t'],['Softmax → next-token distribution p_t'],['Reference token: lisons','Loss = −log p_t[lisons]']]
y=58
for i,labels in enumerate(stages):
    body+=box(40,y,480,80,labels,GREEN if i==4 else BLUE)
    if i<4:body+=line(280,y+80,280,y+108)
    y+=112
body+='<path d="M40,546 L15,546 L15,98 L40,98" fill="none" stroke="#ffa3a3" stroke-width="3" marker-end="url(#arrow)"/>'
body+=txt(280,647,'Backward: task error changes shared parameters.',20,RED)
body+=txt(280,683,'Local probe isolates C → squared error only.',20)
save('27-learning-path.svg','Translation loss supplies gradients through vocabulary decoder and attention paths while the local probe isolates retrieval',710,body)

print('Wrote eight original SVG figures.')
