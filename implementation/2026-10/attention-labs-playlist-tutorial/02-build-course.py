"""Assemble authored teaching files, numerical references and passive figures."""
from pathlib import Path
import hashlib
import html
import json
import re
import subprocess
import xml.etree.ElementTree as ET

DOCS = Path(__file__).resolve().parents[3]
BASE = DOCS / 'tutorials/projects/attention-labs'
SOURCE = DOCS.parents[1] / '.ai/attention-labs-source/2026-10-30'
INVENTORY = json.loads((SOURCE / 'inventory.json').read_text())
VIDEOS = {x['lecture']: x for x in INVENTORY if x['lecture'] <= 8}
SUBTITLES = {x['lecture']: x for x in json.loads((SOURCE / 'subtitle-provenance.json').read_text())['subtitles']}
assert set(SUBTITLES) == set(range(1, 9))
for subtitle in SUBTITLES.values():
    assert hashlib.sha256((SOURCE / subtitle['file']).read_bytes()).hexdigest() == subtitle['sha256']


def svg(path, title, body, height=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = f'''<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{height}" viewBox="0 0 920 {height}" role="img" aria-label="{html.escape(title)}">
<title>{html.escape(title)}</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#376b66"/></marker></defs>
<rect width="920" height="{height}" rx="12" fill="#f8fbfa"/>
<style>text{{font-family:Arial,sans-serif;fill:#173d3b;font-size:22px}}.heading{{font-size:27px;font-weight:bold}}.small{{font-size:18px}}.box{{fill:#e1f2ec;stroke:#4f8a7d;stroke-width:2}}.arrow{{stroke:#376b66;stroke-width:3;fill:none;marker-end:url(#arrow)}}</style>
<text x="30" y="45" class="heading">{html.escape(title)}</text>{body}</svg>'''
    tree = ET.fromstring(raw)
    styles = {
        'heading': {'font-size': '27', 'font-weight': 'bold'},
        'small': {'font-size': '18'},
        'box': {'fill': '#e1f2ec', 'stroke': '#4f8a7d', 'stroke-width': '2'},
        'arrow': {'stroke': '#376b66', 'stroke-width': '3', 'fill': 'none', 'marker-end': 'url(#arrow)'},
    }
    for element in tree.iter():
        if element.tag.endswith('}text'):
            element.attrib.update({'font-family': 'Arial,sans-serif', 'fill': '#173d3b', 'font-size': '22'})
        element.attrib.update(styles.get(element.attrib.pop('class', ''), {}))
    for child in list(tree):
        if child.tag.endswith('}style'):
            tree.remove(child)
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    path.write_text(ET.tostring(tree, encoding='unicode'))


def box(x, y, width, label, subtitle='', height=85):
    return (f'<rect class="box" x="{x}" y="{y}" width="{width}" height="{height}" rx="8"/>'
            f'<text x="{x + 16}" y="{y + 34}">{html.escape(label)}</text>'
            f'<text x="{x + 16}" y="{y + 64}" class="small">{html.escape(subtitle)}</text>')


def line(x1, y1, x2, y2):
    return f'<path class="arrow" d="M{x1} {y1}L{x2} {y2}"/>'


def images():
    root = BASE / 'attention-foundations/v1/images'
    body = box(30, 100, 210, 'Input positions', 'token feature rows') + line(245, 142, 325, 142)
    body += box(330, 100, 240, 'Retrieve context', 'compare → weight → mix') + line(575, 142, 655, 142)
    body += box(660, 100, 230, 'Predict a token', 'vocabulary logits')
    body += '<text x="40" y="270">Attention returns features; a separate head predicts vocabulary tokens.</text><text x="40" y="315">A new query can retrieve a new mixture from the same stored content.</text>'
    svg(root / '01-context-map.svg', 'From input information to a contextual prediction', body, 365)
    body = ''
    for i, (name, sub) in enumerate([('h₁', 'first input'), ('h₂', 'prefix summary'), ('h₃ = c', 'fixed handoff')]):
        body += box(50 + i * 280, 110, 180, name, sub)
        if i < 2: body += line(235 + i * 280, 152, 322 + i * 280, 152)
    body += '<text x="50" y="270">Toy recurrence: hₜ = 0.5 hₜ₋₁ + xₜ</text><text x="50" y="315">Inputs [1, 0, 2] → states [1, 0.5, 2.25]</text><text x="50" y="360">Earlier input coefficient at step 3: 0.5 × 0.5 = 0.25</text>'
    svg(root / '02-fixed-summary.svg', 'All source information crosses one final summary', body)
    body = box(30, 105, 220, 'Query + memory', 's and source rows hⱼ') + line(255, 147, 315, 147)
    body += box(320, 105, 250, 'Compatibility', 'uᵀ tanh(Wₛs + Wₕhⱼ)') + line(575, 147, 645, 147)
    body += box(650, 105, 240, 'Normalize', 'one weight per source')
    body += box(320, 270, 300, 'Return a context', 'c = Σⱼ αⱼ hⱼ') + '<path class="arrow" d="M770 195V305H628"/>'
    body += '<text x="40" y="405" class="small">Memory stays fixed while the query changes the weights and context.</text>'
    svg(root / '03-additive-flow.svg', 'A query-dependent context from retained states', body)
    body = '<text x="45" y="100">Scores [1, 1, 2], key width 2</text>'
    for y, label, values in [(150, 'Unscaled', [0.211942, 0.211942, 0.576117]), (290, 'Divide by √2', [0.248255, 0.248255, 0.50349])]:
        body += f'<text x="45" y="{y + 25}">{label}</text>'
        for j, value in enumerate(values):
            x = 280 + j * 200
            body += f'<rect x="{x}" y="{y + 75 - value * 120}" width="90" height="{value * 120}" fill="#4b9480"/><text x="{x}" y="{y + 105}" class="small">{value:.6f}</text>'
    svg(root / '04-scale.svg', 'Scaling changes concentration, not the ranking', body)
    body = '<text x="310" y="95">Source position (key/value)</text>'
    weights = [[.422319, .155362, .422319], [.155362, .422319, .422319], [.211942, .211942, .576117]]
    for i, row in enumerate(weights):
        body += f'<text x="30" y="{170 + 90 * i}">Query {i + 1}</text>'
        for j, val in enumerate(row):
            x, y = 220 + j * 205, 120 + i * 90
            body += f'<rect x="{x}" y="{y}" width="180" height="70" rx="5" fill="#c1dfd2"/><text x="{x + 20}" y="{y + 44}">{val:.6f}</text>'
    body += '<text x="220" y="420" class="small">Normalize each query row over allowed source positions.</text>'
    svg(root / '05-self-matrix.svg', 'Self-attention keeps one distribution per query row', body, 455)
    body = box(30, 190, 140, 'X', '3 × 2')
    for y, label, sub in [(100, 'Q = XWQ', 'comparison requests'), (230, 'K = XWK', 'comparison features'), (360, 'V = XWV', 'delivered content')]:
        body += line(175, 230, 260, y + 40) + box(265, y, 270, label, sub)
    body += box(640, 160, 250, 'A = softmax(QKᵀ/√2)', '3 × 3 weights', height=95)
    body += line(540, 140, 635, 198) + line(540, 270, 635, 230)
    body += box(640, 360, 250, 'Z = AV', '3 × 2 output') + line(765, 260, 765, 350) + line(540, 400, 635, 400)
    svg(root / '06-projections.svg', 'Routing features and delivered features take different paths', body, 490)
    body = '<text x="280" y="100">Key 1</text><text x="465" y="100">Key 2</text><text x="650" y="100">Key 3</text>'
    for i in range(3):
        body += f'<text x="40" y="{170 + i * 90}">Query {i + 1}</text>'
        for j in range(3):
            x, y = 250 + 185 * j, 125 + 90 * i
            body += f'<rect x="{x}" y="{y}" width="155" height="65" rx="5" fill="{"#b9dcca" if j <= i else "#f0d7d7"}"/><text x="{x + 18}" y="{y + 41}">{"allowed" if j <= i else "blocked"}</text>'
    body += '<text x="40" y="430" class="small">Allowed = source is real AND source position ≤ query position.</text>'
    svg(root / '07-causal-mask.svg', 'A causal mask permits a prefix, including the diagonal', body, 470)
    root = BASE / 'attention-heads/v1/images'
    body = box(30, 200, 130, 'X', 'T × D')
    body += line(165, 240, 245, 140) + line(165, 240, 245, 330)
    body += box(250, 100, 230, 'Head 1', 'own Q, K, V and A₁') + box(250, 290, 230, 'Head 2', 'own Q, K, V and A₂')
    body += line(485, 140, 565, 235) + line(485, 330, 565, 260)
    body += box(570, 205, 160, 'Concat', 'T × (H dᵥ)') + line(735, 247, 775, 247)
    body += box(780, 205, 110, 'Wₒ', 'mix heads')
    body += '<text x="40" y="430" class="small">Different attention matrices can route different value channels independently.</text>'
    svg(root / '01-heads.svg', 'Independent mixtures are joined, then mixed across features', body, 480)
    body = ''
    stages = [('Project', '[B,T,D]'), ('Split', '[B,T,H,dₕ]'), ('Transpose', '[B,H,T,dₕ]'), ('Attend', '[B,H,T,T]')]
    for i, (label, sub) in enumerate(stages):
        body += box(25 + i * 225, 110, 195, label, sub)
        if i < 3: body += line(224 + i * 225, 151, 245 + i * 225, 151)
    body += '<text x="40" y="265">Value mixing → [B,H,T,dₕ]</text><text x="40" y="310">Transpose back → [B,T,H,dₕ] → merge → [B,T,D]</text><text x="40" y="355">Q/K contraction uses the feature axis; softmax uses the source-token axis.</text>'
    svg(root / '02-packed-shapes.svg', 'Packing exposes heads as a batch-like axis', body, 410)


PRACTICE = {
    'map': ('A layer is causal, uses a dot-product score and has multiple heads. Is that contradictory?', 'No. The score defines compatibility, causality defines permitted source positions, and the head arrangement defines separate or shared projections. These choices can coexist. Attention weights are over source positions; vocabulary probabilities belong to a later prediction head.', 'Explain the three different questions and keep source positions separate from vocabulary entries.'),
    'summary': ('In the scalar encoder, double only the first input. Why does the last state change by 0.25, and must every real RNN behave that way?', 'The effect passes through two multiplications by 0.5, giving 0.25. A real RNN can have different matrices, nonlinearities and gates, so this is an illustrative sensitivity calculation rather than a universal memory law. A decoder still uses a separate vocabulary head and can produce a different number of tokens.', 'Trace both intervening updates and state the limitation of the scalar example.'),
    'additive': ('If tanh is removed from uᵀ(Wₛs + Wₕh), can changing only s change the softmax weights?', 'Not in this separable scorer: uᵀWₛs adds the same scalar to every source score for a fixed query. Softmax is invariant to that common offset, so the relative weights do not change. Nonlinear interaction makes the query contribution affect relative comparisons.', 'Show the common-offset cancellation; distinguish a new query from updating model parameters.'),
    'scale': ('The number of keys doubles but their width stays two. Should the conventional scale change from √2 to √4?', 'No. The scale uses the number of summed coordinate products in each dot product, d_k=2, rather than the number of keys. More keys changes the number of scores and the normalization denominator, not the coordinate count in a score.', 'Identify the key-feature width and separate it from the source-position axis.'),
    'self': ('XXᵀ is symmetric. Must its row-softmax also be symmetric? What does a zero query do?', 'No: rows can have different normalization denominators. In this fixture A_13≈0.422319 while A_31≈0.211942. A real zero query gives equal finite scores, hence uniform weights over allowed keys; it does not declare padding.', 'Use the unequal pair of entries and explain the explicit validity-mask requirement.'),
    'projection': ('Change W_V while keeping X, W_Q and W_K fixed. What must stay fixed, and what can change?', 'Q, K, scores and attention weights stay fixed. V and the returned contexts can change. With a task loss, downstream gradients can also change because the output being measured changed. Identical position-free queries with the same memory still cannot be distinguished by merely naming the roles differently.', 'Separate the routing path from the content path; do not treat attention weights as vocabulary predictions.'),
    'causal': ('A dropout sample has a weight above one. Is it invalid? Also, does zeroing and renormalizing unmasked softmax necessarily leak future scores?', 'A dropout sample need not sum to one, and surviving weights are scaled by 1/(1-p); values above one are possible. The local expectation matches the original weights. Zeroing forbidden weights and renormalizing is mathematically equivalent to masked softmax when the allowed denominator is nonzero, but it is numerically fragile if allowed probabilities have already underflowed. Simply zeroing without renormalizing is a different operation.', 'State both the sample-versus-expectation distinction and the exact-arithmetic masking equivalence.'),
    'heads': ('After concatenating two scalar head outputs, does W_O mix tokens or head features?', 'It mixes head-feature channels separately at each token. Token-to-token mixing already occurred inside each attention head. In our first row, [1,1] multiplied by the declared W_O becomes [0.5,1.5]. Multiple heads allow different distributions but do not guarantee unique linguistic roles.', 'Identify the feature contraction and reproduce the small output-projection calculation.'),
    'layout': ('Why is [B,T,H,d_h] the wrong layout for Q @ K.transpose(-2,-1)?', 'Its last two axes are heads and features. That product compares heads for each token, producing [B,T,H,H]. Transposing to [B,H,T,d_h] makes the contraction produce [B,H,T,T], one token-by-token score matrix per head and sequence. Merging outputs also requires restoring token/head order, not merely matching element counts.', 'Name the query-token and source-token axes before multiplication and the feature order after merging.'),
    'boundary': ('Does packing query/key/value projections implement GQA or reduce the count of stored KV heads?', 'No. Our packed MHA preserves separate Q/K/V parameters per head and H_q=H_kv. GQA deliberately shares key/value heads among query heads, which needs an explicit grouping rule and corresponding correctness checks. Neither packing nor grouping automatically removes the sequence-length cost of the score calculation.', 'Separate an equivalent tensor layout from a changed head-sharing relationship.'),
}


def package(slug, files, objective_keys, lectures):
    root = BASE / slug / 'v1'
    data = json.loads((root / 'lesson.json').read_text())
    data['prerequisites'] = ['Vectors, matrix multiplication, a model parameter, and the role of a loss. No Python syntax course.']
    if slug == 'attention-heads':
        data['prerequisites'].append('The single-head projections, stable softmax and causal rule from Attention Foundations.')
    data['objectives'] = []
    data['chapters'] = []
    data['practice'] = []
    for index, (filename, key) in enumerate(zip(files, objective_keys)):
        path = root / 'lessons' / filename
        text = path.read_text()
        def code(match):
            name = match.group(1)
            raw = (root / 'examples' / name).read_text().rstrip()
            language = 'cpp' if name.endswith('.cpp') else 'python'
            return f'<!-- example:{name}:start -->\n\nComplete file: [{name}](examples/{name}).\n\n```{language}\n{raw}\n```\n\n<!-- example:{name}:end -->'
        text = re.sub(r'<!-- example:([^:]+):start -->.*?<!-- example:\1:end -->', code, text, flags=re.S)
        text = re.sub(r'\{\{code:([^}]+)\}\}', code, text)
        path.write_text(text)
        title = text.splitlines()[0].removeprefix('## ')
        data['objectives'].append({'id': key, 'text': title})
        data['chapters'].append({'id': key, 'title': title, 'content_path': f'lessons/{filename}',
                                 'objective_ids': [key], 'diagram_ids': [], 'table_ids': [],
                                 'scenario_ids': [], 'practice_ids': [key + '-check']})
        question, answer, rubric = PRACTICE[key]
        data['practice'].append({'id': key + '-check', 'question': question, 'answer': answer,
                                 'rubric': rubric, 'objective_ids': [key], 'claim_ids': [key + '-claim']})
    data['sources'] = []
    for lecture in lectures:
        item = VIDEOS[lecture]
        for kind, url in [('video', 'https://www.youtube.com/watch?v=' + item['id']),
                          ('slides', f'https://lecture{lecture}-attention-mechanisms.vercel.app/')]:
            data['sources'].append({'id': f'lecture-{lecture}-{kind}', 'kind': 'external',
                'title': f'Vizuara · The Attention Family · Lecture {lecture} · {kind}', 'url': url,
                'accessed_at': '2026-10-30',
                'locator': f"Video page/description and full English automatic caption text, {SUBTITLES[lecture]['start']}–{SUBTITLES[lecture]['end']}" if kind == 'video' else f'Complete original Slidev presentation, {len(json.loads((SOURCE / f"lecture{lecture}-slides-browser.json").read_text())["slides"])} slides',
                'proof_limit': 'Full automatic caption/transcript text read alongside all slides. Recognition errors are checked against slides and primary references; this is not a frame-by-frame audiovisual audit.' if kind == 'video' else 'Complete rendered presentation reviewed; original deck remains externally hosted and is not republished.'})
    for identifier, title, url, locator in [
        ('bahdanau', 'Bahdanau et al. · Neural Machine Translation by Jointly Learning to Align and Translate', 'https://arxiv.org/abs/1409.0473', 'Alignment model and source-memory motivation'),
        ('luong', 'Luong et al. · Effective Approaches to Attention-based Neural Machine Translation', 'https://aclanthology.org/D15-1166/', 'Global attention and score alternatives'),
        ('transformer', 'Vaswani et al. · Attention Is All You Need', 'https://arxiv.org/html/1706.03762v7', 'Sections 3.2.1–3.2.3 and the scaling assumptions'),
        ('d2l', 'Dive into Deep Learning · Attention Scoring Functions', 'https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html', 'Sections 11.3.2–11.3.4: masked normalization, scaled dot product, additive scoring'),
        ('gqa', 'Ainslie et al. · GQA, version 3', 'https://arxiv.org/abs/2305.13245v3', 'Abstract: MHA, intermediate grouped KV heads and MQA definitions only; model results not reproduced'),
    ]:
        data['sources'].append({'id': identifier, 'kind': 'external', 'title': title, 'url': url,
            'accessed_at': '2026-10-30', 'locator': locator,
            'proof_limit': 'Primary reference or textbook explanation inspected; not an experiment reproduced here.'})
    claim_sources = {'map': ['lecture-1-video', 'lecture-1-slides'],
                     'summary': ['lecture-2-video', 'bahdanau'],
                     'additive': ['bahdanau', 'd2l'], 'scale': ['luong', 'transformer', 'd2l'],
                     'boundary': ['transformer', 'gqa']}
    data['claims'] = [{'id': key + '-claim', 'text': data['objectives'][i]['text'],
                       'source_ids': claim_sources.get(key, ['transformer', 'd2l']), 'status': 'illustrative',
                       'proof_limit': 'Original small worked example and stated mathematical assumptions. Source review and executed author checks do not establish a trained model or learner mastery.'}
                      for i, key in enumerate(objective_keys)]
    data['examples'] = []
    for path in sorted((root / 'examples').glob('*')):
        if path.name in ['row_math.py', 'attention_math.py', 'multihead_torch.py'] or path.suffix not in ['.py', '.cpp']:
            continue
        if path.suffix == '.cpp':
            binary = SOURCE / 'attention-cpp'
            subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', str(path), '-o', str(binary)], check=True)
            command = [str(binary)]
        else:
            command = ['python3', str(path)]
        result = json.loads(subprocess.check_output(command, text=True))
        data['examples'].append({'id': 'example-' + re.sub(r'[^a-z0-9-]', '-', path.stem.lower()).strip('-'),
            'language': 'cpp' if path.suffix == '.cpp' else 'python', 'path': 'examples/' + path.name,
            'purpose': 'Independent illustrative numerical reference, with exact declared input and output.',
            'expected_output': result})
    (root / 'lesson.json').write_text(json.dumps(data, indent=2) + '\n')
    start = root / 'lessons/01-start.md'
    if start.exists() and start.name not in files:
        assert 'tutorial-draft-template' in start.read_text()
        start.unlink()
    source_rows = ['# Original lecture coverage', '', '| Lecture | Included topic | Original video | Original presentation |', '| --- | --- | --- | --- |']
    for lecture in lectures:
        item = VIDEOS[lecture]
        source_rows.append(f'| {lecture} | {item["title"].split(" | ")[0]} | [Video](https://www.youtube.com/watch?v={item["id"]}) | [Slides](https://lecture{lecture}-attention-mechanisms.vercel.app/) |')
    source_rows += ['', 'Lecture numbers come from the video titles; the playlist is displayed in reverse order. Lecture 9 is excluded. One additional unavailable playlist entry was hidden by YouTube; its unknown content is not reconstructed.', '', 'The original presentations stay with Vizuara and are viewed from their original hosts. The tutorial prose, numerical fixtures, code and diagrams are original. All eight English automatic subtitle/transcript texts were read from opening through closing and compared with the complete presentations. DownSub supplied timed VTT files for Lectures 1–8. Lecture 6 was also retrieved through Transcript.you; all 1,417 segment texts match the DownSub copy. Raw captions stay in the ignored workspace source archive.', '', '| Lecture | Caption segments | First–last caption timestamp | Acquisition |', '| --- | --- | --- | --- |']
    for lecture in lectures:
        sub = SUBTITLES[lecture]
        service = 'Transcript.you' if 'transcript.you' in sub['acquisition'] else 'DownSub'
        source_rows.append(f"| {lecture} | {sub['cues']} | {sub['start']}–{sub['end']} | [{service}]({sub['acquisition']}) |")
    source_rows += ['', 'Automatic recognition is fallible. Slide/paper checks resolve technical terms; the transcript review does not establish an error-free transcription or full audiovisual audit. Compact source hashes and format details are in [source-provenance.json](project/source-provenance.json).']
    (root / 'project/source-map.md').write_text('\n'.join(source_rows) + '\n')
    (root / 'project/source-provenance.json').write_text(json.dumps(
        {'accessed_at': '2026-10-30', 'review': 'Full English automatic caption texts and all original slides',
         'raw_captions': 'Ignored workspace source archive; not republished',
         'subtitles': [SUBTITLES[n] for n in lectures]}, indent=2) + '\n')
    (root / 'project/implementation.md').write_text(f'''# Return to this route

This is the `{slug}` tutorial under Attention Labs. The runnable files are illustrative references inside this package; no learner model has been created or trained. Read the lessons in manifest order. The complete source and figures are available through the tutorial downloads.

Source: Vizuara's The Attention Family playlist, lectures {', '.join(map(str, lectures))}, plus the original linked presentations. Read the source map for the exact mapping and evidence limits. Lecture 9 is omitted. All fundamental mechanisms stay together; MHA lives in the separate Attention Heads route.

From this directory run the displayed `python3 examples/...` commands, and the displayed C++ command where applicable. Every helper is included in the package and shown before its first consequential use. The PyTorch packed reference, when present, uses its separately stated dependency and execution record.

Author arithmetic and rendering checks are recorded in reference-runs.md and the owning implementation package. Learner explanation is unassessed. No dataset, fitted translator, benchmark, hosted notebook execution, model training or new learning milestone is implied.
''')
    (root / 'project/readiness.md').write_text('''# What the examples establish

The exit is an explanation you can defend using the numbers: identify input and output axes, derive the weight distribution and context, explain a changed input, and diagnose the relevant invalid or misleading case. The revealable checks in the lessons provide worked answers; they do not record a learner result.

Author checks cover the declared small forward calculations and meaningful interventions. The projection example also compares a derivative with finite differences; the packed reference compares outputs and gradients with an explicit loop. These establish the stated component examples in their tested environment.

A complete application would additionally need data/tokenization, surrounding model layers, a task objective, training and evaluation, persistent state and actual recovery evidence. Those are outside this tutorial. Positional-encoding Lecture 9 remains omitted; GQA/MQA/MLA additions remain future work in the Attention Heads folder.

No learner acceptance, mastery, hosted execution, task accuracy or hardware performance is claimed from authoring and local checks.
''')
    learning = ['# Learning points', '']
    for key in objective_keys:
        learning += [PRACTICE[key][1], '']
    (root / 'project/learning-points.md').write_text('\n'.join(learning))
    results = [{'id': e['id'], 'path': e['path'], 'output': e['expected_output']} for e in data['examples']]
    (root / 'project/reference-runs.md').write_text('# Author numerical references\n\nThese are computed toy examples, not model results or learner answers. Run from this package directory. PyTorch 2.14.1+cpu was already installed in the existing Attention project environment; no dependencies or model data were downloaded for author verification.\n\n```json\n' + json.dumps(results, indent=2) + '\n```\n\nThe separate packed PyTorch program was run on CPU float64 and checked output and query-gradient agreement with an explicit loop, plus invariance of earlier outputs to a changed future token. Detailed independent verification is saved in the owning implementation record.\n')
    (root / 'research.md').write_text('''# Source and teaching review — 2026-10-30

Purpose: a self-contained course for a Python/C++ engineer learning attention mechanisms, with a whole-task overview followed by progressively deeper calculations. Original explanations, code, diagrams and synthetic coordinates are used throughout. The existing Attention Is All You Need tutorial and its accepted four-row fixture are not revised.

The user requested Chrome or Brave. Chrome verified the public playlist, video pages, original slide links and the full Slidev overview pages. The playlist lists Lectures 9 down to 1 plus one unavailable hidden entry. We map by the lecture number in each title, include Lectures 1–8, omit Lecture 9, and separate basic single-head foundations (1–7) from head/KV arrangements (8). No directory is created per basic attention type.

Read all slides in the eight included decks: 21, 20, 28, 17, 27, 25, 21 and 22 slides respectively (181 total). Their progressive motivation and single-operation-per-pass structure inform this route. We retain close video/deck provenance while using a new three-row numerical example, an explicit decoder vocabulary head, a checked projection derivative and a packed-loop gradient comparison.

Also inspected Dive into Deep Learning 1.0.3, Attention Pooling by Similarity (11.2 introduction and pooling equation), Attention Scoring Functions (11.3 masked softmax, dot-product scaling, additive scoring), the Bahdanau alignment paper, Luong's score alternatives, and Attention Is All You Need v7 sections 3.2.1–3.2.3. D2L's explicit width contracts and distinction between score and normalization are useful; our route keeps those contracts beside the worked inputs rather than sending the learner through framework helper code. Primary papers bound the historical and architectural claims.

Departures from source simplifications: the category map is a useful checklist, not an exclusive partition of every method; attention does not always mean softmax; learned RNNs have no universal twenty-word collapse; a conventional decoder predicts vocabulary logits rather than nearest embeddings; scale normalization needs distributional assumptions; concentrated softmax does not imply every training gradient vanishes; raw embeddings can themselves be trainable; shared projections cannot distinguish identical position-free inputs with identical memory; zero-then-renormalize masking is equivalent in exact arithmetic but can fail numerically; dense causal code may still compute forbidden raw scores; dropout preserves a local expectation, not unit row sums in each draw or a whole nonlinear model's evaluation output; MHA allows but does not guarantee head specialization; packed execution has no hardware-independent timing guarantee.

Caption review: after direct timed-text requests and Chrome's YouTube transcript failed, the user supplied DownSub and Transcript.you. Chrome retrieved complete English automatic caption texts through DownSub for all eight lectures. Lecture 6 was initially obtained as a timestamped displayed transcript through Transcript.you, then downloaded as DownSub VTT after its temporary rate limit cleared; all 1,417 segment texts match. All eight texts were read from opening through closing and compared with the full presentations, including the live code demonstrations. Transcript.you requested registration for a later request; no account or paid trial was created. Raw captions and reading copies stay ignored, with compact hashes/provenance in the source map. Automatic captions can misrecognize words; this review is not a frame-by-frame audiovisual audit. Original source links remain available beside each lesson. Original slide applications are embedded from their hosts rather than copied or rehosted.

The spoken review added explicit embedding lookup/training choices, the available-state-before-next-token sequence, rectangular attention and output-head widths, score-difference and random-distribution reasoning, negative/zero-sum normalization counterexamples, dropout's regularization purpose, all-input-features head projections and output-order bookkeeping. GQA version 3 was inspected only for future head-sharing vocabulary; no quality or speed result is reproduced.

The two routes preserve all requested topics. Positional Lecture 9 and unavailable video content are outside scope. Future GQA/MQA/MLA work is identified but not implemented. Source/math/render checks, full-model behavior, hosted runs and learner understanding are separate evidence categories.
''')
    return data


if __name__ == '__main__':
    images()
    foundation = package('attention-foundations', [
        '01-context-and-map.md', '02-fixed-summary.md', '03-additive-retrieval.md',
        '04-dot-products-and-scale.md', '05-one-sequence.md', '06-learned-projections.md',
        '07-causality-and-dropout.md'], ['map', 'summary', 'additive', 'scale', 'self', 'projection', 'causal'], range(1, 8))
    heads = package('attention-heads', ['01-independent-heads.md', '02-packed-layout.md', '03-head-layout-boundaries.md'],
                    ['heads', 'layout', 'boundary'], [8])
    print(json.dumps({'chapters': len(foundation['chapters']) + len(heads['chapters']),
                      'declared_examples': len(foundation['examples']) + len(heads['examples']),
                      'original_decks': 8, 'lecture9': 'excluded'}))
