"""Move runnable entry files directly below src and reconcile current consumers."""
import json
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
ASSEMBLER=ROOT/'ai_projects/docs/implementation/2026-10/attention-is-all-you-need-source-backed-tutorial/08-assemble.py'
recovery=ROOT/'.ai/attention-layout/2026-10-23'
for name in ('source-bundle.json','source-manifest.json'):
    target=recovery/('before-direct-entrypoints-'+name)
    if not target.exists():shutil.copyfile(COURSE/'code'/name,target)
cli=PROJECT/'src/cli'
names=['attention','tokenizer','prepare_data','train','inference','compute_bleu']
for name in names:
    target=PROJECT/'src'/(name+'.py')
    if target.exists():raise RuntimeError('Target already exists: '+str(target))
    (cli/(name+'.py')).rename(target)
(cli/'__init__.py').unlink()
if (cli/'__pycache__').exists():shutil.rmtree(cli/'__pycache__')
cli.rmdir()
for name in ('.ai-project.toml','pyproject.toml'):
    path=PROJECT/name
    path.write_text(path.read_text().replace('src/cli/','src/').replace('cli.attention:main','attention:main'))
path=PROJECT/'notebooks/attention_mechanism.ipynb'
notebook=json.loads(path.read_text())
for cell in notebook['cells']:
    cell['source']=[line.replace('from cli.attention import','from attention import') for line in cell['source']]
path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
paths=[PROJECT/'README.md',PROJECT/'notebooks/README.md',PROJECT/'docs/COLAB.md',PROJECT/'docs/SOURCES.md',ASSEMBLER]
paths+=list((COURSE/'lessons').glob('*.md'))+list((COURSE/'project').rglob('*.md'))+[COURSE/'research.md']
for path in paths:
    text=path.read_text().replace('src/cli/','src/').replace('`cli.attention`','`attention`')
    text=text.replace('src/core, src/data, src/training, src/cli and src/utils','src/core, src/data, src/training and src/utils, with runnable scripts directly under src')
    text=text.replace('core/data/training/cli/utils','core/data/training/utils, with entry scripts directly under src')
    text=text.replace('`core`, `data`, `training`, `cli` and `utils`','`core`, `data`, `training` and `utils`, alongside direct runnable scripts')
    text=text.replace('`core` owns attention/model/configuration; `data` owns tokenizers and dataset handling; `training` owns fitting/recovery; `cli` owns runnable commands; `utils` owns graph/display helpers.','`core` owns attention/model/configuration; `data` owns tokenizers and dataset handling; `training` owns fitting/recovery; `utils` owns graph/display helpers. Runnable commands live directly under `src/`, with no `cli/` folder.')
    path.write_text(text)
tree='''src/
  attention.py              Runnable attention demo
  tokenizer.py              Future tokenizer preparation command
  prepare_data.py           Future dataset preparation command
  train.py                  Future training command
  inference.py              Future translation command
  compute_bleu.py           Future BLEU command
  core/                     attention.py, model.py, config.py
  data/                     tokenizer.py, dataset.py
  training/                 trainer.py, checkpoint.py
  utils/                    attention_visuals.py
notebooks/
  attention_mechanism.ipynb
'''
for path in (PROJECT/'README.md',COURSE/'lessons/02-project-setup.md'):
    text=path.read_text()
    text=re.sub(r'```text\nsrc/\n.*?```','```text\n'+tree+'```',text,count=1,flags=re.S)
    text=text.replace('The package installer exposes `core`, `data`, `training` and `utils`, alongside direct runnable scripts;','The package installer exposes `core`, `data`, `training` and `utils`, plus the runnable modules directly under `src/`;')
    path.write_text(text)
# Existing layout verification now checks the user's final entry-point placement.
path=HERE/'11-verify-layout.py';text=path.read_text()
text=text.replace("assert not list((PROJECT/'src').glob('*.py'))","assert {p.name for p in (PROJECT/'src').glob('*.py')} == {'attention.py','tokenizer.py','prepare_data.py','train.py','inference.py','compute_bleu.py'}\nassert not (PROJECT/'src/cli').exists()")
text=text.replace("('core','data','training','cli','utils')","('core','data','training','utils')")
text=text.replace("['core','data','training','cli','utils']","['core','data','training','utils']")
text=text.replace('from cli.attention import run_demo','from attention import run_demo')
text=text.replace('cli.attention','attention').replace("'cli/attention.py'","'attention.py'")
text=text.replace("('src/','attention_is_all_you_need/')","('src/','cli/','attention_is_all_you_need/')")
text=text.replace("['-q','-c','/dev/null',str(source/'tests')]","['-q','-p','no:cacheprovider','-c','/dev/null',str(source/'tests')]")
path.write_text(text)
path=HERE/'path-mapping.json'
mapping=json.loads(path.read_text());mapping={k:v.replace('src/cli/','src/') for k,v in mapping.items()}
path.write_text(json.dumps(mapping,indent=2)+'\n')
print(json.dumps({'entry_scripts':'src/','removed':'src/cli/','moved':len(names)}))
