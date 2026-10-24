"""Apply the user-requested responsibility-based project layout once."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
mapping={
 'attention.py':'src/cli/attention.py',
 'attention_visuals.py':'src/utils/attention_visuals.py',
 'attention_mechanism.ipynb':'notebooks/attention_mechanism.ipynb',
 'tokenizer.py':'src/cli/tokenizer.py',
 'prepare_data.py':'src/cli/prepare_data.py',
 'train.py':'src/cli/train.py',
 'inference.py':'src/cli/inference.py',
 'compute_bleu.py':'src/cli/compute_bleu.py',
 'src/attention_is_all_you_need/attention.py':'src/core/attention.py',
 'src/attention_is_all_you_need/model.py':'src/core/model.py',
 'src/attention_is_all_you_need/config.py':'src/core/config.py',
 'src/attention_is_all_you_need/data.py':'src/data/dataset.py',
 'src/attention_is_all_you_need/tokenizer.py':'src/data/tokenizer.py',
 'src/attention_is_all_you_need/trainer.py':'src/training/trainer.py',
 'src/attention_is_all_you_need/checkpoint.py':'src/training/checkpoint.py',
}
for source,target in mapping.items():
 old,new=PROJECT/source,PROJECT/target
 if not old.is_file() or new.exists():raise RuntimeError('Expected unmigrated file: '+source)
 new.parent.mkdir(parents=True,exist_ok=True);old.rename(new)
legacy=PROJECT/'src/attention_is_all_you_need'
(legacy/'__init__.py').unlink()
if (legacy/'__pycache__').exists():shutil.rmtree(legacy/'__pycache__')
legacy.rmdir()
for folder,description in {'core':'Attention, model architecture and shared configuration.','data':'Translation tokenization and dataset preparation.','training':'Training and full-state checkpoint recovery.','cli':'Runnable project commands.','utils':'Graph, matrix-display and notebook helpers.'}.items():
 (PROJECT/'src'/folder/'__init__.py').write_text('"""'+description+'"""\n')
for relative in ('src/cli/attention.py','src/utils/attention_visuals.py','tests/test_attention.py'):
 p=PROJECT/relative;p.write_text(p.read_text().replace('from attention_is_all_you_need.attention import','from core.attention import'))
p=PROJECT/'tests/test_attention_visuals.py';p.write_text(p.read_text().replace('from attention_visuals import','from utils.attention_visuals import'))
p=PROJECT/'src/utils/attention_visuals.py'
text=p.read_text().replace('QA = Path(__file__).resolve().parent / "runs" / "attention-visuals"','''def _default_output_directory():
    """Keep plots under project runs from source, root or notebook launches."""
    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/core/attention.py").is_file():
            return candidate / "runs" / "attention-visuals"
    return Path.cwd() / "runs" / "attention-visuals"


QA = _default_output_directory()''')
text=text.replace('def configure(*, save_figures=True):','def configure(*, save_figures=True, output_dir=None):').replace('global _save_figures\n    _save_figures = save_figures','global _save_figures, QA\n    _save_figures = save_figures\n    if output_dir is not None:\n        QA = Path(output_dir).expanduser().resolve()')
p.write_text(text)
p=PROJECT/'notebooks/attention_mechanism.ipynb';nb=json.loads(p.read_text())
for cell in nb['cells']:
 text=''.join(cell['source'])
 if cell['cell_type']=='code' and 'root = Path.cwd()' in text:
  text=text[text.index('import torch'):]
 text=text.replace('from attention import run_demo','from cli.attention import run_demo')
 text=text.replace('from attention import (','from core.attention import (')
 text=text.replace('import attention_visuals as vis','from utils import attention_visuals as vis')
 text=text.replace('src/attention_is_all_you_need/attention.py','src/core/attention.py')
 text=text.replace('`attention_visuals.py`','`src/utils/attention_visuals.py`')
 text=text.replace('importable from `attention.py`','importable with `from core.attention import Attention`')
 text=text.replace('jupyter lab attention_mechanism.ipynb','jupyter lab notebooks/attention_mechanism.ipynb')
 text=text.replace('Sources are in `docs/SOURCES.md`.','Sources are in the project’s `docs/SOURCES.md`. The notebook lives in `notebooks/`; implementation and drawing helpers live in grouped `src/` folders.')
 cell['source']=text.splitlines(keepends=True)
nb['metadata']['attention_visuals']['implementation']='src/core/attention.py'
nb['metadata']['attention_visuals']['visualization_module']='src/utils/attention_visuals.py'
p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
p=PROJECT/'.gitignore';p.write_text(p.read_text().replace('!attention_mechanism.ipynb','!/notebooks/attention_mechanism.ipynb'))
p=PROJECT/'.ai-project.toml';text=p.read_text()
for source,target in mapping.items():
 if '/' not in source:text=text.replace('path = "'+source+'"','path = "'+target+'"')
p.write_text(text)
p=PROJECT/'pyproject.toml';text=p.read_text()
text=text.replace('[dependency-groups]','[project.scripts]\nattention-demo = "cli.attention:main"\n\n[dependency-groups]')
text=text.replace('packages = ["src/attention_is_all_you_need"]','only-include = ["src"]\nsources = ["src"]\n\n[tool.hatch.build.targets.sdist]\ninclude = ["src", "notebooks", "tests", "configs", "README.md", "LICENSE", "pyproject.toml", "uv.lock", ".python-version", ".ai-project.toml", "docs/SOURCES.md", "docs/COLAB.md", "docs/sources.json"]')
text=text.replace('pythonpath = ["."]','pythonpath = ["src"]')
p.write_text(text)
(HERE/'path-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
print(json.dumps({'moved':len(mapping),'groups':['core','data','training','cli','utils'],'notebooks':'notebooks/attention_mechanism.ipynb'}))
