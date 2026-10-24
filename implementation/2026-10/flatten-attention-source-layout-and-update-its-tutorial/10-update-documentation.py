"""Reconcile the current project/course with the grouped source layout."""
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PROJECT=ROOT/'ai_projects/projects/transformer-labs/attention-is-all-you-need'
COURSE=ROOT/'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
ASSEMBLER=ROOT/'ai_projects/docs/implementation/2026-10/attention-is-all-you-need-source-backed-tutorial/08-assemble.py'
mapping={
 'src/attention_is_all_you_need/attention.py':'src/core/attention.py',
 'src/attention_is_all_you_need/model.py':'src/core/model.py',
 'src/attention_is_all_you_need/config.py':'src/core/config.py',
 'src/attention_is_all_you_need/data.py':'src/data/dataset.py',
 'src/attention_is_all_you_need/tokenizer.py':'src/data/tokenizer.py',
 'src/attention_is_all_you_need/trainer.py':'src/training/trainer.py',
 'src/attention_is_all_you_need/checkpoint.py':'src/training/checkpoint.py',
 'src/attention_is_all_you_need/':'src/',
 'from attention_is_all_you_need.attention import':'from core.attention import',
 'from attention import Attention':'from core.attention import Attention',
 'code/attention.py':'code/src/cli/attention.py',
 'code/attention_visuals.py':'code/src/utils/attention_visuals.py',
 'jupyter lab attention_mechanism.ipynb':'jupyter lab notebooks/attention_mechanism.ipynb',
 'python attention.py':'python src/cli/attention.py',
 '`attention_visuals.py`':'`src/utils/attention_visuals.py`',
 '[attention_visuals.py](attention_visuals.py)':'[attention_visuals.py](src/utils/attention_visuals.py)',
 '[attention_visuals.py](../attention_visuals.py)':'[attention_visuals.py](../src/utils/attention_visuals.py)',
 '`attention_mechanism.ipynb`':'`notebooks/attention_mechanism.ipynb`',
 '[attention_mechanism.ipynb](attention_mechanism.ipynb)':'[attention_mechanism.ipynb](notebooks/attention_mechanism.ipynb)',
}
targets=[PROJECT/'README.md',PROJECT/'docs/SOURCES.md',PROJECT/'docs/COLAB.md',ASSEMBLER]
targets+=list((COURSE/'lessons').glob('*.md'))+list((COURSE/'project').rglob('*.md'))+[COURSE/'research.md',COURSE/'examples/verify_component.py']
for path in targets:
 text=path.read_text()
 for before,after in mapping.items():text=text.replace(before,after)
 text=text.replace('no new model implementation','no new translation-model implementation')
 path.write_text(text)

# Current course code snippets use core/cli/data/training imports and paths.
for path in (COURSE/'lessons').glob('*.md'):
 text=path.read_text()
 for name in ['tokenizer','prepare_data','train','inference','compute_bleu']:
  text=text.replace('`'+name+'.py`','`src/cli/'+name+'.py`')
 text=text.replace('`attention.py` | Exposes','`src/cli/attention.py` | Exposes')
 text=text.replace('The package\'s `model.py`, `data.py`, `tokenizer.py`, `trainer.py`, `checkpoint.py` and `config.py` are also placeholders.','The modules `src/core/model.py`, `src/core/config.py`, `src/data/dataset.py`, `src/data/tokenizer.py`, `src/training/trainer.py` and `src/training/checkpoint.py` are also placeholders.')
 text=text.replace('The top-level','The command').replace('later top-level','later command')
 text=text.replace('Its Python package lives under `src/`.','Its implementation is grouped directly below `src/`: `core`, `data`, `training`, `cli` and `utils`. The former outer project-named namespace has been removed; installation exposes these responsibility packages.')
 text=text.replace('The notebook contains 37 code cells and 28 saved PNG figures. Its saved executed state was inspected with no saved errors. These counts describe the file; they do not mean its cells were all freshly reexecuted for this documentation change.','The notebook is stored outside source code in `notebooks/attention_mechanism.ipynb`. All 37 code cells were freshly executed from both the project root and the notebooks directory after the layout change, producing 28 PNG figures with no cell errors.')
 text=text.replace('The code is unchanged by this documentation work.','The code has been reorganized by responsibility; the attention module\'s numerical implementation and demo outputs are unchanged.')
 text=text.replace('The notebook\'s saved outputs remain separate from fresh execution evidence.','Fresh full notebook execution is recorded separately from the attention tests and demo.')
 path.write_text(text)

# The assembler must capture the new source layout and retain uncommitted provenance.
text=ASSEMBLER.read_text()
text=re.sub(r"names=\[.*?\]\nnames \+=", "names=['README.md','LICENSE','pyproject.toml','uv.lock','.python-version','.ai-project.toml']\nnames +=",text,count=1,flags=re.S)
text=text.replace("folder in ('src','tests','configs')","folder in ('src','tests','configs','notebooks')")
text=text.replace("PROJECT/'attention_mechanism.ipynb'","PROJECT/'notebooks/attention_mechanism.ipynb'")
text=text.replace("('visual-source','attention_visuals.py'","('visual-source','src/utils/attention_visuals.py'")
text=text.replace("'working_tree':False","'working_tree':bool(git(OWNER,'status','--porcelain','--',str(file.relative_to(OWNER))))")
text=text.replace('The source notebook\'s 37 saved code cells and 28 figures were inspected; a complete fresh notebook execution is not claimed by this pass.','The notebook now lives in notebooks/, outside the responsibility packages core/data/training/cli/utils under src/. Its 37 code cells have been freshly executed from both root and notebooks working directories, producing 28 figures without cell errors.')
text=text.replace('The original notebook remains unchanged; selected existing PNGs are copied verbatim with provenance in the authoring record.','The notebook paths/imports were updated and all cells reexecuted after the source reorganization. Attention arithmetic and fixed-seed demo outputs remain unchanged; selected figures are copied with source provenance.')
text=text.replace('attention_visuals.py supplies','src/utils/attention_visuals.py supplies')
text=text.replace('The notebook has 37 saved executed cells and 28 figures; a full new notebook/hosted execution is not claimed.','The notebook lives in notebooks/ and all 37 code cells execute from both project-root and notebooks working directories, producing 28 figures. Hosted execution remains unverified.')
ASSEMBLER.write_text(text)

# Explain organization where setup first introduces its files.
path=COURSE/'lessons/02-project-setup.md'
text=path.read_text()
section='''### Keep code grouped by responsibility

The original project-named directory under src was a conventional namespace for installation. This application now uses responsibility packages directly below src, as requested. `core` owns attention/model/configuration; `data` owns tokenizers and dataset handling; `training` owns fitting/recovery; `cli` owns runnable commands; `utils` owns graph/display helpers. Each group is importable after `uv sync`; notebooks import that same source instead of defining a second implementation.

```text
src/
  core/       attention.py, model.py, config.py
  data/       tokenizer.py, dataset.py
  training/   trainer.py, checkpoint.py
  cli/        attention.py, tokenizer.py, prepare_data.py, train.py,
              inference.py, compute_bleu.py
  utils/      attention_visuals.py
notebooks/
  attention_mechanism.ipynb
```

Hatch's source mapping installs these groups without an extra outer project namespace. The console command `uv run --frozen attention-demo` and the file command `uv run --frozen python src/cli/attention.py` call the same demo. Plot helpers keep generated images in the project's ignored `runs/attention-visuals/` directory when launched from either supported notebook location; `configure(output_dir=...)` selects an explicit destination for an installed standalone copy.

'''
if '### Keep code grouped by responsibility' not in text:text=text.replace('### Know which file owns which responsibility',section+'### Know which file owns which responsibility')
text=text.replace('is tracked by the `ai_projects` repository. Do not initialize another Git repository inside it or recreate its scaffold.','is tracked by the `ai_projects` repository. Continue the existing project.')
path.write_text(text)

# Reconcile current review records; old dated receipts remain historical evidence.
for path in [COURSE/'project/implementation.md',COURSE/'project/readiness.md',COURSE/'project/reference-runs.md']:
 text=path.read_text()
 text=text.replace('37 saved executed code cells','37 freshly executed code cells')
 text=text.replace('Saved outputs inspected; no new hosted execution','Fresh root/notebooks executions; no hosted execution')
 text=text.replace('Model code, the original notebook, captions and Bike source/results are preserved by this authoring work.','Attention mathematics, captions and Bike source/results are preserved. Source modules are grouped by responsibility, and the notebook is relocated and freshly executed.')
 text=text.replace('Those saved results were inspected. The 35 existing tests and module/demo probes are fresh checks; a new execution of every notebook cell or a live hosted widget session is not claimed.','All notebook code cells were freshly executed from the project root and notebooks directory after relocation. The two complete runs produce 28 PNG figures each with no cell errors. Live hosted widgets remain untested.')
 path.write_text(text)

print('Current project/course paths and grouped-layout explanations updated.')
