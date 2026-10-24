# Layout contracts and execution

See 07-layout-contract.md and path-mapping.json for the reviewed layout and exact original-to-current mapping. Recovery archive bytes were verified before moving files. Reusable attention remains byte-identical at src/core/attention.py; the command becomes src/cli/attention.py and imports core.attention. Each source group is a regular package with __init__.py. Hatch only-includes src and removes its source prefix when building the wheel. The console entry is attention-demo -> cli.attention:main.

Graph helpers live at src/utils/attention_visuals.py. Their output discovery finds the source project or a project ancestor, with current-directory fallback in installed standalone use; configure(output_dir=...) overrides that destination. The notebook imports core, utils and cli modules and is executed from both project-root and notebooks directories.

Current README, source/Colab guidance, tutorial lessons/review records/probes, source-capture generator and local derived notes use the grouped paths. The capture records dirty working-tree provenance, synchronizes obsolete copies, retains source hashes and includes 32 files in its safe restorable JSON bundle. The separate documented Colab ZIP includes those files plus three public source documents. Parent runtime data ignores are explicitly overridden for src/data source.

Verification scripts cover pre/post demo hashes, build contents, installed imports with project source removed from search paths, existing tests, root/notebook execution, source restoration and tutorial snippets. Reader validation/build/browser checks cover both hosting bases and 1440/768/390/320 layouts, source downloads, notebook links, practice and navigation. Restart the managed reader after final source publication.
