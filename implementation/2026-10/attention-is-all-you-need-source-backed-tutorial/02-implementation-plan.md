# Artifact and reader implementation

Canonical course: docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1, stable named application ID. lesson.json orders twelve lessons and declares their objective/retrieval checks. Existing source/figures and original MIT attribution are preserved. Shared index integration does not change numbered courses.

Project-local docs keep phases separate from implementation/00_project-setup and 01_attention-mechanism. Each implementation record has resume, learning points, worked notes, tasks, comparisons and tracker. The continuous worked notes are derived from the canonical lesson text by 08-assemble.py, with rebased links and locally copied ignored assets so the project reader can safely resolve them. Parts 2-3 are overview-only planned phases.

The source JSON bundle records exact text and hashes for 25 files, including lock, notebook, real code and honest placeholders. restore_source.py verifies paths/hashes and refuses existing destinations. verify_component.py exercises the frozen real module; arithmetic.py supplies independent list/scalar answers. The source verification script restores a disposable project and runs its 35 existing tests plus complete lesson snippets against restored imports.

Use learn examples before render, then native tests/build/browser checks. Focused root/prefix builds use separate test-results directories; check every new lesson/project link, images, equations, reveal state, navigation, downloads, print, and 1440/768/390/320 containment. Preserve private-export separation. Restart managed preview after final verification.
