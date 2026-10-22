"""Study/export checks: topic coverage, personal-note preservation and execution."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from course_study_notebook import notebook, notes_text, write_notebook
from course_source_export import export_sources


def cell_source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def assert_cleared_compilable(test, document):
    test.assertEqual(document["nbformat"], 4)
    ids = [cell["id"] for cell in document["cells"]]
    test.assertEqual(len(ids), len(set(ids)))
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            test.assertIsNone(cell["execution_count"])
            test.assertEqual(cell["outputs"], [])
            compile(cell_source(cell), cell["id"], "exec")


def execute_study(extracted, *, interpreter=sys.executable):
    """One fresh interpreter/kernel, sequential cells, with neighboring sources."""
    script = """import contextlib, io, json, traceback
from pathlib import Path
document=json.loads(Path('study.ipynb').read_text())
namespace={'__name__':'__main__'}
records=[]
for cell in document['cells']:
    if cell['cell_type']!='code': continue
    source=cell['source']
    source=''.join(source) if isinstance(source,list) else source
    stream=io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            exec(compile(source,cell['id'],'exec'),namespace)
    except Exception:
        print(json.dumps({'cell':cell['id'],'output':stream.getvalue(),'error':traceback.format_exc()}))
        raise
    records.append({'id':cell['id'],'stdout':stream.getvalue()})
print(json.dumps({'code_cells':len(records),'cells':records}))
"""
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([interpreter, "-c", script], cwd=extracted,
                            text=True, capture_output=True, timeout=60, env=environment)
    return result


class CourseStudyChecks(unittest.TestCase):
    def topic_titles(self):
        # In source exports, the workbook is the frozen course-topic manifest;
        # the full authoring manifest lives beside the canonical examples only.
        source = Path(__file__).resolve().parent
        manifest = source.parent / "lesson.json"
        if manifest.is_file():
            return [chapter["title"] for chapter in json.loads(manifest.read_text())["chapters"]]
        return [line[3:] for line in notes_text().splitlines() if line.startswith("## Lesson ")]

    def test_every_manifest_topic_has_notes_and_notebook_space(self):
        text = notes_text()
        document = notebook()
        markdown = "\n".join(cell_source(cell) for cell in document["cells"] if cell["cell_type"] == "markdown")
        titles = self.topic_titles()
        self.assertGreater(len(titles), 0)
        for title in titles:
            self.assertEqual(text.count("## " + title + "\n"), 1, title)
            self.assertEqual(markdown.count("## " + title + "\n"), 1, title)
        self.assertEqual(sum(line.startswith("## Lesson ") for line in text.splitlines()), len(titles))
        self.assertNotIn("[[practice:", markdown)
        self.assertNotIn("[[diagram:", markdown)
        self.assertNotIn("[[scenario:", markdown)

    def test_study_cells_are_cleared_unique_and_compilable(self):
        assert_cleared_compilable(self, notebook())

    def test_preserve_existing_personal_notebook(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "my-study.ipynb"
            personal = b'{"my_notes":"I derived this myself; keep it."}\n'
            destination.write_bytes(personal)
            with self.assertRaises(FileExistsError):
                write_notebook(destination)
            self.assertEqual(destination.read_bytes(), personal)

    def test_new_written_notebook_matches_cleared_document(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "new-study.ipynb"
            self.assertEqual(write_notebook(destination), destination)
            actual = json.loads(destination.read_text())
            self.assertEqual(actual, notebook())
            assert_cleared_compilable(self, actual)

    def test_preserve_existing_archive_or_neighboring_colab_notebook(self):
        for existing in ("archive", "colab"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as folder:
                destination = Path(folder) / "source.zip"
                chosen = destination if existing == "archive" else destination.with_suffix(".colab.ipynb")
                personal = b"previous personal export or notebook; preserve exact bytes"
                chosen.write_bytes(personal)
                with self.assertRaises(FileExistsError):
                    export_sources(destination)
                self.assertEqual(chosen.read_bytes(), personal)
                counterpart = destination.with_suffix(".colab.ipynb") if existing == "archive" else destination
                self.assertFalse(counterpart.exists())

    def test_archive_members_hashes_source_and_study_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "source.zip"
            result = export_sources(destination)
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), result["sha256"])
            colab_path = Path(result["notebook"])
            colab = json.loads(colab_path.read_text())
            assert_cleared_compilable(self, colab)
            self.assertIn(result["sha256"], "\n".join(cell_source(cell) for cell in colab["cells"]))
            with zipfile.ZipFile(destination) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(Path(name).name == name for name in names))
                manifest = json.loads(archive.read("source-manifest.json"))
                self.assertEqual(set(names), set(manifest["files"]) | {"source-manifest.json"})
                self.assertEqual(result["files"], len(manifest["files"]))
                for name, expected in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected, name)
                self.assertEqual(archive.read("study-notes.md").decode(), notes_text())
                self.assertEqual(json.loads(archive.read("study.ipynb")), notebook())
                assert_cleared_compilable(self, json.loads(archive.read("study.ipynb")))
                for name in ("model_learning_probes.py", "test_model_learning_probes.py", "course_study_notebook.py", "test_course_study.py"):
                    self.assertEqual(archive.read(name), Path(__file__).with_name(name).read_bytes())

    def test_fresh_extracted_sources_execute_all_optional_study_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "source.zip"
            export_sources(destination)
            extracted = Path(folder) / "extracted"
            extracted.mkdir()
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(extracted)  # This archive was just generated and checked as flat.
            result = execute_study(extracted)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = json.loads(result.stdout)
            code_cells = [cell for cell in notebook()["cells"] if cell["cell_type"] == "code"]
            self.assertEqual(records["code_cells"], len(code_cells))
            self.assertEqual([cell["id"] for cell in records["cells"]], [cell["id"] for cell in code_cells])
            # The delivered applied probes also run as a direct standalone program.
            direct = subprocess.run([sys.executable, "model_learning_probes.py"], cwd=extracted,
                                    capture_output=True, text=True, timeout=60,
                                    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertTrue(json.loads(direct.stdout)["visibility"]["translation_future_prefix_invariant"])


if __name__ == "__main__":
    unittest.main()
