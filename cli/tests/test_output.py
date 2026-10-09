from pathlib import Path
import os
import subprocess
import sys

from ai_cli.runner import RunLog


def test_output_rotation_keeps_recent_stream_and_bounded_archive(tmp_path):
    path = tmp_path / "run.log"
    log = RunLog(path, limit=1000)
    try:
        log.write(b"a" * 600)
        log.write(b"b" * 600)
        log.write(b"c" * 600)
    finally:
        log.stream.close()
    assert path.read_bytes() == b"c" * 600
    assert path.with_suffix(".log.1").read_bytes() == b"b" * 600


def test_pruning_keeps_recent_owned_logs_and_unrelated_files(manager):
    folder = manager.runtime / "logs"
    for index in range(5):
        path = folder / f"project-docs-job-{index:032x}.log"
        path.write_text(str(index))
        os.utime(path, (index, index))
    unrelated = folder / "project-docs-job-my-important-file.log"
    unrelated.write_text("keep me")
    manager.prune_logs("project-docs", "job")
    assert unrelated.read_text() == "keep me"
    assert len(list(folder.glob("project-docs-job-*.log"))) == 3


def test_no_registered_or_built_tests_is_failure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.21)\nproject(no_tests NONE)\ninclude(CTest)\n")
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True, capture_output=True)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "ai_cli.native_tests", str(build)], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "cannot pass" in result.stdout
