import json
import os
from pathlib import Path
import subprocess
import sys

from ai_cli.__main__ import main


def test_headless_status_and_doctor(workspace, capsys):
    assert main(["--root", str(workspace), "status", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == "project-docs"
    assert main(["--root", str(workspace), "doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["platform"] == "linux"


def test_cli_exit_codes_and_persistence_across_subprocesses(manager):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    def invoke(*argv):
        return subprocess.run([sys.executable, "-m", "ai_cli", "--root", str(manager.root), *argv],
                              env=env, capture_output=True, text=True, timeout=10)
    start = invoke("start", "project-docs", "--json")
    assert start.returncode == 0, start.stderr
    assert manager.ready(manager.record("project-docs"))
    assert invoke("stop", "project-docs").returncode == 0
    assert invoke("build", "project-docs").returncode == 0
    failure = invoke("test", "project-docs", "--json")
    assert failure.returncode == 1
    assert json.loads(failure.stdout)["returncode"] == 7


def test_invalid_port_and_unknown_project_give_actionable_error(workspace, capsys):
    assert main(["--root", str(workspace), "port", "project-docs", "0"]) == 1
    assert "1024" in capsys.readouterr().err
    assert main(["--root", str(workspace), "start", "unknown"]) == 1
    assert "ai list" in capsys.readouterr().err
