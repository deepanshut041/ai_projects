from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from ai_cli.manager import Manager, identity, owned, read_json
from ai_cli.registry import Step, UserError


def test_persistent_service_restart_port_and_logs(manager):
    record = manager.launch("project-docs", "start")
    manager.wait_ready("project-docs", 5)
    # A completely new controller attaches to the already-owned supervisor.
    other = Manager(manager.root)
    assert other.record("project-docs")["pid"] == record["pid"]
    assert other.ready(other.record("project-docs"))
    with urlopen(record["url"]) as response:
        assert response.read() == b"ai fixture"
    with pytest.raises(UserError, match="already"):
        other.launch("project-docs", "start")
    with pytest.raises(UserError, match="Stop"):
        other.set_port("project-docs", 46099)
    assert other.stop("project-docs")
    assert not owned(record, manager.root)
    manager.reap()
    next_record = manager.launch("project-docs", "start")
    manager.wait_ready("project-docs", 5)
    assert next_record["token"] != record["token"]
    assert "$ " in manager.logs("project-docs")


def test_external_listener_conflict_never_stops_listener(manager):
    with socket.socket() as external:
        external.bind(("127.0.0.1", 0))
        external.listen()
        port = external.getsockname()[1]
        with pytest.raises(UserError, match="in use"):
            manager.set_port("project-docs", port)
        # Simulate a conflict that appears after configuration was saved.
        with manager.locked():
            from ai_cli.manager import atomic_json
            atomic_json(manager.runtime / "ports.json", {"project-docs": port})
        with pytest.raises(UserError, match="nothing was stopped"):
            manager.launch("project-docs", "start")
        assert external.getsockname()[1] == port
        assert not manager.stop("project-docs")


def test_stale_pid_and_spoofed_record_cannot_signal_foreign_process(manager):
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        with manager.locked():
            state = manager.state()
            state["runs"]["project-docs:service"] = {
                "key": "project-docs:service", "project": "project-docs", "pid": foreign.pid,
                "identity": identity(foreign.pid), "token": "spoofed", "status": "running"}
            manager.save(state)
        assert not manager.stop("project-docs")
        assert foreign.poll() is None
        assert manager.snapshot()[0]["runs"]["service"]["status"] == "stale"
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_concurrent_start_only_creates_one_supervisor(manager):
    def launch():
        try:
            return Manager(manager.root).launch("project-docs", "start")
        except UserError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda _: launch(), range(2)))
    assert sum(item is not None for item in result) == 1
    manager.wait_ready("project-docs", 5)


def test_job_exit_code_and_failure_are_retained(manager):
    manager.launch("project-docs", "build")
    result = manager.wait_job("project-docs", 5)
    assert result["status"] == "finished"
    assert result["returncode"] == 0
    assert "job completed" in manager.logs("project-docs", "job")
    manager.launch("project-docs", "test")
    result = manager.wait_job("project-docs", 5)
    assert result["status"] == "failed"
    assert result["returncode"] == 7


def test_job_overlap_refused_and_missing_executable_visible(manager):
    step = Step((sys.executable, "-c", "import time; time.sleep(30)"), str(manager.root))
    manager.launch("project-docs", "build", steps=[step])
    with pytest.raises(UserError, match="already"):
        manager.launch("project-docs", "test")
    with pytest.raises(UserError, match="active"):
        manager.launch("project-docs", "start")
    manager.stop("project-docs", "job")
    manager.launch("project-docs", "build", steps=[Step(("/missing-quantum-executable",), str(manager.root))])
    result = manager.wait_job("project-docs", 5)
    assert result["returncode"] == 127
    assert "FileNotFoundError" in manager.logs("project-docs", "job")


def test_cancellation_kills_descendants_that_ignore_term(manager):
    child_pid = manager.root / "child.pid"
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent_code = f"import subprocess,time,pathlib; p=subprocess.Popen([{sys.executable!r},'-c',{child_code!r}]); pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)"
    manager.launch("project-docs", "build", steps=[Step((sys.executable, "-c", parent_code), str(manager.root))])
    deadline = time.monotonic() + 5
    while not child_pid.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert child_pid.exists()
    descendant = int(child_pid.read_text())
    time.sleep(.2)  # Let the child install its signal handler.
    assert manager.stop("project-docs", "job", timeout=.4)
    deadline = time.monotonic() + 3
    while identity(descendant) and time.monotonic() < deadline:
        time.sleep(.05)
    assert identity(descendant) is None


def test_argv_literal_no_shell_and_bounded_log_tail(manager):
    literal = "$(touch sentinel) `touch sentinel2` [red]output[/red]"
    step = Step((sys.executable, "-c", "import sys; print(sys.argv[1]); print('x'*50000)", literal), str(manager.root))
    manager.launch("project-docs", "build", steps=[step])
    manager.wait_job("project-docs", 5)
    assert not (manager.root / "sentinel").exists()
    assert not (manager.root / "sentinel2").exists()
    assert len(manager.logs("project-docs", "job", limit=1000)) <= 1000


def test_corrupt_state_fails_without_overwriting(manager):
    state = manager.runtime / "state.json"
    state.write_text("broken state")
    with pytest.raises(UserError, match="repair"):
        manager.snapshot()
    assert state.read_text() == "broken state"
    state.unlink()  # Fixture teardown can now inspect its empty state.


@pytest.mark.parametrize("filename,payload", [("state.json", '{"runs": []}'), ("ports.json", '{"project-docs": "wrong"}')])
def test_invalid_runtime_shapes_fail_without_overwriting(manager, filename, payload):
    path = manager.runtime / filename
    path.write_text(payload)
    with pytest.raises(UserError, match="Invalid"):
        manager.snapshot()
    assert path.read_text() == payload
    path.unlink()


def test_stop_all_also_stops_owned_runs_removed_from_registry(manager):
    record = manager.launch("project-docs", "start")
    manager.wait_ready("project-docs", 5)
    manager.registry.projects.clear()
    manager.stop_all()
    assert not owned(record, manager.root)
    assert manager.record("project-docs")["status"] == "stopped"
