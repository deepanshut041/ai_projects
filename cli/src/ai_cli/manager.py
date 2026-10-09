"""Durable local process control. Never infer process ownership from a port."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import re
import subprocess
import sys
import time
import uuid
from urllib.error import URLError
from urllib.request import urlopen

from .registry import Registry, Step, UserError


def read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default.copy()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        return data
    except (ValueError, OSError) as exc:
        raise UserError(f"Cannot read {path}; preserve it and repair the JSON: {exc}") from exc


def atomic_json(path: Path, value: dict):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.chmod(0o600)
    temp.replace(path)


def identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def owned(record: dict, root: Path) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or not record.get("identity") or identity(pid) != record["identity"]:
        return False
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        return ("ai_cli.runner" in argv and record.get("token") in argv
                and str(root) in argv and os.getpgid(pid) == pid)
    except (OSError, UnicodeError):
        return False


def port_available(port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class Manager:
    def __init__(self, root: Path):
        if sys.platform != "linux":
            raise UserError("AI process management currently supports Linux (/proc and process groups).")
        self.root = root.resolve()
        self.registry = Registry(self.root)
        self.runtime = self.root / ".ai"
        self.runtime.mkdir(mode=0o700, exist_ok=True)
        (self.runtime / "logs").mkdir(exist_ok=True)
        self.children: dict[int, subprocess.Popen] = {}

    def reap(self):
        for pid, process in list(self.children.items()):
            if process.poll() is not None:
                self.children.pop(pid, None)

    def prune_logs(self, key: str, category: str):
        pattern = re.compile(rf"{re.escape(key)}-{category}-[0-9a-f]{{32}}\.log$")
        logs = [p for p in (self.runtime / "logs").glob(f"{key}-{category}-*.log")
                if pattern.fullmatch(p.name) and not p.is_symlink()]
        for old in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[2:]:
            old.unlink(missing_ok=True)
            old.with_suffix(".log.1").unlink(missing_ok=True)

    @contextmanager
    def locked(self):
        with (self.runtime / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def state(self) -> dict:
        data = read_json(self.runtime / "state.json", {"runs": {}})
        runs = data.get("runs")
        if not isinstance(runs, dict) or any(
            not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*:(service|job)", key)
            or not isinstance(record, dict) or not isinstance(record.get("status"), str)
            for key, record in runs.items()
        ):
            raise UserError("Invalid .ai/state.json run data; preserve it and repair the JSON")
        return data

    def save(self, state: dict):
        atomic_json(self.runtime / "state.json", state)

    def ports(self) -> dict:
        ports = read_json(self.runtime / "ports.json", {})
        if any(type(number) is not int or not 1024 <= number <= 65535 for number in ports.values()):
            raise UserError("Invalid .ai/ports.json; ports must be integers between 1024 and 65535")
        return ports

    def port(self, key: str) -> int | None:
        return self.ports().get(key, self.registry.get(key).port)

    def set_port(self, key: str, port: int):
        project = self.registry.get(key)
        if "start" not in project.actions:
            raise UserError(f"{key} has no site/service to assign a port")
        if type(port) is not int or not 1024 <= port <= 65535:
            raise UserError("Choose a port between 1024 and 65535")
        with self.locked():
            runs = self.state()["runs"]
            record = runs.get(f"{key}:service", {})
            if owned(record, self.root):
                raise UserError("Stop the service before changing its port")
            ports = self.ports()
            for other in self.registry.projects.values():
                if other.id != key and ports.get(other.id, other.port) == port:
                    raise UserError(f"Port {port} is configured for {other.id}")
            if not port_available(port):
                raise UserError(f"Port {port} is in use by another process; nothing was stopped")
            ports[key] = port
            atomic_json(self.runtime / "ports.json", ports)

    def launch(self, key: str, action: str, *, steps: list[Step] | None = None) -> dict:
        project = self.registry.get(key)
        category = "service" if action == "start" else "job"
        run_key = f"{key}:{category}"
        with self.locked():
            state = self.state()
            old = state["runs"].get(run_key, {})
            if owned(old, self.root):
                raise UserError(f"{key} already has a running {category}; stop it first")
            other = state["runs"].get(f"{key}:{'job' if category == 'service' else 'service'}", {})
            if owned(other, self.root) and (category == "service" or action != "test"):
                raise UserError(f"{key} has an active {other['action']} run; stop it before {action}")
            # Native builds share files; prevent setup/build/test/run jobs overlapping.
            port = self.port(key) if category == "service" else None
            if port:
                for other_key, other in state["runs"].items():
                    if other_key != run_key and other.get("port") == port and owned(other, self.root):
                        raise UserError(f"Port {port} belongs to managed {other['project']}")
                if not port_available(port):
                    raise UserError(f"Port {port} is in use by another process; nothing was stopped")
            commands = steps if steps is not None else self.registry.plan(key, action, port)
            self.prune_logs(key, category)  # Keep two previous runs plus this new run.
            token = uuid.uuid4().hex
            record = {"project": key, "key": run_key, "action": action, "token": token,
                      "port": port, "url": project.url(port) if port else None,
                      "started_at": time.time(), "status": "starting", "returncode": None,
                      "steps": [step.as_dict() for step in commands],
                      "log": str(self.runtime / "logs" / f"{key}-{category}-{token}.log")}
            state["runs"][run_key] = record
            self.save(state)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            env["PYTHONUNBUFFERED"] = "1"
            try:
                process = subprocess.Popen([sys.executable, "-m", "ai_cli.runner", "--root",
                                            str(self.root), "--key", run_key, "--token", token],
                                           cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           start_new_session=True)
                record["pid"] = process.pid
                record["identity"] = identity(process.pid)
                self.children[process.pid] = process
                self.save(state)
            except OSError as exc:
                record.update(status="failed", returncode=127, error=str(exc))
                self.save(state)
                raise UserError(f"Unable to launch {key}: {exc}") from exc
        return record

    def snapshot(self) -> list[dict]:
        self.reap()
        state = self.state()
        ports = self.ports()
        result = []
        for project in self.registry.projects.values():
            runs = {}
            for category in ("service", "job"):
                record = state["runs"].get(f"{project.id}:{category}")
                if record:
                    record = record.copy()
                    record["alive"] = owned(record, self.root)
                    if not record["alive"] and record["status"] in ("starting", "running", "stopping"):
                        record["status"] = "stale"
                    runs[category] = record
            port = ports.get(project.id, project.port)
            result.append({"id": project.id, "label": project.label, "group": project.group,
                           "path": str(project.path), "kind": project.kind,
                           "available": project.available, "actions": project.actions,
                           "port": port, "url": project.url(port), "runs": runs})
        return result

    def record(self, key: str, category: str = "service") -> dict | None:
        record = self.state()["runs"].get(f"{key}:{category}")
        if key not in self.registry.projects and not record:
            self.registry.get(key)
        return record

    def ready(self, record: dict) -> bool:
        if not owned(record, self.root):
            return False
        if not record.get("url"):
            return record.get("status") == "running"
        try:
            with urlopen(record["url"], timeout=0.5) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    def wait_ready(self, key: str, timeout: float = 20) -> dict:
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            record = self.record(key)
            if record and self.ready(record):
                return record
            if record and record["status"] in ("failed", "stopped", "finished"):
                raise UserError(f"{key} exited before becoming ready; run 'ai logs {key}'")
            time.sleep(0.15)
        raise UserError(f"{key} is still starting; check status/logs. The managed process was left running.")

    def wait_job(self, key: str, timeout: float | None = None) -> dict:
        until = time.monotonic() + timeout if timeout is not None else float("inf")
        while time.monotonic() < until:
            self.reap()
            record = self.record(key, "job")
            if not record:
                raise UserError(f"No job for {key}")
            if record["status"] in ("failed", "finished", "stopped"):
                return record
            if not owned(record, self.root):
                latest = self.record(key, "job")
                if latest and latest["status"] in ("failed", "finished", "stopped"):
                    return latest
                raise UserError(f"Job {key} lost its supervisor; check logs")
            time.sleep(0.1)
        raise UserError(f"{key} job still running; check status/logs")

    def stop(self, key: str, category: str = "service", timeout: float = 5) -> bool:
        with self.locked():
            state = self.state()
            record = state["runs"].get(f"{key}:{category}")
            if key not in self.registry.projects and not record:
                self.registry.get(key)
            if not record or not owned(record, self.root):
                return False
            record["status"] = "stopping"
            self.save(state)
            try:
                os.killpg(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                return False
        until = time.monotonic() + timeout
        while owned(record, self.root) and time.monotonic() < until:
            time.sleep(0.05)
        if owned(record, self.root):
            try:
                os.killpg(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            with self.locked():
                state = self.state()
                current = state["runs"].get(record["key"], {})
                if current.get("token") == record["token"]:
                    current.update(status="stopped", returncode=-signal.SIGKILL, ended_at=time.time())
                    self.save(state)
        self.reap()
        return True

    def stop_all(self):
        # A repository removed from the current registry may still have an owned run.
        for key in list(self.state()["runs"]):
            project, category = key.rsplit(":", 1)
            self.stop(project, category)

    def logs(self, key: str, category: str = "service", limit: int = 32768) -> str:
        record = self.record(key, category)
        if not record:
            return "No runs yet."
        if not isinstance(record.get("log"), str):
            return "No log path recorded for this run."
        path = Path(record["log"]).resolve()
        if not path.is_relative_to((self.runtime / "logs").resolve()):
            raise UserError("Log path escapes the runtime folder")
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - limit))
                return stream.read(limit).decode("utf-8", errors="replace")
        except FileNotFoundError:
            return "Waiting for log output…"
