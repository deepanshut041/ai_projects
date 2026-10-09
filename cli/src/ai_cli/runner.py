"""Token-bound detached supervisor, with bounded persistent output."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import traceback

from .manager import Manager


class RunLog:
    def __init__(self, path: Path, limit: int = 2 * 1024 * 1024):
        self.path = path
        self.limit = limit
        self.lock = threading.Lock()
        self.stream = path.open("ab", buffering=0)

    def write(self, data: bytes):
        with self.lock:
            if self.stream.tell() + len(data) > self.limit:
                self.stream.close()
                self.path.replace(self.path.with_suffix(".log.1"))
                self.stream = self.path.open("ab", buffering=0)
            self.stream.write(data)

    def say(self, message: str):
        self.write((message + "\n").encode())


def descendants_in_group() -> bool:
    """Keep ownership alive while a cancelled descendant ignores SIGTERM."""
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and int(path.name) != os.getpid():
            try:
                stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if stat[0] != "Z" and int(stat[2]) == os.getpid():
                    return True
            except (OSError, IndexError, ValueError):
                continue
    return False


def supervise(manager: Manager, key: str, token: str) -> int:
    with manager.locked():
        state = manager.state()
        record = state["runs"].get(key)
        if not record or record["token"] != token or record.get("pid") != os.getpid():
            return 1
        record["status"] = "running"
        manager.save(state)
    log = RunLog(Path(record["log"]))
    interrupted = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: interrupted.set())
    code = 0
    child = None
    try:
        for step in record["steps"]:
            if interrupted.is_set():
                code = -signal.SIGTERM
                break
            env = os.environ.copy()
            env.update(step["env"])
            log.say(f"$ {shlex.join(step['argv'])}\n  cwd: {step['cwd']}")
            child = subprocess.Popen(step["argv"], cwd=step["cwd"], env=env,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)

            def pump(child=child):
                while data := child.stdout.read1(8192):
                    log.write(data)

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            while child.poll() is None:
                # SIGTERM was delivered to the entire process group by the controller.
                # Keep the supervisor alive until the child exits, allowing safe escalation.
                time.sleep(0.05)
            code = child.returncode
            reader.join(timeout=2)
            while descendants_in_group():
                time.sleep(0.05)
            if interrupted.is_set():
                code = -signal.SIGTERM
                break
            if code:
                break
    except Exception:
        log.say(traceback.format_exc())
        code = 127
    finally:
        status = "stopped" if interrupted.is_set() else ("failed" if code else "finished")
        log.say(f"\n[{status}; exit {code}]")
        with manager.locked():
            state = manager.state()
            current = state["runs"].get(key, {})
            if current.get("token") == token:
                current.update(status=status, returncode=code, ended_at=time.time())
                manager.save(state)
        log.stream.close()
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    return supervise(Manager(args.root), args.key, args.token)


if __name__ == "__main__":
    raise SystemExit(main())
