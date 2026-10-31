"""JSON identities and complete local artifact publication."""

import hashlib
import json
import os
import platform
import subprocess
import uuid
from importlib.metadata import version
from pathlib import Path


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def sync_directory(directory: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durable_write(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def publish_directory(destination: Path, files: dict[str, bytes]) -> None:
    """Publish once on one filesystem; retain incomplete saves for inspection."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.pending"
    temporary.mkdir()
    for name, content in files.items():
        if Path(name).name != name:
            raise ValueError("Artifact members must be plain filenames.")
        durable_write(temporary / name, content)
    sync_directory(temporary)
    os.rename(temporary, destination)
    sync_directory(destination.parent)


def implementation_identity(project_root: Path) -> dict:
    files = [*project_root.glob("*.py"), *project_root.glob("src/**/*")]
    identities = {f.relative_to(project_root).as_posix(): digest(f.read_bytes())
                  for f in sorted(files) if f.is_file() and f.suffix in (".py", ".html", ".css", ".js")}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    return {"git_head": head, "files_sha256": identities,
            "python": platform.python_version(), "numpy": version("numpy"),
            "dependencies_sha256": {n: digest((project_root/n).read_bytes()) for n in ("pyproject.toml", "uv.lock")}}
