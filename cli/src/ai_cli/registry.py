"""Discover checked-out submodules without walking content or raw sources."""
from __future__ import annotations

import configparser
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from dataclasses import dataclass, field


class UserError(Exception):
    """A recoverable management error to show without a traceback."""


def workspace_root() -> Path:
    if value := os.environ.get("AI_WORKSPACE"):
        return Path(value).resolve()
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "ai_projects" / ".gitmodules").exists():
            return base
    return Path(__file__).resolve().parents[4]


def contained(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise UserError(f"Project path escapes workspace: {value}")
    if "sources" in result.relative_to(root.resolve()).parts:
        raise UserError("Raw source archives cannot be managed as projects.")
    return result


@dataclass(frozen=True)
class Step:
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"argv": list(self.argv), "cwd": self.cwd, "env": self.env}


@dataclass(frozen=True)
class Project:
    id: str
    label: str
    group: str
    path: Path
    kind: str
    scripts: tuple[str, ...] = ()
    port: int | None = None
    url_path: str = "/"
    overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.path.is_dir()

    @property
    def actions(self) -> tuple[str, ...]:
        result = list(self.overrides)
        if self.kind == "node":
            result.append("setup")
            if "dev" in self.scripts or "preview" in self.scripts:
                result.append("start")
            result.extend(a for a in ("build", "test") if a in self.scripts)
        if self.kind == "python":
            result.extend(("setup", "build", "test"))
            if (self.path / "demo.py").is_file():
                result.append("run")
        if self.kind == "cmake":
            result.extend(("build", "test"))
        return tuple(dict.fromkeys(result))

    def url(self, port: int | None = None) -> str | None:
        number = port or self.port
        return f"http://127.0.0.1:{number}{self.url_path}" if number else None


class Registry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.projects: dict[str, Project] = {}
        self._submodules("ai_projects", "Projects")
        self._submodules("ai_research", "Research")
        self._custom()

    def _add(self, key: str, label: str, group: str, path: str,
             port: int | None = None, url: str = "/", overrides: dict | None = None):
        if not all(isinstance(value, str) and value for value in (key, label, group, path, url)):
            raise UserError("Project id, label, group, path and url_path must be nonempty strings")
        if group not in ("Projects", "Research"):
            raise UserError("Project group must be Projects or Research")
        repo = contained(self.root, path)
        scripts = ()
        kind = "repository"
        if (repo / "package.json").is_file():
            try:
                package = json.loads((repo / "package.json").read_text())
                if not isinstance(package, dict) or not isinstance(package.get("scripts", {}), dict):
                    raise ValueError("scripts must be an object")
                scripts = tuple(package.get("scripts", {}))
            except (ValueError, OSError) as exc:
                raise UserError(f"Invalid package.json in {repo}: {exc}") from exc
            kind = "node"
        elif (repo / "pyproject.toml").is_file():
            kind = "python"
        elif (repo / "CMakeLists.txt").is_file():
            kind = "cmake"
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", key):
            raise UserError(f"Invalid project id: {key}")
        if not url.startswith("/") or any(c in url for c in "\r\n?#"):
            raise UserError(f"Invalid URL path for {key}")
        if port is not None and (type(port) is not int or not 1024 <= port <= 65535):
            raise UserError(f"Invalid port for {key}; use 1024..65535")
        if key in self.projects and overrides is None:
            raise UserError(f"Duplicate project id: {key}; set an explicit id in ai-workspace.toml")
        self.projects[key] = Project(key, label, group, repo, kind, scripts, port, url, overrides or {})

    def _submodules(self, owner: str, group: str):
        file = self.root / owner / ".gitmodules"
        if not file.exists():
            return
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(file)
        except configparser.Error as exc:
            raise UserError(f"Invalid {file}: {exc}") from exc
        for section in parser.sections():
            if not parser.has_option(section, "path"):
                continue
            relative = parser.get(section, "path")
            contained(self.root / owner, relative)
            key = Path(relative).name.replace("_", "-")
            if key in self.projects:
                key = f"{group.lower()}-{key}"
            port = 4274 if key == "research-docs" else 4273 if key == "project-docs" else None
            url = "/research_docs/" if key == "research-docs" else "/project_docs/" if key == "project-docs" else "/"
            self._add(key, key.replace("-", " ").title(), group, f"{owner}/{relative}", port, url)
        # Assign predictable ports to other discovered websites.
        next_port = 4380 if group == "Research" else 4480
        for key, project in list(self.projects.items()):
            if project.group == group and "start" in project.actions and project.port is None:
                self._add(key, project.label, group, str(project.path.relative_to(self.root)),
                          next_port, overrides={})
                next_port += 1

    def _custom(self):
        file = self.root / "ai-workspace.toml"
        if not file.exists():
            return
        try:
            data = tomllib.loads(file.read_text())
            configs = data.get("projects", {})
            if not isinstance(configs, dict):
                raise UserError("ai-workspace.toml projects must be tables")
            for key, config in configs.items():
                if not isinstance(config, dict):
                    raise UserError(f"Project {key} must be a TOML table")
                old = self.projects.get(key)
                overrides = dict(old.overrides) if old else {}
                for action in ("start", "build", "test", "run", "setup"):
                    if action in config:
                        argv = config[action]
                        if not isinstance(argv, list) or not argv or not all(isinstance(s, str) and s for s in argv):
                            raise UserError(f"{key}.{action} must be a nonempty argv array")
                        overrides[action] = tuple(argv)
                self._add(key, config.get("label", old.label if old else key),
                          config.get("group", old.group if old else "Projects"),
                          config.get("path", str(old.path.relative_to(self.root)) if old else key),
                          config.get("port", old.port if old else None),
                          config.get("url_path", old.url_path if old else "/"), overrides)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise UserError(f"Invalid ai-workspace.toml: {exc}") from exc

    def get(self, key: str) -> Project:
        try:
            return self.projects[key]
        except KeyError as exc:
            raise UserError(f"Unknown project {key!r}; run 'ai list'") from exc

    def plan(self, key: str, action: str, port: int | None = None) -> list[Step]:
        project = self.get(key)
        if not project.available:
            raise UserError(f"{key} is not checked out: {project.path}")
        cwd = str(project.path)
        env = {"PORT": str(port or project.port)} if port or project.port else {}
        if action in project.overrides:
            argv = tuple(arg.replace("{port}", str(port or project.port or ""))
                         .replace("{workspace}", str(self.root)).replace("{project}", cwd)
                         for arg in project.overrides[action])
            if any("{port}" in arg for arg in project.overrides[action]) and not (port or project.port):
                raise UserError(f"{key}: configure a port to use the {{port}} placeholder")
            return [Step(argv, cwd, env)]
        if project.kind == "python" and action in ("setup", "build", "test", "run"):
            return self._python(project, action)
        if project.kind == "node":
            if action == "setup":
                return [Step(("npm", "ci"), cwd)]
            script = ("dev" if "dev" in project.scripts else "preview") if action == "start" else action
            if script not in project.scripts:
                raise UserError(f"{key} has no {action} script")
            steps = [] if (project.path / "node_modules").is_dir() else [Step(("npm", "ci"), cwd)]
            return steps + [Step(("npm", "run", script), cwd, env)]
        if project.kind == "cmake" and action in ("build", "test", "run", "setup"):
            return self._cmake(project, action)
        raise UserError(f"{key} does not support {action}; configure an argv command in ai-workspace.toml")

    def _python(self, project: Project, action: str) -> list[Step]:
        cwd = str(project.path)
        if not (project.path / "uv.lock").is_file():
            raise UserError(f"{project.id}: create uv.lock with 'uv lock' before managed jobs")
        venv = str(self.root / ".ai" / "envs" / project.id)
        env = {"UV_PROJECT_ENVIRONMENT": venv, "UV_LINK_MODE": "copy", "MPLBACKEND": "Agg"}
        sync = Step(("uv", "sync", "--frozen", "--project", cwd, "--python", sys.executable), cwd, env)
        python = str(Path(venv) / "bin" / "python")
        if action == "setup":
            return [sync]
        if action == "build":
            out = str(self.root / ".ai" / "builds" / project.id)
            return [sync, Step(("uv", "build", "--project", cwd, "--out-dir", out), cwd, env)]
        if action == "test":
            if not (project.path / "tests").is_dir():
                raise UserError(f"{project.id}: no tests directory; no tests is a failure")
            return [sync, Step((python, "-m", "pytest", "-q", "tests"), cwd, env)]
        if not (project.path / "demo.py").is_file():
            raise UserError(f"{project.id}: add demo.py or configure an explicit run command")
        return [sync, Step((python, "demo.py"), cwd, env)]

    def _cmake(self, project: Project, action: str) -> list[Step]:
        cwd = str(project.path)
        build = str(self.root / ".ai" / "builds" / project.id)
        venv = str(self.root / ".ai" / "envs" / project.id)
        python = str(Path(venv) / "bin" / "python")
        cmake = (project.path / "CMakeLists.txt").read_text()
        binding = re.search(r"option\((\w+_BUILD_PYTHON)\b", cmake)
        steps = []
        if binding:
            if not (project.path / "uv.lock").is_file():
                raise UserError(f"{project.id}: Python build requires uv.lock or custom build/test commands")
            steps.append(Step(("uv", "sync", "--frozen", "--project", cwd, "--group", "dev",
                               "--no-install-project", "--python", sys.executable), cwd,
                              {"UV_PROJECT_ENVIRONMENT": venv, "UV_LINK_MODE": "copy"}))
        if action == "setup":
            if not steps:
                raise UserError(f"{project.id} needs only CMake and a C++ compiler; no setup job")
            return steps
        configure = ["cmake", "-S", cwd, "-B", build, "-DCMAKE_BUILD_TYPE=Debug", "-DBUILD_TESTING=ON"]
        configure += [f"-D{option}=ON" for option in re.findall(r"option\((\w+_BUILD_TESTS)\b", cmake)]
        if binding:
            configure += [f"-D{binding[1]}=ON", f"-DPython_EXECUTABLE={python}"]
        steps += [Step(tuple(configure), cwd),
                  Step(("cmake", "--build", build, "--parallel", "2"), cwd)]
        if binding:
            module = binding[1].removesuffix("_BUILD_PYTHON").lower()
            steps.append(Step((python, "-c", f"import {module}; print('Python bindings loaded')"),
                              cwd, {"PYTHONPATH": build}))
        if action == "test":
            tests = sorted((project.path / "python").glob("test_*.py"))
            options = ("--cpp-only",) if binding and tests else ()
            steps.append(Step((sys.executable, "-m", "ai_cli.native_tests", build, *options), cwd))
            if binding and tests:
                steps.append(Step((python, "-m", "pytest", "-q", *(str(p) for p in tests)), cwd,
                                  {"PYTHONPATH": build, "MPLBACKEND": "Agg"}))
        return steps
