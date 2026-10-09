"""Headless controls and the interactive home use the same manager."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import subprocess
import webbrowser

from .manager import Manager
from .registry import UserError, workspace_root


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="ai", description="AI home — sites, ports, builds and tests")
    result.add_argument("--root", type=Path, default=workspace_root(), help="AI workspace folder")
    sub = result.add_subparsers(dest="command")
    tui = sub.add_parser("tui", help="Open the terminal dashboard (default)")
    tui.add_argument("--smoke", action="store_true", help="Noninteractive configuration/UI check")
    for name in ("list", "status", "doctor"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--json", action="store_true")
    for name in ("start", "restart", "build", "test", "run", "setup", "plan"):
        cmd = sub.add_parser(name)
        cmd.add_argument("project", help="Project id from 'ai list'")
        if name == "plan":
            cmd.add_argument("action", choices=["start", "build", "test", "run", "setup"])
        cmd.add_argument("--json", action="store_true")
        if name in ("start", "restart"):
            cmd.add_argument("--port", type=int, help="Persist a different port (requires stopped service)")
            cmd.add_argument("--no-wait", action="store_true", help="Return before HTTP readiness")
        elif name != "plan":
            cmd.add_argument("--detach", action="store_true", help="Run job in background; view status/logs later")
    stop = sub.add_parser("stop")
    stop.add_argument("project")
    stop.add_argument("--job", action="store_true", help="Cancel its build/test/setup/run job")
    sub.add_parser("stop-all", help="Stop all tool-owned services and jobs")
    logs = sub.add_parser("logs")
    logs.add_argument("project")
    logs.add_argument("--job", action="store_true")
    port = sub.add_parser("port")
    port.add_argument("project")
    port.add_argument("number", type=int)
    open_cmd = sub.add_parser("open", help="Open the project's local website in a browser")
    open_cmd.add_argument("project")
    create = sub.add_parser("new", help="Create and locally register an independent Python repository")
    create.add_argument("area", choices=("project", "research"))
    create.add_argument("slug")
    create.add_argument("--title", required=True)
    research = sub.add_parser("research", help="Author research records and maintain source provenance")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    new_record = research_sub.add_parser("new")
    new_record.add_argument("kind", choices=("paper", "experiment", "question", "collection"))
    new_record.add_argument("slug")
    new_record.add_argument("--title", required=True)
    for name in ("list", "sources"):
        research_sub.add_parser(name)
    fetch = research_sub.add_parser("fetch", help="Explicit bounded download into ignored sources archive")
    fetch.add_argument("source_id")
    fetch.add_argument("--max-mib", type=int, default=20)
    source = research_sub.add_parser("source")
    source_sub = source.add_subparsers(dest="source_command", required=True)
    add = source_sub.add_parser("add")
    add.add_argument("id")
    add.add_argument("--title", required=True)
    add.add_argument("--url", required=True)
    add.add_argument("--collection", default="inbox")
    add.add_argument("--kind", default="Paper")
    tasks = sub.add_parser("task", help="Create or list durable implementation packages")
    task_sub = tasks.add_subparsers(dest="task_command", required=True)
    for name in ("new", "list"):
        task = task_sub.add_parser(name)
        task.add_argument("--owner", default="projects", help="projects, research, or an existing project id")
        if name == "new":
            task.add_argument("title")
            task.add_argument("--domains", default="project")
            task.add_argument("--source-request")
            task.add_argument("--compact", action="store_true")
    check = sub.add_parser("check", help="Run an action across all supported entries and summarize failures")
    check.add_argument("--action", choices=("test", "build", "run", "setup"), default="test")
    check.add_argument("--group", choices=("Projects", "Research"))
    sub.add_parser("relink", help="Recreate local repository caches after moving/restoring the workspace")
    archive = sub.add_parser("snapshot", help="Save editable source and Git histories without caches or raw archives")
    archive.add_argument("output", type=Path)
    return result


def doctor(manager: Manager) -> dict:
    tools = {name: shutil.which(name) for name in ("uv", "node", "npm", "git", "cmake", "ctest", "c++", "make")}
    required = {"uv", "git"}
    if any(p.kind == "node" for p in manager.registry.projects.values()):
        required |= {"node", "npm"}
    missing = sorted(name for name in required if not tools[name])
    missing_repos = [p.id for p in manager.registry.projects.values() if not p.available]
    node_version = subprocess.run([tools["node"], "--version"], capture_output=True, text=True).stdout.strip() if tools["node"] else None
    node_ok = not any(p.kind == "node" for p in manager.registry.projects.values()) or bool(node_version and int(node_version.lstrip("v").split(".")[0]) >= 22)
    return {"ok": not missing and not missing_repos and node_ok, "root": str(manager.root), "runtime": str(manager.runtime), "platform": sys.platform,
            "python": sys.version.split()[0], "tools": tools,
            "required_tools": sorted(required), "missing_required_tools": missing,
            "node_version": node_version, "node_compatible": node_ok,
            "missing_repositories": missing_repos}


def emit(value, as_json: bool = False):
    print(json.dumps(value, indent=2, default=str) if as_json else value)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        manager = Manager(args.root)
        command = args.command or "tui"
        if command == "tui":
            from .tui import AIHome
            app = AIHome(manager)
            if getattr(args, "smoke", False):
                import asyncio

                async def smoke():
                    async with app.run_test(size=(100, 32)) as pilot:
                        await pilot.pause()
                        print(json.dumps({"ui": "ok", "projects": list(manager.registry.projects),
                                          "markdown_browser": False}))

                asyncio.run(smoke())
            elif not sys.stdin.isatty():
                raise UserError("The TUI needs a terminal. Use 'ai tui --smoke', 'status', or 'list' here.")
            else:
                app.run()
        elif command in ("list", "status"):
            rows = manager.snapshot()
            if args.json:
                emit(rows, True)
            else:
                print(f"{'PROJECT':42} {'SERVICE':12} {'JOB':18} PORT")
                for row in rows:
                    service = row["runs"].get("service", {}).get("status", "stopped")
                    job = row["runs"].get("job", {})
                    job_text = f"{job.get('action', '')} {job.get('status', '—')}".strip()
                    print(f"{row['id']:42} {service:12} {job_text:18} {row['port'] or '—'}")
        elif command == "doctor":
            result = doctor(manager)
            emit(result, args.json)
            return 0 if result["ok"] else 1
        elif command == "new":
            from .workflows import new_repository
            with manager.locked():
                emit(new_repository(manager.root, args.area, args.slug, args.title), True)
        elif command == "research":
            from .workflows import new_record, research_root, catalog, add_source, fetch_source
            action = args.research_command
            if action == "new":
                emit(new_record(manager.root, args.kind, args.slug, args.title), True)
            elif action == "list":
                emit([str(p.relative_to(research_root(manager.root))) for p in sorted((research_root(manager.root) / "content").glob("*/*.md"))], True)
            elif action == "sources":
                emit(catalog(manager.root), True)
            elif action == "source":
                with manager.locked():
                    emit(add_source(manager.root, args.id, args.title, args.url, args.collection, args.kind), True)
            elif action == "fetch":
                emit(fetch_source(manager.root, args.source_id, args.max_mib), True)
        elif command == "task":
            from .task_packages import initialize, packages
            repo = manager.root / "ai_projects" if args.owner == "projects" else manager.root / "ai_research/research_docs" if args.owner == "research" else manager.registry.get(args.owner).path
            target = repo / "docs/implementation"
            if args.task_command == "list":
                emit([str(p) for p in packages(target)], True)
            else:
                try:
                    path = initialize(target, args.title, args.domains, source_request=args.source_request or args.title, compact=args.compact)
                except (ValueError, KeyError) as exc:
                    raise UserError(str(exc)) from exc
                emit({"path": str(path), "state": "scaffold; scope and completion unassessed"}, True)
        elif command == "check":
            results = []
            for project in manager.registry.projects.values():
                if args.group and project.group != args.group or args.action not in project.actions:
                    continue
                try:
                    manager.launch(project.id, args.action)
                    record = manager.wait_job(project.id)
                    ok = record["status"] == "finished" and record["returncode"] == 0
                    results.append({"id": project.id, "ok": ok, "returncode": record["returncode"]})
                    print(f"{project.id}: {'passed' if ok else 'failed'}", file=sys.stderr)
                except (UserError, OSError) as exc:
                    results.append({"id": project.id, "ok": False, "error": str(exc)})
            if not results:
                raise UserError('No projects support the requested action/group')
            emit(results, True)
            return 0 if all(row["ok"] for row in results) else 1
        elif command == "relink":
            from .workflows import relink
            with manager.locked():
                emit(relink(manager.root), True)
        elif command == "snapshot":
            from .workflows import snapshot
            emit(snapshot(manager.root, args.output), True)
        elif command == "plan":
            steps = manager.registry.plan(args.project, args.action, manager.port(args.project))
            if args.json:
                emit([s.as_dict() for s in steps], True)
            else:
                import shlex
                for step in steps:
                    print(f"{step.cwd}\n  {shlex.join(step.argv)}")
        elif command in ("start", "restart"):
            if command == "restart":
                manager.stop(args.project)
            if args.port is not None:
                manager.set_port(args.project, args.port)
            record = manager.launch(args.project, "start")
            if not args.no_wait:
                record = manager.wait_ready(args.project)
            emit(record if args.json else f"{args.project}: {record['url'] or 'started'} (pid {record['pid']})", args.json)
        elif command in ("build", "test", "run", "setup"):
            record = manager.launch(args.project, command)
            if args.detach:
                emit(record if args.json else f"{command} started for {args.project}; see status and logs --job", args.json)
            else:
                record = manager.wait_job(args.project)
                if args.json:
                    emit(record, True)
                else:
                    print(manager.logs(args.project, "job"))
                return 0 if record["status"] == "finished" and record["returncode"] == 0 else 1
        elif command == "stop":
            stopped = manager.stop(args.project, "job" if args.job else "service")
            print("Stopped." if stopped else "No owned process running.")
        elif command == "stop-all":
            manager.stop_all()
            print("Stopped all owned processes.")
        elif command == "logs":
            print(manager.logs(args.project, "job" if args.job else "service"))
        elif command == "port":
            manager.set_port(args.project, args.number)
            print(f"{args.project}: port {args.number}")
        elif command == "open":
            project = manager.registry.get(args.project)
            url = project.url(manager.port(args.project))
            if not url:
                raise UserError("This project has no website")
            webbrowser.open(url)
            print(url)
        return 0
    except KeyboardInterrupt:
        print("Detached run continues; use 'ai stop PROJECT --job' to cancel.", file=sys.stderr)
        return 130
    except (UserError, OSError) as exc:
        print(f"ai: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
