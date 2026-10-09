"""Create local planning packages without overwriting existing work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

WORKSPACE = Path(__file__).resolve().parents[4]
TEMPLATES = Path(__file__).resolve().parent / "templates"
DOMAINS = {
    "project": "ai-project-workflow",
    "paper": "ai-paper-review",
    "experiment": "ai-experiment-planning",
    "reproduction": "ai-reproduction-planning",
}
FULL = (
    "01-research.md", "02-human-plan.md", "02-implementation-plan.md",
    "03-tasks.md", "04-diagrams.md", "05-tracker.md", "06-review.md", "00-resume.md",
)
COMPACT = ("02-human-plan.md", "05-tracker.md", "00-resume.md")
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def calendar_month(now=None):
    now = now or datetime.now().astimezone()
    return f"{now.year}-{now.month:02d}"


def task_slug(title, explicit=None):
    value = explicit if explicit is not None else re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) or len(value) > 80 or value in RESERVED:
        raise ValueError("Use --slug with 1-80 lowercase letters/digits and internal hyphens; paths and Windows device names are not allowed.")
    return value


def selected_domains(value):
    result = list(dict.fromkeys(part.strip() for part in value.split(",")))
    unknown = [part for part in result if part not in DOMAINS]
    if unknown:
        raise ValueError(f"Unknown domains {unknown}; choose from {', '.join(DOMAINS)}.")
    return result


def packages(root):
    root = Path(root).resolve()
    found = []
    if not root.exists():
        return found
    for month in sorted(root.iterdir()):
        if not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", month.name) or not month.is_dir() or not month.resolve().is_relative_to(root):
            continue
        for folder in sorted(month.iterdir()):
            if folder.is_dir() and folder.resolve().is_relative_to(root):
                # Include incomplete/legacy task folders so they are not silently replaced.
                found.append(folder)
    return found


def initialize(root, title, domains, *, slug=None, month=None, source_request="", compact=False):
    title = title.strip()
    if not title or "\n" in title or "\r" in title:
        raise ValueError("Provide a nonempty one-line task title.")
    if not source_request.strip():
        raise ValueError("Provide --source-request with the actual request or an accurate restatement.")
    domains = selected_domains(domains)
    slug = task_slug(title, slug)
    month = month or calendar_month()
    if not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", month):
        raise ValueError("Month must look like 2026-09.")
    root = Path(root).resolve()
    target = root / month / slug
    if not target.resolve().is_relative_to(root):
        raise ValueError("Resolved task path leaves the selected implementation root.")
    matches = [folder for folder in packages(root) if folder.name == slug]
    if matches:
        raise FileExistsError("Reuse the existing task and its 00-resume.md/05-tracker.md: " + ", ".join(str(path) for path in matches))
    if target.exists():
        raise FileExistsError(f"Target already exists and was preserved: {target}")
    names = COMPACT if compact else FULL
    values = {
        "title": title,
        "slug": slug,
        "month": month,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_request": source_request.strip(),
        "lead_skill": DOMAINS[domains[0]],
        "selected_skills": ", ".join(DOMAINS[domain] for domain in domains),
        "package_size": "compact" if compact else "full",
        "artifact_links": "\n".join(f"- [{name}]({name})" for name in names),
    }
    # Render/validate all resources before creating a destination directory.
    rendered = {}
    for name in names:
        template = (TEMPLATES / name).read_text(encoding="utf-8")
        rendered[name] = re.sub(r"\{\{([a-z_]+)\}\}", lambda match: values[match[1]], template)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()  # Exclusive creation also protects against concurrent initializers.
    for name, content in rendered.items():
        with (target / name).open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an unassessed plan scaffold; never execute or overwrite it.")
    init.add_argument("title")
    init.add_argument("--domains", required=True, help="Comma-separated domains, lead first: " + ", ".join(DOMAINS))
    init.add_argument("--source-request", required=True)
    init.add_argument("--slug")
    init.add_argument("--month")
    init.add_argument("--compact", action="store_true")
    listing = sub.add_parser("list", help="List existing task folders, including older quarters.")
    listing.add_argument("--json", action="store_true")
    for command in (init, listing):
        command.add_argument("--root", type=Path, default=WORKSPACE / "docs" / "implementation", help="Implementation root (default: this workspace's docs/implementation folder).")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            path = initialize(args.root, args.title, args.domains, slug=args.slug, month=args.month, source_request=args.source_request, compact=args.compact)
            print(path)
            print("Scaffold created. Research, execution scope, validation, and completion remain unassessed.")
        else:
            found = packages(args.root)
            if args.json:
                print(json.dumps([{"path": str(path), "month": path.parent.name, "slug": path.name, "has_resume": (path / "00-resume.md").is_file()} for path in found], indent=2))
            else:
                print("\n".join(str(path) for path in found) or "No task folders yet.")
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
