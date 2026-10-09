"""Workspace authoring tools; all generated destinations are bounded and exclusive."""
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib
import uuid
from urllib.parse import urlparse
from urllib.request import urlopen

from .registry import UserError


def slug(value):
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*', value) or len(value) > 64:
        raise UserError('Use a slug of 1-64 lowercase letters, digits and internal hyphens, starting with a letter')
    return value


def write_json(path, value):
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def git(repo, *args):
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    if result.returncode:
        raise UserError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def register(root, owner, child):
    """Local Git snapshots are necessary for a real gitlink; nothing is pushed."""
    git(child, 'init', '-b', 'main')
    git(child, 'add', '.')
    git(child, 'commit', '-m', 'Initialize local workspace repository')
    cache = root / '.ai/repositories' / (child.name + '.git')
    cache.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['git', 'clone', '--bare', str(child), str(cache)], capture_output=True, text=True)
    if result.returncode:
        raise UserError(result.stderr.strip())
    relative = str(child.relative_to(owner))
    git(owner, '-c', 'protocol.file.allow=always', 'submodule', 'add', '../.ai/repositories/' + cache.name, relative)
    git(owner, 'submodule', 'absorbgitdirs', relative)
    git(child, 'remote', 'add', 'origin', str(cache))


def new_repository(root, area, name, title):
    slug(name)
    if not title.strip() or '\n' in title or '\r' in title:
        raise UserError('Provide a nonempty one-line title')
    from .registry import Registry
    if name in Registry(root).projects:
        raise UserError(f'Project id already exists: {name}')
    owner = root / ('ai_projects' if area == 'project' else 'ai_research')
    if not (owner / '.git').exists():
        raise UserError(f'Initialize the {owner.name} umbrella before adding repositories')
    target = owner / 'projects' / name if area == 'project' else owner / name
    if not target.resolve().is_relative_to(owner.resolve()):
        raise UserError('Scaffold path escapes its owning umbrella')
    if target.exists():
        raise UserError(f'Existing destination was preserved: {target}')
    if (root / '.ai/repositories' / (name + '.git')).exists():
        raise UserError('A local repository cache already has that name; preserve it and choose another slug')
    # Check identity before creating files; use the user's configured local identity.
    for field in ('user.name', 'user.email'):
        git(owner, 'config', '--get', field)
    package = name.replace('-', '_')
    target.mkdir(parents=True)
    (target / 'src' / package).mkdir(parents=True)
    (target / 'tests').mkdir()
    (target / 'tests/.gitkeep').write_text('')
    (target / 'src' / package / '__init__.py').write_text('"""Implementation pending; define the contract before adding behavior."""\n')
    (target / 'demo.py').write_text(f'import json\nprint(json.dumps({{"project": {name!r}, "state": "scaffold; implementation pending"}}))\n')
    (target / '.gitignore').write_text('.venv/\n__pycache__/\n.pytest_cache/\ndist/\nruns/\ndata/\nweights/\n.env\n.env.*\n*.egg-info/\n')
    (target / 'README.md').write_text(f'# {title}\n\nAn independent {area} repository. Implementation and tests are pending.\n\nUse `./ai task new "Define {name}" --owner {name}` from the workspace. Specify input/output contracts, data provenance, baselines, leakage controls, numerical tolerances and compute limits before implementation.\n\n`./ai setup {name}`, `./ai build {name}` and `./ai run {name}` exercise the scaffold. `./ai test {name}` intentionally fails until real tests exist.\n')
    (target / 'AGENTS.md').write_text(f'# {title}\n\nRead README.md and the workspace research workflow before editing. Keep an implementation package in docs/implementation/. Preserve history. Keep raw data, weights, credentials and generated runs ignored. Validate independent reference results and failure cases. No external publication is implied.\n')
    (target / 'pyproject.toml').write_text(f'''[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = []

[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[dependency-groups]
dev = ["pytest>=9,<10"]

[tool.pytest.ini_options]
testpaths = ["tests"]
''')
    result = subprocess.run(['uv', 'lock', '--project', str(target)], capture_output=True, text=True)
    if result.returncode:
        raise UserError(f'Scaffold preserved at {target}; uv lock failed: {result.stderr.strip()}')
    try:
        register(root, owner, target)
    except UserError as exc:
        raise UserError(f'Scaffold preserved at {target}; Git registration needs repair: {exc}') from exc
    return {'path': str(target), 'id': name, 'state': 'scaffold', 'parent': 'submodule entry staged; no parent commit made'}


def research_root(root):
    result = root / 'ai_research/research_docs'
    if not result.is_dir():
        raise UserError('research_docs is not checked out')
    return result


def new_record(root, kind, name, title):
    slug(name)
    if not title.strip() or '\n' in title or '\r' in title:
        raise UserError('Provide a nonempty one-line title')
    site = research_root(root)
    categories = {'paper': 'papers', 'experiment': 'experiments', 'question': 'questions', 'collection': 'collections'}
    target = site / 'content' / categories[kind] / (name + '.md')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(site.resolve()):
        raise UserError('Record path escapes research_docs')
    template = site / 'templates' / (kind + '.md' if kind != 'collection' else 'record.md')
    text = template.read_text()
    frontmatter, body = text.split('---', 2)[1:]
    lines = frontmatter.strip().splitlines()
    lines = ['title: ' + json.dumps(title) if line.startswith('title:') else 'updated: ' + json.dumps(date.today().isoformat()) if line.startswith('updated:') else line for line in lines]
    try:
        with target.open('x') as stream:
            stream.write('---\n' + '\n'.join(lines) + '\n---' + body)
    except FileExistsError as exc:
        raise UserError(f'Existing record was preserved: {target}') from exc
    return {'path': str(target), 'state': 'draft; no review or experiment result claimed'}


def catalog(root):
    file = research_root(root) / 'catalog/sources.json'
    try:
        result = json.loads(file.read_text())
        if result.get('schema') != 1 or not isinstance(result.get('sources'), list) or not isinstance(result.get('collections'), list):
            raise ValueError('expected schema 1, collections and sources lists')
        return result
    except (ValueError, OSError, AttributeError) as exc:
        raise UserError(f'Preserve and repair the source catalog: {exc}') from exc


def valid_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise UserError('Use an HTTP(S) source URL without embedded credentials')
    return url


def add_source(root, source_id, title, url, collection, kind):
    slug(source_id)
    slug(collection)
    valid_url(url)
    data = catalog(root)
    if any(source['id'] == source_id for source in data['sources']):
        raise UserError('Existing source id was preserved')
    if not any(group['id'] == collection for group in data['collections']):
        data['collections'].append({'id': collection, 'title': collection.replace('-', ' ').title()})
    entry = {'id': source_id, 'title': title, 'url': url, 'collection': collection, 'kind': kind,
             'access': 'not-checked', 'review': 'not-reviewed', 'added': date.today().isoformat()}
    data['sources'].append(entry)
    write_json(research_root(root) / 'catalog/sources.json', data)
    return entry


def fetch_source(root, source_id, max_mib=20):
    if type(max_mib) is not int or not 1 <= max_mib <= 100:
        raise UserError('max-mib must be an integer between 1 and 100')
    data = catalog(root)
    entry = next((source for source in data['sources'] if source['id'] == source_id), None)
    if not entry:
        raise UserError('Unknown source id')
    url = valid_url(entry['url'])
    folder = research_root(root) / 'sources' / slug(source_id)
    if not folder.resolve().is_relative_to(research_root(root).resolve()):
        raise UserError('Source archive path escapes research_docs')
    if folder.exists():
        raise UserError(f'Existing source archive was preserved: {folder}')
    with urlopen(url, timeout=30) as response:
        valid_url(response.geturl())
        payload = response.read(max_mib * 1024 * 1024 + 1)
        mime = response.headers.get('Content-Type', 'application/octet-stream')
    if len(payload) > max_mib * 1024 * 1024:
        raise UserError('Source exceeds the download limit; nothing archived')
    folder.mkdir(parents=True)
    suffix = '.pdf' if mime.split(';')[0] == 'application/pdf' else '.html' if 'html' in mime else '.bin'
    (folder / ('original' + suffix)).write_bytes(payload)
    metadata = {'id': source_id, 'url': url, 'retrieved': date.today().isoformat(), 'bytes': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest(), 'content_type': mime, 'review': 'not-reviewed'}
    write_json(folder / 'metadata.json', metadata)
    return {'path': str(folder), **metadata}


def relink(root):
    """Recreate local remotes after relocation without editing tracked .gitmodules."""
    cache = root / '.ai/repositories'
    cache.mkdir(parents=True, exist_ok=True)
    linked = []
    import configparser
    for name in ('ai_projects', 'ai_research'):
        owner = root / name
        config = configparser.ConfigParser(interpolation=None)
        config.read(owner / '.gitmodules')
        for section in config.sections():
            relative = config.get(section, 'path')
            child = (owner / relative).resolve()
            if not child.is_relative_to(owner.resolve()):
                raise UserError('Submodule path escapes its owner')
            module_name = section.removeprefix('submodule "').removesuffix('"')
            declared = config.get(section, 'url', fallback='')
            if '://' in declared or declared.startswith('git@'):
                linked.append({'id': child.name, 'remote': declared, 'state': 'external registration preserved'})
                continue
            repo = cache / (child.name + '.git')
            if not repo.exists():
                result = subprocess.run(['git', 'clone', '--bare', str(child), str(repo)], capture_output=True, text=True)
                if result.returncode:
                    raise UserError(result.stderr.strip())
            git(owner, 'config', f'submodule.{module_name}.url', str(repo))
            remote = git(child, 'remote').splitlines()
            if 'origin' in remote:
                existing = git(child, 'remote', 'get-url', 'origin')
                if '://' in existing or existing.startswith('git@'):
                    linked.append({'id': child.name, 'remote': existing, 'state': 'external origin preserved'})
                    continue
                git(child, 'remote', 'set-url', 'origin', str(repo))
            else:
                git(child, 'remote', 'add', 'origin', str(repo))
            linked.append({'id': child.name, 'remote': str(repo)})
    return linked


def snapshot(root, output):
    output = output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise UserError('Choose a new snapshot destination outside the workspace')
    excluded = {'.ai', '.venv', 'node_modules', 'dist', 'sources', 'runs', 'data', 'weights', '__pycache__', '.pytest_cache', '.ruff_cache', 'test-results', '.playwright-cli', 'output'}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(output, 'x:gz') as archive:
            def filtered(info):
                parts = Path(info.name).parts
                if any(part in excluded or part == '.env' or part.startswith('.env.') for part in parts):
                    return None
                return info
            archive.add(root, arcname='ai', filter=filtered)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {'path': str(output), 'bytes': output.stat().st_size, 'includes': 'editable source and Git histories',
            'excludes': sorted(excluded), 'restore': 'Extract to a new directory, then run ./ai relink and ./ai doctor'}
