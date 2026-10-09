import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from ai_cli.registry import Registry, UserError
from ai_cli.workflows import new_record, add_source, fetch_source, snapshot, new_repository, relink
from ai_cli.task_packages import initialize


@pytest.fixture
def authored(tmp_path):
    site = tmp_path / 'ai_research/research_docs'
    (site / 'catalog').mkdir(parents=True)
    (site / 'templates').mkdir()
    (site / 'catalog/sources.json').write_text(json.dumps({'schema': 1, 'collections': [], 'sources': [], 'archive': {'files': 0, 'bytes': 0}}))
    (site / 'templates/question.md').write_text('---\ntitle: "Placeholder"\ndescription: "Draft"\nupdated: "2026-01-01"\nstatus: "open"\n---\n\n## Question\n')
    return tmp_path


def test_records_preserve_existing_work_and_quote_titles(authored):
    result = new_record(authored, 'question', 'my-question', 'What does "good" mean?')
    path = Path(result['path'])
    original = path.read_text()
    assert 'title: "What does \\"good\\" mean?"' in original
    with pytest.raises(UserError, match='preserved'):
        new_record(authored, 'question', 'my-question', 'Replacement')
    assert path.read_text() == original
    with pytest.raises(UserError):
        new_record(authored, 'question', '../escape', 'Unsafe')


def test_source_registration_download_hash_limits_and_overwrite_protection(authored, monkeypatch):
    add_source(authored, 'a-paper', 'Original', 'https://example.org/paper', 'literature', 'Paper')
    with pytest.raises(UserError, match='preserved'):
        add_source(authored, 'a-paper', 'Another', 'https://example.org/other', 'literature', 'Paper')
    with pytest.raises(UserError):
        add_source(authored, 'invalid', 'Bad', 'file:///etc/passwd', 'literature', 'Paper')
    def response(payload):
        stream = io.BytesIO(payload)
        stream.headers = {'Content-Type': 'application/pdf'}
        stream.geturl = lambda: 'https://example.org/paper'
        return stream
    monkeypatch.setattr('ai_cli.workflows.urlopen', lambda *a, **k: response(b'x' * (1024 * 1024 + 1)))
    with pytest.raises(UserError, match='limit'):
        fetch_source(authored, 'a-paper', max_mib=1)
    assert not (authored / 'ai_research/research_docs/sources/a-paper').exists()
    monkeypatch.setattr('ai_cli.workflows.urlopen', lambda *a, **k: response(b'%PDF-test original bytes'))
    result = fetch_source(authored, 'a-paper')
    assert result['bytes'] == 24
    assert len(result['sha256']) == 64
    assert (Path(result['path']) / 'original.pdf').read_bytes() == b'%PDF-test original bytes'
    with pytest.raises(UserError, match='preserved'):
        fetch_source(authored, 'a-paper')


def test_task_packages_do_not_overwrite_or_duplicate_across_months(tmp_path):
    path = initialize(tmp_path, 'Train model', 'project,experiment', source_request='Create a bounded model experiment', month='2026-10')
    assert len(list(path.glob('*.md'))) == 8
    (path / '00-resume.md').write_text('user checkpoint')
    with pytest.raises(FileExistsError, match='Reuse'):
        initialize(tmp_path, 'Train model', 'project', source_request='Repeat', month='2026-11')
    assert (path / '00-resume.md').read_text() == 'user checkpoint'


def test_snapshot_preserves_editable_sources_and_excludes_archives_secrets_and_caches(authored, tmp_path):
    (authored / 'README.md').write_text('workspace source')
    for name in ['.ai', '.venv', 'node_modules', 'sources', 'data', 'weights', 'runs']:
        folder = authored / name
        folder.mkdir()
        (folder / 'large').write_text('must not archive')
    (authored / '.env').write_text('private')
    output = tmp_path.parent / (tmp_path.name + '.tar.gz')
    try:
        snapshot(authored, output)
        with tarfile.open(output) as archive:
            names = archive.getnames()
            assert 'ai/README.md' in names
            assert not any(name.endswith('/large') or name.endswith('/.env') for name in names)
        with pytest.raises(UserError):
            snapshot(authored, output)
    finally:
        output.unlink(missing_ok=True)


def test_scaffold_is_a_real_submodule_and_relink_preserves_external_origins(tmp_path):
    owner = tmp_path / 'ai_projects'
    owner.mkdir()
    subprocess.run(['git', '-C', str(owner), 'init', '-b', 'main'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(owner), 'config', 'user.name', 'Local Test'], check=True)
    subprocess.run(['git', '-C', str(owner), 'config', 'user.email', 'test@example.invalid'], check=True)
    result = new_repository(tmp_path, 'project', 'new-lab', 'New Lab')
    child = Path(result['path'])
    assert (child / '.git').is_file()
    assert Registry(tmp_path).get('new-lab').kind == 'python'
    index = subprocess.run(['git', '-C', str(owner), 'ls-files', '--stage', 'projects/new-lab'], capture_output=True, text=True, check=True).stdout
    assert index.startswith('160000 ')
    with pytest.raises(UserError, match='already exists'):
        new_repository(tmp_path, 'project', 'new-lab', 'Replacement')
    external = 'https://example.org/new-lab.git'
    subprocess.run(['git', '-C', str(child), 'remote', 'set-url', 'origin', external], check=True)
    relink(tmp_path)
    assert subprocess.run(['git', '-C', str(child), 'remote', 'get-url', 'origin'], capture_output=True, text=True, check=True).stdout.strip() == external
