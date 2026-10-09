from pathlib import Path
import json
import asyncio

import pytest
from textual.widgets import DataTable, Input

from ai_cli.registry import Registry, UserError
from ai_cli.manager import Manager
from ai_cli.tui import AIHome


def test_python_actions_use_locked_isolated_environments(workspace):
    repo = workspace / 'ai_research/experiment'
    (repo / 'tests').mkdir(parents=True)
    (repo / 'pyproject.toml').write_text('[project]\nname="experiment"\nversion="0.1.0"\n')
    (repo / 'uv.lock').write_text('version=1\n')
    (repo / 'demo.py').write_text('print("local demo")')
    (workspace / 'ai_research/.gitmodules').write_text('[submodule "experiment"]\npath=experiment\n')
    registry = Registry(workspace)
    assert set(registry.get('experiment').actions) == {'setup', 'build', 'test', 'run'}
    test = registry.plan('experiment', 'test')
    assert '--frozen' in test[0].argv
    assert test[0].env['UV_PROJECT_ENVIRONMENT'] == str(workspace / '.ai/envs/experiment')
    assert test[-1].argv[-3:] == ('pytest', '-q', 'tests')
    assert str(workspace / '.ai/builds/experiment') in registry.plan('experiment', 'build')[-1].argv
    (repo / 'uv.lock').unlink()
    with pytest.raises(UserError, match='uv.lock'):
        registry.plan('experiment', 'test')


def test_workspace_has_only_two_groups_and_malformed_metadata_fails(workspace):
    with (workspace / 'ai-workspace.toml').open('a') as stream:
        stream.write('\n[projects.invalid]\npath="ai_projects"\ngroup="Masters"\n')
    with pytest.raises(UserError, match='Projects or Research'):
        Registry(workspace)
    (workspace / 'ai-workspace.toml').unlink()
    (workspace / 'ai_projects/.gitmodules').write_text('invalid metadata without section')
    with pytest.raises(UserError, match='Invalid'):
        Registry(workspace)


def test_two_section_navigation_cursor_filter_refresh_and_small_terminal(workspace):
    projects = workspace / 'ai_projects'
    modules = ['[submodule "docs"]\npath=project_docs\n']
    for index in range(10):
        repo = projects / 'projects' / f'code-{index}'
        repo.mkdir(parents=True)
        (repo / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
        modules.append(f'[submodule "code-{index}"]\npath=projects/code-{index}\n')
    (projects / '.gitmodules').write_text('\n'.join(modules))
    research = workspace / 'ai_research'
    (research / 'research_docs').mkdir(parents=True)
    (research / '.gitmodules').write_text('[submodule "docs"]\npath=research_docs\n')
    (research / 'research_docs/package.json').write_text(json.dumps({'scripts': {'dev': 'fixture'}}))
    manager = Manager(workspace)
    async def scenario():
        app = AIHome(manager)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert len(app.query('.section-heading')) == 2
            assert app.query_one('#project-table', DataTable).row_count == 11
            assert app.query_one('#research-table', DataTable).row_count == 1
            await pilot.press(*(['down'] * 11))
            assert app.selected == 'research-docs'
            assert app.query_one('#research-table', DataTable).show_cursor
            assert not app.query_one('#project-table', DataTable).show_cursor
            app.last_table = None
            app.render_state(manager.snapshot())
            await pilot.pause()
            assert app.selected == 'research-docs'
            await pilot.press('up')
            assert app.selected == 'code-9'
            app.query_one('#filter', Input).value = 'research'
            await pilot.pause()
            assert app.selected == 'research-docs'
            assert app.query_one('#project-table', DataTable).row_count == 0
            assert app.query_one('#research-table', DataTable).row_count == 1
            await pilot.resize_terminal(120, 36)
            await pilot.pause()
            assert app.selected == 'research-docs'
    try:
        asyncio.run(scenario())
    finally:
        manager.stop_all()
