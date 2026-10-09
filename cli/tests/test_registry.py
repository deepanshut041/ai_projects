import json
from pathlib import Path
import sys

import pytest

from ai_cli.registry import Registry, UserError


def test_discovers_new_research_submodule_and_missing_project(workspace):
    owner = workspace / "ai_research"
    repo = owner / "new_experiment"
    repo.mkdir(parents=True)
    (owner / ".gitmodules").write_text('''[submodule "new"]
path = new_experiment
url = https://example.invalid/new.git
[submodule "docs"]
path = research_docs
url = https://example.invalid/docs.git
''')
    (repo / "package.json").write_text(json.dumps({"scripts": {"dev": "node app.js", "test": "node test.js"}}))
    registry = Registry(workspace)
    project = registry.get("new-experiment")
    assert project.group == "Research"
    assert project.port == 4380
    assert project.actions == ("setup", "start", "test")
    assert not registry.get("research-docs").available
    with pytest.raises(UserError, match="not checked out"):
        registry.plan("research-docs", "start")


@pytest.mark.parametrize("path", ["../outside", "ai_research/research_docs/sources/raw"])
def test_rejects_outside_and_archive_projects(workspace, path):
    (workspace / "ai-workspace.toml").write_text(f'[projects.bad]\npath = "{path}"\n')
    with pytest.raises(UserError):
        Registry(workspace)


def test_rejects_symlink_escaping_workspace(workspace, tmp_path):
    (workspace / "escape").symlink_to("/tmp")
    (workspace / "ai-workspace.toml").write_text('[projects.escape]\npath = "escape"\n')
    with pytest.raises(UserError, match="escapes"):
        Registry(workspace)


def test_shell_command_override_must_be_argv(workspace):
    (workspace / "ai-workspace.toml").write_text('[projects.project-docs]\nstart = "echo bad"\n')
    with pytest.raises(UserError, match="argv array"):
        Registry(workspace)


def test_argv_placeholders_are_expanded_without_shell(workspace):
    (workspace / "ai-workspace.toml").write_text('''[projects.project-docs]
path = "ai_projects/project_docs"
start = ["server", "--port", "{port}", "{project}", "{workspace}"]
''')
    step = Registry(workspace).plan("project-docs", "start", 4310)[0]
    assert step.argv == ("server", "--port", "4310", str(workspace / "ai_projects/project_docs"), str(workspace))


def test_research_name_collision_gets_a_distinct_group_id(workspace):
    owner = workspace / "ai_research"
    (owner / "project-docs").mkdir(parents=True)
    (owner / ".gitmodules").write_text('[submodule "project-docs"]\npath = project-docs\n')
    projects = Registry(workspace).projects
    assert "project-docs" in projects and "research-project-docs" in projects
    assert projects["research-project-docs"].path == owner / "project-docs"


def test_cmake_plans_use_isolated_build_and_python_environment(workspace):
    owner = workspace / "ai_projects"
    repo = owner / "projects" / "demo"
    repo.mkdir(parents=True)
    (owner / ".gitmodules").write_text('[submodule "demo"]\npath = projects/demo\n')
    (repo / "CMakeLists.txt").write_text('option(DEMO_BUILD_PYTHON "bindings" ON)\n')
    (repo / "uv.lock").write_text("version = 1\n")
    (repo / "python").mkdir()
    (repo / "python/test_bindings.py").write_text("def test_binding(): pass\n")
    plan = Registry(workspace).plan("demo", "test")
    assert plan[0].env["UV_PROJECT_ENVIRONMENT"] == str(workspace / ".ai/envs/demo")
    configure = plan[1].argv
    assert str(workspace / ".ai/builds/demo") in configure
    assert any(arg.startswith("-DPython_EXECUTABLE=") for arg in configure)
    assert "ai_cli.native_tests" in plan[-2].argv
    assert plan[-1].env["PYTHONPATH"] == str(workspace / ".ai/builds/demo")
    assert (repo / "python/test_bindings.py").as_posix() in plan[-1].argv
