import asyncio
import time
import socket
import json
import sys

from textual.widgets import Button, DataTable, Input, RichLog, Select, Static

from ai_cli.tui import AIHome
from ai_cli.manager import Manager


async def eventually(pilot, predicate, seconds=6):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await pilot.pause(.1)
        if predicate():
            return
    raise AssertionError("UI condition did not become true")


def test_tui_real_service_job_controls_logs_and_filter(manager):
    async def scenario():
        app = AIHome(manager)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            assert app.query_one("#project-table", DataTable).row_count == 1
            await pilot.click("#start")
            await eventually(pilot, lambda: not app.busy and manager.ready(manager.record("project-docs") or {}))
            assert app.query_one("#stop", Button).disabled is False
            assert app.query_one("#port", Input).disabled is True
            assert app.query_one("#log", RichLog).lines
            await pilot.click("#stop")
            await eventually(pilot, lambda: not app.busy and not manager.record("project-docs").get("status") == "running")
            await pilot.click("#build")
            await eventually(pilot, lambda: manager.record("project-docs", "job") and manager.record("project-docs", "job")["status"] == "finished")
            await eventually(pilot, lambda: "exit 0" in str(app.query_one("#selected-meta", Static).render()))
            assert app.query_one("#stream", Select).value == "job"
            await pilot.click("#test")
            await eventually(pilot, lambda: "exit 7" in str(app.query_one("#selected-meta", Static).render()))
            await pilot.press("f")
            await pilot.press(*"missing")
            assert app.query_one("#project-table", DataTable).row_count == 0
            await pilot.press("ctrl+u", "escape")
            assert app.query_one("#project-table", DataTable).row_count == 1
    asyncio.run(scenario())


def test_compact_tui_port_edit_and_service_survives_ui_exit(manager):
    async def scenario():
        app = AIHome(manager)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for selector in ("#start", "#stop", "#restart", "#open", "#build", "#test", "#run", "#setup", "#cancel", "#save-port", "#log"):
                widget = app.query_one(selector)
                assert widget.region.width > 0 and widget.region.height > 0
                assert widget.region.right <= 80
                assert widget.region.bottom <= 24
            port = app.query_one("#port", Input)
            port.value = "0"
            await pilot.click("#save-port")
            assert "1024" in str(app.query_one("#message", Static).render())
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                new_port = probe.getsockname()[1]
            port.value = str(new_port)
            await pilot.click("#filter")
            app.render_state(manager.snapshot())
            assert port.value == str(new_port)  # A status poll must not discard unsaved edits.
            await pilot.click("#save-port")
            assert manager.port("project-docs") == new_port
            await pilot.resize_terminal(120, 36)
            await pilot.pause()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.query_one("#project-table", DataTable).row_count == 1
            await pilot.click("#start")
            await eventually(pilot, lambda: not app.busy and manager.ready(manager.record("project-docs") or {}))
            assert manager.record("project-docs")["port"] == new_port
        assert manager.ready(manager.record("project-docs"))
    asyncio.run(scenario())


def test_switching_projects_changes_actions_and_routes_job_to_selected_repo(workspace):
    repo = workspace / "ai_research" / "another-project"
    repo.mkdir(parents=True)
    script = repo / "job.py"
    script.write_text("from pathlib import Path\nprint('selected repo: ' + Path.cwd().name, flush=True)\n")
    config = workspace / "ai-workspace.toml"
    with config.open("a") as stream:
        stream.write('\n[projects.another-project]\npath = "ai_research/another-project"\n')
        stream.write('group = "Research"\n')
        stream.write('build = ' + json.dumps([sys.executable, str(script)]) + '\n')
    controller = Manager(workspace)
    async def scenario():
        app = AIHome(controller)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("down")
            assert app.selected == "another-project"
            assert app.query_one("#project-table", DataTable).row_count == 1
            assert app.query_one("#research-table", DataTable).row_count == 1
            assert app.query_one("#project-table", DataTable).row_count == 1
            assert app.query_one("#start", Button).disabled
            assert app.query_one("#port", Input).disabled
            assert not app.query_one("#build", Button).disabled
            await pilot.press("b")
            await eventually(pilot, lambda: "exit 0" in str(app.query_one("#selected-meta", Static).render()))
            assert "selected repo: another-project" in controller.logs("another-project", "job")
            assert controller.record("project-docs", "job") is None
            app.render_state(controller.snapshot())
            await pilot.pause()
            assert app.selected == "another-project"  # Refresh keeps the selected section.
            await pilot.press("up")
            assert app.selected == "project-docs"
            assert not app.query_one("#start", Button).disabled
    try:
        asyncio.run(scenario())
    finally:
        controller.stop_all()

