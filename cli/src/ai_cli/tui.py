"""The workspace control room. No document indexing or Markdown browser."""
from __future__ import annotations

import webbrowser
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Select, Static

from .manager import Manager
from .registry import UserError


SECTIONS = (("projects", "Projects"), ("research", "Research"))
TABLE_IDS = {"research": "research-table", "projects": "project-table"}


class ProjectTable(DataTable):
    """Arrow navigation continues across the two workspace sections."""

    def action_cursor_down(self):
        if self.cursor_row >= self.row_count - 1:
            self.app.move_project(1)
        else:
            super().action_cursor_down()

    def action_cursor_up(self):
        if self.cursor_row == 0:
            self.app.move_project(-1)
        else:
            super().action_cursor_up()

    def on_focus(self):
        self.app.table_focused(self)


class AIHome(App[None]):
    TITLE = "AI Home"
    SUB_TITLE = "Sites · projects · ports"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: #101721; }
    Header { background: #152d39; }
    Footer { background: #152d39; }
    #body { height: 1fr; }
    #projects { width: 36%; min-width: 25; border-right: solid #284351; padding: 0 1; }
    #sections { height: 1fr; scrollbar-size: 1 1; }
    .project-table { height: auto; background: #101721; scrollbar-size: 1 1; }
    .section-heading { height: 1; margin-top: 1; text-style: bold; color: #81e0ca; }
    #projects-heading { margin-top: 0; }
    #filter { margin: 0; }
    #overview { height: 1; color: #81b5bf; }
    #detail { width: 1fr; padding: 0 1; }
    #selected-title { height: 1; text-style: bold; color: #81e0ca; }
    #selected-meta { height: 2; color: #aabcca; }
    .controls { height: 3; align: left middle; }
    Button { min-width: 7; width: auto; margin: 0 1 0 0; padding: 0 1; }
    #port { width: 12; margin-right: 1; }
    #save-port { min-width: 7; }
    #stream { width: 1fr; min-width: 12; }
    #url { height: 1; color: #81b5bf; }
    #log { height: 1fr; min-height: 3; border: solid #284351; background: #0b1119; }
    #message { height: 2; color: #e8c68a; }
    #retained { height: 1; color: #6f8c9c; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("f", "filter", "Filter"),
        Binding("s", "start", "Start"), Binding("x", "stop", "Stop"),
        Binding("r", "restart", "Restart", show=False), Binding("b", "build", "Build"),
        Binding("t", "test", "Test"), Binding("o", "open", "Open site", show=False),
        Binding("p", "port", "Port", show=False),
        Binding("escape", "focus_projects", "Projects", show=False),
    ]

    def __init__(self, manager: Manager):
        super().__init__()
        self.manager = manager
        self.selected = next(iter(manager.registry.projects), "")
        self.rows: list[dict] = []
        self.busy = False
        self.category = "service"
        self.last_log = None
        self.last_table = None
        self.port_owner = None
        self.saved_port = ""
        self.section_rows: dict[str, list[dict]] = {section: [] for section, _ in SECTIONS}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="projects"):
                yield Static("Workspace", id="overview")
                yield Input(placeholder="Filter projects…", id="filter")
                with VerticalScroll(id="sections"):
                    for section, title in SECTIONS:
                        yield Static(title, id=f"{section}-heading", classes="section-heading", markup=False)
                        yield ProjectTable(id=TABLE_IDS[section], classes="project-table", cursor_type="row",
                                           zebra_stripes=True, show_header=False, cell_padding=0)
                yield Static("Servers persist on exit", id="retained")
            with Vertical(id="detail"):
                yield Static(id="selected-title", markup=False)
                yield Static(id="selected-meta", markup=False)
                with Horizontal(classes="controls"):
                    yield Button("Start", id="start", variant="success")
                    yield Button("Stop", id="stop")
                    yield Button("Restart", id="restart")
                    yield Button("Open", id="open")
                with Horizontal(classes="controls"):
                    yield Button("Build", id="build")
                    yield Button("Test", id="test", variant="primary")
                    yield Button("Run", id="run")
                    yield Button("Setup", id="setup")
                    yield Button("Cancel", id="cancel")
                with Horizontal(classes="controls"):
                    yield Input(placeholder="Port", id="port", type="integer")
                    yield Button("Save", id="save-port")
                    yield Select([("Server logs", "service"), ("Job logs", "job")],
                                 value="service", allow_blank=False, id="stream")
                yield Static(id="url", markup=False)
                yield RichLog(id="log", wrap=True, min_width=1, markup=False, highlight=False, max_lines=500)
                yield Static("Choose a project. Start a site or run a build/test job.", id="message", markup=False)
        yield Footer()

    def on_mount(self):
        self.configure_columns()
        self.render_state(self.manager.snapshot())
        self.set_interval(1, self.refresh_state)
        self.action_focus_projects()

    def configure_columns(self):
        for table in self.query(ProjectTable):
            table.clear(columns=True)
            table.add_column("Project", width=max(10, int(self.size.width * .36) - 14))
            table.add_column("State", width=8)
        self.last_table = None

    def on_resize(self):
        if self.is_mounted and self.query("#project-table"):
            self.configure_columns()
            self.render_state(self.rows)

    @work(thread=True, exclusive=True, group="refresh", exit_on_error=False)
    def refresh_state(self):
        try:
            self.call_from_thread(self.render_state, self.manager.snapshot())
        except (UserError, OSError) as exc:
            self.call_from_thread(self.message, str(exc))

    def message(self, text: str):
        self.query_one("#message", Static).update(text)

    def section_for(self, row: dict) -> str:
        group = row["group"].lower().replace("’", "'")
        if group in ("research", "projects"):
            return group
        path = Path(row["path"])
        if path.is_relative_to(self.manager.root / "ai_research"):
            return "research"
        return "projects"

    def table_for(self, key: str) -> ProjectTable | None:
        for section, rows in self.section_rows.items():
            if any(row["id"] == key for row in rows):
                return self.query_one(f"#{TABLE_IDS[section]}", ProjectTable)
        return None

    def table_focused(self, table: ProjectTable):
        if table.row_count:
            key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
            if key:
                self.select_project(key)

    def focus_project(self, key: str):
        table = self.table_for(key)
        if table is not None:
            table.move_cursor(row=table.get_row_index(key))
            table.focus()
            table.scroll_visible(animate=False)
            self.select_project(key)

    def move_project(self, direction: int):
        keys = [row["id"] for section, _ in SECTIONS for row in self.section_rows[section]]
        if keys and self.selected in keys:
            index = max(0, min(len(keys) - 1, keys.index(self.selected) + direction))
            self.focus_project(keys[index])

    def render_state(self, rows: list[dict]):
        self.rows = rows
        active = sum(bool(run.get("alive")) for row in rows for run in row["runs"].values())
        self.query_one("#overview", Static).update(f"{len(rows)} projects · {active} active")
        query = self.query_one("#filter", Input).value.lower().strip().replace("'", "").replace("’", "")
        titles = dict(SECTIONS)
        filtered = [row for row in rows if query in
                    f"{row['id']} {row['label']} {row['group']} {titles[self.section_for(row)]}"
                    .lower().replace("'", "").replace("’", "")]
        signature = [(row["id"], row["label"], row["group"], row["path"], row["runs"], row["available"])
                     for row in filtered]
        if signature != self.last_table:
            self.last_table = signature
            self.section_rows = {section: [row for row in filtered if self.section_for(row) == section]
                                 for section, _ in SECTIONS}
            ordered = [row for section, _ in SECTIONS for row in self.section_rows[section]]
            if ordered and self.selected not in {row["id"] for row in ordered}:
                self.select_project(ordered[0]["id"])
            for section, title in SECTIONS:
                section_rows = self.section_rows[section]
                self.query_one(f"#{section}-heading", Static).update(f"{title} ({len(section_rows)})")
                table = self.query_one(f"#{TABLE_IDS[section]}", ProjectTable)
                table.clear()
                table.display = bool(section_rows)
                table.disabled = not section_rows
                table.styles.height = min(7, len(section_rows))
                for row in section_rows:
                    running = [r for r in row["runs"].values() if r.get("alive")]
                    failed = any(r["status"] in ("failed", "stale") for r in row["runs"].values())
                    status = "Running" if running else "Failed" if failed else "Ready" if row["available"] else "Missing"
                    style = "#81e0ca" if running else "#eea1a1" if failed else "#aabcca"
                    badge = Text("│", style="#284351")
                    badge.append(status, style=style)
                    table.add_row(Text(row["label"], overflow="ellipsis", no_wrap=True), badge, key=row["id"])
                table.show_cursor = self.selected in {row["id"] for row in section_rows}
                if self.selected in {row["id"] for row in section_rows}:
                    table.move_cursor(row=table.get_row_index(self.selected))
        self.render_detail()

    def select_project(self, key: str):
        active_table = self.table_for(key)
        for table in self.query(ProjectTable):
            table.show_cursor = table is active_table
        if key != self.selected:
            self.selected = key
            self.last_log = None
            self.query_one("#port", Input).value = str(self.manager.port(key) or "")
            row = next((row for row in self.rows if row["id"] == key), {})
            if row.get("runs", {}).get("job") and "start" not in row.get("actions", ()):
                self.query_one("#stream", Select).value = "job"
            self.render_detail()

    def render_detail(self):
        row = next((row for row in self.rows if row["id"] == self.selected), None)
        if not row:
            return
        self.query_one("#selected-title", Static).update(row["label"])
        service = row["runs"].get("service", {})
        job = row["runs"].get("job", {})
        server_text = service.get("status", "stopped") if "start" in row["actions"] else "no site"
        job_text = f"{job.get('action', 'job')}: {job.get('status', 'idle')}"
        if job.get("returncode") is not None:
            job_text += f" (exit {job['returncode']})"
        self.query_one("#selected-meta", Static).update(f"{row['group']} · {server_text} · {job_text}\n{row['id']}")
        self.query_one("#url", Static).update(row["url"] or "Native project · build and test output below")
        alive_service = bool(service.get("alive"))
        alive_job = bool(job.get("alive"))
        port = self.query_one("#port", Input)
        saved = str(row["port"] or "")
        if alive_service or self.port_owner != self.selected or port.value == self.saved_port:
            port.value = saved
        self.port_owner = self.selected
        self.saved_port = saved
        for action in ("start", "build", "test", "run"):
            self.query_one(f"#{action}", Button).disabled = (
                self.busy or not row["available"] or action not in row["actions"]
                or (alive_service or alive_job if action == "start" else alive_job)
                or (alive_service and action in ("build", "run")))
        self.query_one("#stop", Button).disabled = self.busy or not alive_service
        self.query_one("#restart", Button).disabled = self.busy or alive_job or "start" not in row["actions"] or not row["available"]
        self.query_one("#open", Button).disabled = not bool(row["url"])
        self.query_one("#cancel", Button).disabled = self.busy or not alive_job
        project = self.manager.registry.get(self.selected)
        setup_supported = row["kind"] == "node" or "setup" in project.overrides or (project.path / "uv.lock").is_file()
        self.query_one("#setup", Button).disabled = self.busy or alive_job or alive_service or not row["available"] or not setup_supported
        self.query_one("#save-port", Button).disabled = self.busy or alive_service or "start" not in row["actions"]
        port.disabled = self.busy or alive_service or "start" not in row["actions"]
        try:
            content = self.manager.logs(self.selected, self.category)
        except (UserError, OSError) as exc:
            content = str(exc)
        if (self.selected, self.category, content) != self.last_log:
            self.last_log = (self.selected, self.category, content)
            log = self.query_one("#log", RichLog)
            log.clear()
            log.write(Text.from_ansi(content))

    @on(DataTable.RowHighlighted, ".project-table")
    def highlight_project(self, event: DataTable.RowHighlighted):
        table = event.data_table
        # Ignore queued cursor messages from another section or before a refresh.
        if (table.has_focus and table.row_count and event.row_key.value
                and event.row_key == table.coordinate_to_cell_key(table.cursor_coordinate).row_key):
            self.select_project(event.row_key.value)

    @on(Input.Changed, "#filter")
    def filter_changed(self):
        self.last_table = None
        self.render_state(self.rows)

    @on(Select.Changed, "#stream")
    def stream_changed(self, event: Select.Changed):
        if event.value in ("service", "job"):
            self.category = event.value
            self.last_log = None
            self.render_detail()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed):
        action = event.button.id
        if action == "open":
            self.action_open()
        elif action == "save-port":
            value = self.query_one("#port", Input).value
            try:
                self.manager.set_port(self.selected, int(value))
                self.message(f"Saved port {value}. Start the site when ready.")
                self.refresh_state()
            except (ValueError, UserError) as exc:
                self.message(str(exc) if isinstance(exc, UserError) else "Enter a port between 1024 and 65535.")
        elif action:
            self.begin(action)

    def begin(self, action: str):
        button = self.query_one(f"#{action}", Button)
        if self.busy or button.disabled:
            return
        self.busy = True
        key = self.selected
        if action in ("build", "test", "run", "setup", "cancel"):
            self.query_one("#stream", Select).value = "job"
        else:
            self.query_one("#stream", Select).value = "service"
        self.message(f"{action.title()} · {self.manager.registry.get(key).label}…")
        self.render_detail()
        self.control(key, action)

    @work(thread=True, group="control", exit_on_error=False)
    def control(self, key: str, action: str):
        try:
            if action in ("stop", "cancel"):
                self.manager.stop(key, "job" if action == "cancel" else "service")
                text = f"{key}: {'job cancelled' if action == 'cancel' else 'server stopped'}."
            else:
                if action == "restart":
                    self.manager.stop(key)
                self.manager.launch(key, "start" if action == "restart" else action)
                if action in ("start", "restart"):
                    record = self.manager.wait_ready(key)
                    text = f"Ready · {record['url'] or key}"
                else:
                    text = f"{action.title()} running. Logs and exit status update here."
            self.call_from_thread(self.finished_control, text)
        except (UserError, OSError) as exc:
            self.call_from_thread(self.finished_control, str(exc))

    def finished_control(self, text: str):
        self.busy = False
        self.message(text)
        self.render_state(self.manager.snapshot())

    def action_filter(self):
        self.query_one("#filter", Input).focus()

    def action_focus_projects(self):
        if self.table_for(self.selected) is not None:
            self.focus_project(self.selected)
        else:
            self.action_filter()

    def action_port(self):
        self.query_one("#port", Input).focus()

    def action_open(self):
        project = self.manager.registry.get(self.selected)
        url = project.url(self.manager.port(self.selected))
        if url:
            webbrowser.open(url)
            self.message(f"Opened {url}")

    def action_start(self): self.begin("start")
    def action_stop(self): self.begin("stop")
    def action_restart(self): self.begin("restart")
    def action_build(self): self.begin("build")
    def action_test(self): self.begin("test")
