import json
from pathlib import Path
import socket
import sys

import pytest

from ai_cli.manager import Manager


def free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


@pytest.fixture
def workspace(tmp_path: Path):
    notes = tmp_path / "ai_projects/project_docs"
    notes.mkdir(parents=True)
    (tmp_path / "ai_projects/.gitmodules").write_text('[submodule "docs"]\npath = project_docs\n')
    (notes / "package.json").write_text(json.dumps({"scripts": {"dev": "fake", "test": "fake", "build": "fake"}}))
    server = notes / "server.py"
    server.write_text('''import os
from http.server import HTTPServer, BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ai fixture")
HTTPServer.allow_reuse_address = True
HTTPServer(("127.0.0.1", int(os.environ["PORT"])), Handler).serve_forever()
''')
    job = notes / "job.py"
    job.write_text('''import sys, time
print("job running", flush=True)
time.sleep(.25)
print("job completed", flush=True)
sys.exit(int(sys.argv[1]))
''')
    def argv(*items):
        return json.dumps(list(items))
    (tmp_path / "ai-workspace.toml").write_text(f'''[projects.project-docs]
path = "ai_projects/project_docs"
group = "Projects"
start = {argv(sys.executable, "-u", str(server))}
build = {argv(sys.executable, "-u", str(job), "0")}
test = {argv(sys.executable, "-u", str(job), "7")}
port = {free_port()}
''')
    return tmp_path


@pytest.fixture
def manager(workspace):
    controller = Manager(workspace)
    yield controller
    controller.stop_all()
    controller.reap()
