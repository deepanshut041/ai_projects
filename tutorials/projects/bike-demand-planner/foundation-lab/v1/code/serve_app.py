"""Run the local historical-replay and benchmark application."""

import json
from pathlib import Path

from bike_demand_planner.application import ReplayApplication, make_server
from bike_demand_planner.artifacts import read_json


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    model_directory = project_root/"models/selected"
    prepared = sorted((project_root/"runs").glob("application-*/application.json"))
    port = 8503
    if not prepared:
        print(json.dumps({"status": "needs_run", "message": "Run application_example.py before starting the replay application."}))
        raise SystemExit(2)
    record = read_json(prepared[-1])
    if record["status"] != "completed":
        print(json.dumps({"status": "needs_run", "message": "The latest application preparation failed; inspect its receipt before retrying."}))
        raise SystemExit(2)
    application = ReplayApplication(model_directory, prepared[-1].parent/"application-data", expected_bundle_sha256=record["bundle_sha256"])
    server = make_server(application, port)
    print(json.dumps({"status": "ready", "url": f"http://127.0.0.1:{server.server_port}/", "scope": "local development validation replay"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
