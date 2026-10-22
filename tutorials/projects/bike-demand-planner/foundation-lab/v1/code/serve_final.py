"""Serve the fixed final assessment and historical replay locally."""

import json
from pathlib import Path

from bike_demand_planner.artifacts import read_json
from bike_demand_planner.final_application import FinalApplication, make_final_server


if __name__ == "__main__":
    project=Path(__file__).resolve().parent
    records=sorted((project/"runs").glob("final-*/assessment.json"))
    prepared=sorted((project/"runs").glob("application-*/application.json"))
    if not records or not prepared or read_json(records[-1])["status"]!="completed" or read_json(prepared[-1])["status"]!="completed":
        print(json.dumps({"status":"needs_run","message":"Complete application_example.py and final_evaluation_example.py; inspect failed receipts before retrying."}))
        raise SystemExit(2)
    app=FinalApplication(project/"models/selected",prepared[-1].parent/"application-data",records[-1],project/"models/final-assessment/protocol.json")
    server=make_final_server(app)
    print(json.dumps({"status":"ready","url":f"http://127.0.0.1:{server.server_port}/","scope":"local final assessment and validation replay"}),flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
