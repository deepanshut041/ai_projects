"""Loopback-only historical replay with separate prediction and observation paths."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .artifacts import digest, read_json
from .saved_predictor import SavedPredictor


class ReplayApplication:
    def __init__(self, bundle: Path, data_directory: Path, *, expected_bundle_sha256=None):
        self.predictor = SavedPredictor(bundle, expected_manifest_sha256=expected_bundle_sha256)
        metadata = read_json(data_directory / "manifest.json")
        if metadata.get("version") != "hourly-application-data-v1" or metadata["bundle_sha256"] != self.predictor.identity:
            raise ValueError("Application data and predictor identities differ.")
        if set(metadata["artifacts_sha256"]) != {"replay.json", "benchmarks.json"}:
            raise ValueError("Unexpected application data members.")
        for name, identity in metadata["artifacts_sha256"].items():
            if digest((data_directory/name).read_bytes()) != identity:
                raise ValueError("Application evidence differs from its saved identity.")
        self.replay = read_json(data_directory / "replay.json")
        self.benchmarks = read_json(data_directory / "benchmarks.json")
        for artifact in (self.replay, self.benchmarks):
            if artifact["dataset"] != self.predictor.candidate["dataset"] or artifact["bundle_sha256"] != self.predictor.identity:
                raise ValueError("Application dataset or bundle context differs.")
        if (self.replay.get("version") != "hourly-validation-observations-v1"
                or self.replay["test_evaluated"] or self.benchmarks.get("version") != "hourly-replay-benchmarks-v1"):
            raise ValueError("Expected development-only application evidence.")
        for time, observation in self.replay["observations"].items():
            request = self.predictor.request_for_target(time)
            if request["forecast_issue_time"] != observation["forecast_issue_time"]:
                raise ValueError("Observation issue time disagrees with the replay contract.")

    def info(self):
        c = self.predictor.candidate
        return {"title": "Bike Demand Planner", "model_kind": c["model_kind"], "model_sha256": c["model_sha256"],
                "bundle_sha256": self.predictor.identity, "schema_version": c["schema"]["version"],
                "feature_count": len(c["feature_names"]), "used_fields": c["used_fields"], "units": c["units"],
                "input_policy": c["input_policy"], "known_limits": c["known_limits"], "final_test_evaluated": False}

    def observation(self, target_time: str) -> dict:
        self.predictor.request_for_target(target_time)
        found = self.replay["observations"].get(target_time)
        return {"target_time": target_time, "available": found is not None, **(found or {}),
                "scope": "observed afterward; separate from prediction; absent records are not zero rentals"}


def make_server(application: ReplayApplication, port: int = 8503) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            body = json.dumps(payload, allow_nan=False).encode() if isinstance(payload, dict) else payload
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                parsed = urlsplit(self.path)
                if parsed.path in ("/", "/app.js", "/style.css"):
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    kind = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                    return self.send(200, files("bike_demand_planner").joinpath("web", name).read_bytes(), kind+"; charset=utf-8")
                if parsed.path == "/api/info":
                    return self.send(200, application.info())
                if parsed.path == "/api/benchmarks":
                    return self.send(200, application.benchmarks)
                if parsed.path in ("/api/request", "/api/observation"):
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    if set(query) != {"target_time"} or len(query["target_time"]) != 1:
                        raise ValueError("Supply one target_time.")
                    time = query["target_time"][0]
                    result = (application.predictor.request_for_target(time) if parsed.path == "/api/request"
                              else application.observation(time))
                    return self.send(200, result)
                return self.send(404, {"error": "Unknown route."})
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            if self.path != "/api/predict":
                return self.send(404, {"error": "Unknown route."})
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://127.0.0.1:{self.server.server_port}":
                return self.send(403, {"error": "Prediction requests must come from this local application."})
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 16_384:
                    raise ValueError("Expected a small JSON calendar request.")
                request = json.loads(self.rfile.read(count))
                self.send(200, application.predictor.predict(request))
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
