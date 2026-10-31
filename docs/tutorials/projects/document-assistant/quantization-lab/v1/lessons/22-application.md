## 22. Connect a local interface that exposes the evidence and limits

The service boundary can now be exercised without a model. An interface adds another possible source of confusion: a fixed demo answer can look like working inference, stale results can appear current, and a comparison view can show incompatible runs together. We design the application around visible provenance and bounded requests.

### Work backward from one useful screen

The eventual local interface needs a document selector/import status, a question field, the selected artifact and context/output limits, an answer/refusal/error panel and citations that open the exact supplied passage. Show actual backend, generation/fallback identity and whether evidence was dropped. A comparison view presents matched reference/candidate answers with question ID, support outcome and measured resource receipt; a missing run is a pending/failure state, not a zero score.

Loading states should distinguish model initialization from document preparation and active generation. Disable incompatible actions while a request owns the runtime or implement explicit queued request ownership. Cancellation must finish cleanup before allowing a new independent request. On restart, show which saved generation passed verification and which were rejected, as Lesson 20 requires.

### An HTTP boundary we can fully test today

The complete reference below starts a temporary loopback server, submits one request, verifies its JSON response and rejects an invalid request. It accepts only `{"fixture_id":"lumen-charge"}`. It does **not** accept arbitrary document questions and then return the same canned answer. This narrow API makes the demonstration's scope testable.

`Handler.respond()` serializes a JSON body, states its content length/type and writes it. `do_POST()` checks route/body size, parses the request and enforces the named fixture. `main()` binds an operating-system-selected free port, runs the server in a thread, exercises success/error paths, then shuts it down and closes the socket. The final JSON reports `model_executed: false` so a caller cannot reasonably treat the result as Gemma inference.

[Download the complete reference](examples/22_local_app.py)

```python
"""Standalone loopback HTTP boundary demo; responses are explicitly fixtures."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import urllib.error
import urllib.request


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return  # Avoid logging user document text in this reference.

    def respond(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != "/answer":
            self.respond(404, {"error": "unknown route"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Body must contain 1..4096 bytes")
            request = json.loads(self.rfile.read(size))
            if request != {"fixture_id": "lumen-charge"}:
                raise ValueError("Only the named teaching fixture is supported")
        except (ValueError, TypeError):
            self.respond(400, {"error": "invalid fixture request", "backend": "fixture"})
            return
        self.respond(200, {"backend": "fixture", "model_executed": False,
                           "answer": "3 hours", "citations": ["lumen-1"]})


def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/answer"
    try:
        request = urllib.request.Request(url, data=b'{"fixture_id":"lumen-charge"}',
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.load(response)
        bad = urllib.request.Request(url, data=b'{}')
        try:
            urllib.request.urlopen(bad, timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("Invalid fixture request accepted")
        assert result["model_executed"] is False
        print(json.dumps({"http_status": 200, "bad_request_status": 400, "result": result}))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 22_local_app.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "http_status": 200,
  "bad_request_status": 400,
  "result": {
    "backend": "fixture",
    "model_executed": false,
    "answer": "3 hours",
    "citations": [
      "lumen-1"
    ]
  }
}
```

A 200 response alone proves only that a handler returned. We also check the explicit backend flag and the bad-request status. If a bad fixture is accepted, inspect request validation. If the program hangs on exit, inspect `shutdown()`, `server_close()` and thread lifetime. The reference's sequential `HTTPServer` is a local integration probe, not a production server with streaming, user sessions, full timeout handling or an implemented document UI.

### Connect CLI and TUI without inventing enabled actions

The project's current TUI remains useful for status, setup and configuration. During real phase 06, add a project-owned application runner only after the adapter and service pass acceptance checks. Its runner must call the same canonical application entry point as the CLI, with a declared loopback bind and run-record destination. Keep checks separate from serving. A TUI action should show startup failure rather than leave a misleading “running” row after the process exits.

The current `answer`, `prepare-data`, `quantize` and `evaluate` actions still return exit 2. Do not change their labels to imply availability as part of tutorial preparation. The teaching HTTP file can be run/downloaded independently, with fixture identity preserved, while actual application integration remains an implementation milestone.

### Define application verification from real interactions

For the final model-backed interface, select the Lumen-like development document, ask a supported question, open the cited text, then ask an unavailable fact and verify refusal. Change documents and ask again; old evidence must not carry over. Submit an over-budget document and a malformed request; error text should state what failed without a fabricated answer. Cancel generation, start another request and verify cleanup. Restart the application, reload the selected generation and replay the frozen questions.

Exercise desktop/mobile layouts if a browser UI is chosen: long document names, long citations, code/table passages, narrow screens, waiting/error/empty states and comparison columns. These checks belong to the final application, not to this tutorial's docs reader. A rendered tutorial page cannot establish that the future assistant UI works.

[[practice:application-transfer]]

The interface contract is concrete, but candidate choice still needs a final rule and untouched assessment. Next we freeze the development decision and handle the possibility that no compressed candidate meets the declared requirements.
