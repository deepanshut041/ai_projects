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
