"""gpu_instances backend against a fake local server (pod and gateway modes)."""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from souvenir.backends.gpu_instances import GpuInstancesBackend


def _png(color=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, format="PNG")
    return buf.getvalue()


class Fake(BaseHTTPRequestHandler):
    seen: list = []

    def _json(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        assert self.headers["Authorization"] == "Bearer k1"
        body = json.loads(self._body() or b"{}")
        Fake.seen.append((self.path, body))
        if self.path == "/v1/images/edit":
            return self._json(200, {"images": [base64.b64encode(_png()).decode()], "seed": body.get("seed")})
        if self.path == "/v1/uploads":
            return self._json(200, {"url": base + "/put/1", "headers": {}, "ref": "upload:upl_1"})
        if self.path.startswith("/v1/jobs"):
            return self._json(202, {"id": "job_1", "status": "queued"})
        self._json(404, {})

    def do_PUT(self):
        self._body()
        self._json(200, {})

    def do_GET(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        if self.path == "/v1/jobs/job_1":
            return self._json(200, {"id": "job_1", "status": "done", "output": {"images": [{"url": base + "/out.png", "content_type": "image/png"}]}})
        if self.path == "/out.png":
            raw = _png((200, 0, 0))
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self._json(404, {})

    def log_message(self, *a):
        pass


def _server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_pod_mode_reads_state_json(tmp_path: Path):
    srv = _server()
    (tmp_path / ".state.json").write_text(json.dumps({"pod_id": "p1", "url": f"http://127.0.0.1:{srv.server_port}", "server_api_key": "k1"}))
    b = GpuInstancesBackend(mode="pod", project_dir=str(tmp_path))
    out = b.edit([Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))], "make it blue", seed=5, width=32, height=24)
    assert out.size == (32, 24)
    path, body = Fake.seen[-1]
    assert path == "/v1/images/edit" and len(body["images"]) == 2 and body["num_inference_steps"] == 4
    assert body["images"][0].startswith("data:image/png;base64,")
    srv.shutdown()


def test_gateway_mode_uploads_submits_and_downloads(tmp_path: Path):
    srv = _server()
    (tmp_path / ".env").write_text(f"PROD_URL=http://127.0.0.1:{srv.server_port}\nPROD_API_KEY=k1\n")
    b = GpuInstancesBackend(mode="gateway", project_dir=str(tmp_path), poll_s=0.01)
    out = b.edit([Image.new("RGB", (64, 64))], "make it blue", seed=1)
    assert out.getpixel((0, 0)) == (200, 0, 0)
    job = [body for path, body in Fake.seen if path.startswith("/v1/jobs")][-1]
    assert job["model"] == "qwen-image-edit-2511" and job["input"]["images"] == ["upload:upl_1"]
    srv.shutdown()
