import hashlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from housemate.core import models
from housemate.core.models import (
    MODELS,
    ModelChecksumError,
    ModelDownloadError,
    ModelSpec,
    ensure_model,
    resolve_wake_head,
)

PAYLOAD = b"not really an onnx model\n" * 100


@pytest.fixture
def server():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            if self.path == "/missing":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", hits
    httpd.shutdown()
    httpd.server_close()


def spec(base: str, sha: str, path: str = "/model.onnx") -> dict[str, ModelSpec]:
    return {"tiny": ModelSpec("tiny", "tiny.onnx", base + path, sha, len(PAYLOAD), "test")}


def test_ensure_model_checksum_mismatch_raises(server, tmp_path):
    base, hits = server
    reg = spec(base, "0" * 64)
    with pytest.raises(ModelChecksumError):
        ensure_model("tiny", tmp_path, registry=reg)
    assert hits == ["/model.onnx"]
    # nothing half-written is left behind, and the target was never created
    assert list((tmp_path / "models").iterdir()) == []
    # no retry storm: an immediate second call fails fast without touching the network
    with pytest.raises(ModelDownloadError):
        ensure_model("tiny", tmp_path, registry=reg)
    assert len(hits) == 1


def test_ensure_model_downloads_once_and_verifies(server, tmp_path):
    base, hits = server
    reg = spec(base, hashlib.sha256(PAYLOAD).hexdigest())
    path = ensure_model("tiny", tmp_path, registry=reg)
    assert path == tmp_path / "models" / "tiny.onnx"
    assert path.read_bytes() == PAYLOAD
    assert ensure_model("tiny", tmp_path, registry=reg) == path
    assert len(hits) == 1  # cached copy verified, not downloaded again
    # a corrupted cached copy is replaced by a fresh verified download
    path.write_bytes(b"tampered")
    assert ensure_model("tiny", tmp_path, registry=reg).read_bytes() == PAYLOAD
    assert len(hits) == 2


def test_ensure_model_network_failure_raises_download_error(server, tmp_path):
    base, hits = server
    reg = spec(base, hashlib.sha256(PAYLOAD).hexdigest(), path="/missing")
    with pytest.raises(ModelDownloadError):
        ensure_model("tiny", tmp_path, registry=reg)
    assert list((tmp_path / "models").iterdir()) == []


def test_unknown_model_name(tmp_path):
    with pytest.raises(ValueError, match="hey_jarvis"):
        ensure_model("ok_google", tmp_path)


def test_manifest_pins_every_model():
    assert set(MODELS) == {"melspectrogram", "embedding", "hey_jarvis", "silero_vad"}
    for m in MODELS.values():
        assert m.url.startswith("https://")
        assert len(m.sha256) == 64 and int(m.sha256, 16) >= 0
        assert m.size > 0 and m.licence
    assert "NC" in MODELS["hey_jarvis"].licence  # openWakeWord models are non-commercial


def test_wake_model_can_be_a_custom_head(tmp_path):
    head = tmp_path / "hey_hermes.onnx"
    head.write_bytes(b"x")
    assert resolve_wake_head(str(head), tmp_path) == head
    assert resolve_wake_head("hey_hermes", tmp_path, bundled_dir=tmp_path) == head
    with pytest.raises(FileNotFoundError):
        resolve_wake_head("hey_nobody", tmp_path, bundled_dir=tmp_path)


def test_no_model_files_in_the_repo():
    root = Path(models.__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "*.onnx", "*.tflite"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.split()
    assert [p for p in tracked if not p.startswith("wakeword/")] == []


def test_importing_the_pipeline_does_not_load_onnxruntime():
    root = Path(models.__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import housemate.core.listener, housemate.core.wake, housemate.core.vad, housemate.core.models;"
        "print('onnxruntime' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
