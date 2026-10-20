"""Embedding: errori del server ritentati dopo aver scaricato gli altri modelli. Nessuna rete."""
import numpy as np
import pytest

from health_agent import embeddings


class R:
    def __init__(self, status=200, data=None, text=""):
        self.status_code, self._d, self.text = status, data or {}, text
        self.ok = status < 400
    def json(self): return self._d


def test_retry_unloads_other_models_then_succeeds(monkeypatch):
    monkeypatch.setattr(embeddings, "RETRY_WAIT", 0)
    posts = []
    answers = iter([R(500, text="model requires more system memory"), R(200, {"embeddings": [[3.0, 4.0]]})])

    def fake_post(url, json=None, timeout=None):
        posts.append((url.rsplit("/", 1)[1], json))
        return next(answers) if url.endswith("/api/embed") else R(200)

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    monkeypatch.setattr(embeddings.requests, "get", lambda url, timeout=None: R(200, {"models": [{"name": "gemma3:12b"}, {"name": "bge-m3:latest"}]}))
    v = embeddings.embed(["ciao"])
    assert np.allclose(v, [[0.6, 0.8]])
    unloads = [p for p in posts if p[0] == "generate"]
    assert unloads == [("generate", {"model": "gemma3:12b", "keep_alive": 0})]    # bge-m3 resta caricato


def test_error_message_carries_status_and_body(monkeypatch):
    monkeypatch.setattr(embeddings, "RETRY_WAIT", 0)
    monkeypatch.setattr(embeddings.requests, "post", lambda url, json=None, timeout=None: R(500, text="boom") if url.endswith("embed") else R(200))
    monkeypatch.setattr(embeddings.requests, "get", lambda url, timeout=None: R(200, {"models": []}))
    with pytest.raises(RuntimeError, match="Ollama 500: boom"):
        embeddings.embed(["x"])
