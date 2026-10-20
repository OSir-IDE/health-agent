"""Guardia oraria del task ripetuto: niente report prima delle 7."""
from datetime import datetime

import main


def test_too_early_default_and_override(monkeypatch):
    monkeypatch.delenv("REPORT_NOT_BEFORE", raising=False)
    assert main.too_early(datetime(2026, 10, 12, 3, 0))
    assert main.too_early(datetime(2026, 10, 12, 6, 59))
    assert not main.too_early(datetime(2026, 10, 12, 7, 0))
    monkeypatch.setenv("REPORT_NOT_BEFORE", "9")
    assert main.too_early(datetime(2026, 10, 12, 8, 30)) and not main.too_early(datetime(2026, 10, 12, 9, 0))


def test_heartbeat_appends_and_trims(tmp_path, monkeypatch):
    import runlog
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "hb.txt"))
    for i in range(7):
        runlog.beat(f"esito{i}", keep=5)
    lines = (tmp_path / "hb.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and lines[-1].endswith("esito6") and lines[0].endswith("esito2")


def test_wait_for_ollama_retries_then_answers():
    from health_agent import llm
    t = {"now": 0.0}
    calls = {"n": 0}

    class R:
        status_code = 200

    def get(url, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("rifiutata")
        return R()

    def sleep(s):
        t["now"] += s

    waited = llm.wait_for_ollama(max_wait=60, interval=5, _get=get, _sleep=sleep, _clock=lambda: t["now"])
    assert waited == 10 and calls["n"] == 3


def test_wait_for_ollama_gives_up():
    from health_agent import llm
    t = {"now": 0.0}

    def get(url, timeout):
        raise ConnectionError("rifiutata")

    def sleep(s):
        t["now"] += s

    assert llm.wait_for_ollama(max_wait=20, interval=5, _get=get, _sleep=sleep, _clock=lambda: t["now"]) is None
