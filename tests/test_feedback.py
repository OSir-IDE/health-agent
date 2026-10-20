"""Test del feedback dai pulsanti: nessuna rete, database temporaneo."""
from datetime import date

import pytest

from health_agent import feedback, telegram_logs


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    monkeypatch.setenv("USER_SUGGESTIONS_PATH", str(tmp_path / "user.json"))


def _news(day="2026-10-09"):
    feedback.record_sent(day, "news", [
        {"idx": 1, "label": "Top", "section": "top", "topic": "econ", "weight": 4},
        {"idx": 2, "label": "AI", "section": "personal", "topic": "ai"},
        {"idx": 3, "label": "Run", "section": "personal", "topic": "run"}], msg_id=77)


def test_keyboard_layout_and_callback_size():
    kb = feedback.keyboard("news", "2026-10-09", 5, {2: -1})
    assert [len(r) for r in kb["inline_keyboard"]] == [4, 4, 2]
    flat = [b for r in kb["inline_keyboard"] for b in r]
    assert all(len(b["callback_data"].encode()) <= 64 for b in flat)
    assert next(b for b in flat if b["callback_data"].endswith("|2|-1"))["text"].startswith("✓")


def test_vote_replaces_previous_and_ignores_unknown():
    _news()
    assert feedback.handle_callback("n|2026-10-09|2|1") == ("news", "2026-10-09", 2, 1)
    feedback.handle_callback("n|2026-10-09|2|-1")
    assert feedback.votes_for("2026-10-09", "news") == {2: -1}
    assert feedback.handle_callback("n|2026-10-09|9|1") is None       # indice mai inviato
    assert feedback.handle_callback("boh") is None


def test_topic_factors_only_from_personal_and_bounded():
    _news()
    for i in (1, 2, 3):
        feedback.handle_callback(f"n|2026-10-09|{i}|{1 if i == 2 else -1}")
    f = feedback.topic_factors(date(2026, 10, 9))
    assert f["ai"] == pytest.approx(1.15) and f["run"] == pytest.approx(0.85)
    assert "econ" not in f                                            # il voto sul primo piano non conta
    topics = feedback.adjusted_topics([{"id": "ai", "weight": 3}, {"id": "w", "weight": 1}], date(2026, 10, 9))
    assert topics[0]["weight"] == pytest.approx(3.45) and topics[1]["weight"] == 1


def test_votes_decay_with_age():
    _news("2026-09-09")
    feedback.handle_callback("n|2026-09-09|2|1")
    assert feedback.topic_factors(date(2026, 10, 9))["ai"] == pytest.approx(1 + 0.15 * 0.5 ** (30 / 30))


def test_adherence():
    feedback.record_sent("2026-10-09", "rec", [{"idx": 1, "label": "a"}, {"idx": 2, "label": "b"}], 5)
    feedback.handle_callback("r|2026-10-09|1|1")
    feedback.handle_callback("r|2026-10-09|2|-1")
    assert feedback.adherence("2026-10-05", "2026-10-11") == {"sent": 2, "done": 1, "skipped": 1}


def test_process_updates_registers_votes_commands_and_refreshes(monkeypatch):
    _news()
    calls = []

    class R:
        status_code = 200
        text = ""
        def __init__(self, data=None): self.data = data
        def raise_for_status(self): pass
        def json(self): return {"result": self.data}

    updates = [
        {"update_id": 1, "callback_query": {"data": "n|2026-10-09|2|1", "message": {"chat": {"id": 5}}}},
        {"update_id": 2, "callback_query": {"data": "n|2026-10-09|3|1", "message": {"chat": {"id": 999}}}},  # chat estranea
        {"update_id": 3, "message": {"chat": {"id": 5}, "text": "#film Perfect Days (2023) | Wenders"}},
        {"update_id": 4, "message": {"chat": {"id": 5}, "text": "ciao"}},
    ]

    def fake_get(url, params=None, timeout=None):
        calls.append(("get", params))
        return R(updates if "offset" not in (params or {}) else [])

    def fake_post(url, json=None, timeout=None):
        calls.append(("post", url.rsplit("/", 1)[1], json))
        return R()

    monkeypatch.setattr(telegram_logs.requests, "get", fake_get)
    monkeypatch.setattr(telegram_logs.requests, "post", fake_post)
    res = telegram_logs.process_updates("T", "5")
    assert res["votes"] == [("news", "2026-10-09", 2, 1)]
    assert res["added"] == [("film", "Perfect Days", "added")]
    assert feedback.votes_for("2026-10-09", "news") == {2: 1}          # il voto della chat estranea è ignorato
    assert ("get", {"offset": 5, "timeout": 0}) in calls                # confermati tutti gli aggiornamenti
    edit = next(c for c in calls if c[0] == "post")
    assert edit[1] == "editMessageReplyMarkup" and edit[2]["message_id"] == 77


def test_buttons_off_by_default_and_switches(monkeypatch):
    monkeypatch.delenv("FEEDBACK_BUTTONS", raising=False)
    monkeypatch.delenv("FEEDBACK_REC_BUTTONS", raising=False)
    assert not feedback.enabled("news") and not feedback.enabled("rec")
    monkeypatch.setenv("FEEDBACK_BUTTONS", "true")
    assert feedback.enabled("news") and not feedback.enabled("rec")
    monkeypatch.setenv("FEEDBACK_REC_BUTTONS", "true")
    assert feedback.enabled("rec")


def test_notifier_no_keyboard_by_default(monkeypatch):
    from health_agent.nodes import notifier
    monkeypatch.delenv("FEEDBACK_BUTTONS", raising=False)
    sent = {}
    monkeypatch.setattr(notifier, "send_message",
                        lambda msg, html=False, reply_markup=None: sent.update(markup=reply_markup) or 42)
    notifier.notifier_node({"run_date": "2026-10-09", "dry_run": False, "recommendations": ["uno"], "errors": []})
    assert sent["markup"] is None


def test_notifier_attaches_keyboard_and_records(monkeypatch):
    from health_agent.nodes import notifier
    monkeypatch.setenv("FEEDBACK_BUTTONS", "true")
    monkeypatch.setenv("FEEDBACK_REC_BUTTONS", "true")
    sent = {}
    monkeypatch.setattr(notifier, "send_message",
                        lambda msg, html=False, reply_markup=None: sent.update(markup=reply_markup) or 42)
    state = {"run_date": "2026-10-09", "dry_run": False, "recommendations": ["uno", "due"], "errors": []}
    out = notifier.notifier_node(state)
    assert not out["errors"] and len(sent["markup"]["inline_keyboard"]) == 2
    assert feedback.stored_keyboard("2026-10-09", "rec")[0] == 42


def test_notifier_send_failure_keeps_telegram_prefix(monkeypatch):
    from health_agent.nodes import notifier
    monkeypatch.setenv("FEEDBACK_BUTTONS", "true")
    monkeypatch.setenv("FEEDBACK_REC_BUTTONS", "true")
    def boom(*a, **k): raise RuntimeError("rete")
    monkeypatch.setattr(notifier, "send_message", boom)
    out = notifier.notifier_node({"run_date": "2026-10-09", "dry_run": False, "recommendations": ["x"], "errors": []})
    assert out["errors"][0].startswith("telegram")
