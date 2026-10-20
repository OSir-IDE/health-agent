"""Prova end-to-end del grafo in dry-run con Garmin finto: nessuna rete, nessun invio."""
from datetime import date, timedelta

import pandas as pd

from health_agent.graph import build_graph
from health_agent.nodes import extras, ingestion


class FakeGarmin:
    def get_sleep_data(self, d): return {"dailySleepDTO": {"sleepTimeSeconds": 5.5 * 3600, "deepSleepSeconds": 6000,
                                                            "remSleepSeconds": 1200, "sleepScores": {"overall": {"value": 60}}},
                                         "restingHeartRate": 37}
    def get_hrv_data(self, d): return {"hrvSummary": {"lastNightAvg": 81, "weeklyAvg": 76, "status": "BALANCED"}}
    def get_stats(self, d): return {"totalSteps": 12000, "averageStressLevel": 22, "bodyBatteryAtWakeTime": 85,
                                    "bodyBatteryHighestValue": 92, "bodyBatteryLowestValue": 21}
    def get_body_battery(self, a, b): return [{"charged": 50, "drained": 60}]
    def get_activities_by_date(self, start, end, *a, **k):
        return [{"activityId": 1, "startTimeLocal": f"{date.fromisoformat(end) - timedelta(days=1)} 07:00:00",
                 "activityName": "Corsa", "activityType": {"typeKey": "running"}, "distance": 8000, "duration": 2520,
                 "averageHR": 150, "maxHR": 171, "aerobicTrainingEffect": 3.1}]


def test_full_run_dry(tmp_path, monkeypatch, capsys):
    for k, v in {"NEWS_ENABLED": "false", "LLM_ENABLED": "false", "RUNS_PATH": str(tmp_path / "runs.csv"),
                 "FEEDBACK_DB": str(tmp_path / "fb.sqlite"), "WEEKLY_CHART_PATH": str(tmp_path / "w.png"),
                 "QUOTES_LOG_PATH": str(tmp_path / "q.csv"), "SUGGESTIONS_LOG_PATH": str(tmp_path / "s.csv"),
                 "USER_SUGGESTIONS_PATH": str(tmp_path / "u.json")}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(ingestion, "get_api", lambda: FakeGarmin())
    monkeypatch.setattr(ingestion, "fetch_telegram_commands", lambda: [])
    monkeypatch.setattr(extras, "fetch_forecast", lambda: (_ for _ in ()).throw(RuntimeError("offline")))

    hist = tmp_path / "history.csv"
    day = date(2026, 10, 11)                                   # domenica
    rows = [{"date": str(day - timedelta(days=k)), "sleep_hours": 7.0 + (k % 3) * 0.2, "sleep_score": 75,
             "hrv_last_night": 75 + k % 4, "resting_hr": 38, "body_battery_at_wake": 80, "stress_avg": 25,
             "steps": 10000 + 300 * k} for k in range(1, 20)]
    pd.DataFrame(rows).to_csv(hist, index=False)

    result = build_graph().invoke({"run_date": str(day), "dry_run": True, "history_path": str(hist), "errors": []})
    out = capsys.readouterr().out
    assert "CORSA" in out and "8,0 km" in out and "5:15/km" in out
    assert "SETTIMANA 05/10 – 11/10" in out and (tmp_path / "w.png").exists()
    assert result["running"]["week"]["n"] == 1
    assert not [e for e in result["errors"] if not e.startswith("extras") and "meteo" not in e], result["errors"]


def test_llm_logs_keep_context_and_sources(tmp_path, monkeypatch):
    import json
    from health_agent import llm
    monkeypatch.setenv("LLM_LOG_PATH", str(tmp_path / "rec.jsonl"))
    monkeypatch.setenv("NEWS_LLM_LOG_PATH", str(tmp_path / "news.jsonl"))
    monkeypatch.setattr(llm, "ask_llm", lambda ctx: {"summary": "s", "recommendations": ["r"]})
    state = {"run_date": "2026-10-09", "metrics": {}, "insights": [], "trends": []}
    llm.llm_recommendations(state, ["regola"])
    rec = json.loads((tmp_path / "rec.jsonl").read_text(encoding="utf-8"))
    assert rec["context"]["data"] == "2026-10-09" and rec["rules"] == ["regola"]
    llm.log_news("2026-10-09", False, [{"n": 1, "title": "T", "text": "x"}], {0: "sintesi"}, "narrazione", [])
    news = json.loads((tmp_path / "news.jsonl").read_text(encoding="utf-8"))
    assert news["stories"][0]["text"] == "x" and news["narrative"] == "narrazione"
