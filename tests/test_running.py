"""Test della sezione corsa: attività Garmin sintetiche (stessi nomi di campo dell'API), nessuna rete."""
from datetime import date, timedelta

import pandas as pd
import pytest

from health_agent import running


@pytest.fixture(autouse=True)
def tmp_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_PATH", str(tmp_path / "runs.csv"))


def act(day, km=8.0, minutes=42.0, hr=150, te=3.0, kind="running", aid=None):
    return {"activityId": aid or int(day.replace("-", "")) + int(km * 10),
            "startTimeLocal": f"{day} 07:30:00", "activityName": "Corsa",
            "activityType": {"typeKey": kind}, "distance": km * 1000, "duration": minutes * 60,
            "averageHR": hr, "maxHR": hr + 20, "aerobicTrainingEffect": te,
            "averageRunningCadenceInStepsPerMinute": 172, "elevationGain": 40}


class FakeApi:
    def __init__(self, acts): self.acts = acts
    def get_activities_by_date(self, start, end, *a, **k): return self.acts


def test_fetch_filters_non_running_and_incomplete():
    acts = [act("2026-10-08"), act("2026-10-07", kind="cycling"), act("2026-10-06", kind="trail_running"),
            {**act("2026-10-05"), "distance": 0}]
    rows = running.fetch_runs(FakeApi(acts), "2026-10-01", "2026-10-09")
    assert [r["type"] for r in rows] == ["running", "trail_running"]
    assert rows[0]["km"] == 8.0 and rows[0]["minutes"] == 42.0


def test_pace_format():
    assert running.pace(42, 8) == "5:15/km"
    assert running.pace(27, 5.2) == "5:12/km"


def test_save_is_idempotent_and_upserts():
    rows = running.fetch_runs(FakeApi([act("2026-10-08")]), "a", "b")
    running.save_runs(rows)
    total = running.save_runs(rows)
    assert len(total) == 1


def _runs(spec):
    rows = running.fetch_runs(FakeApi([act(d, **kw) for d, kw in spec]), "a", "b")
    return running.save_runs(rows)


def test_summary_week_totals_and_no_jump_with_stable_volume():
    spec = [(str(date(2026, 10, 9) - timedelta(days=k)), {"km": 8.0}) for k in (1, 3, 5, 8, 10, 12, 15, 17, 19, 22)]
    s = running.summarize(_runs(spec), "2026-10-09")
    assert s["week"]["n"] == 3 and s["week"]["km"] == pytest.approx(24.0)
    assert s["days_since"] == 1 and not s["jump"]


def test_volume_jump_detected_and_advised():
    spec = [("2026-10-08", {"km": 15.0}), ("2026-10-06", {"km": 10.0}),                     # 25 km questa settimana
            ("2026-09-30", {"km": 8.0}), ("2026-09-23", {"km": 8.0}), ("2026-09-16", {"km": 8.0})]
    s = running.summarize(_runs(spec), "2026-10-09")
    assert s["jump"] and s["jump_pct"] > 200
    assert "salito" in running.recommendations(s, {})[0]


def test_hard_run_yesterday_suggests_recovery():
    s = running.summarize(_runs([("2026-10-08", {"te": 4.5})]), "2026-10-09")
    assert any("impegnativa" in r for r in running.recommendations(s, {}))


def test_idle_suggestion_only_when_rested_and_dry():
    s = running.summarize(_runs([("2026-10-04", {})]), "2026-10-09")
    weather = {"best": {"dry": True, "start": "7", "end": "9"}}
    assert "buon momento" in running.recommendations(s, {"weather": weather, "insights": []})[0]
    tired = {"weather": weather, "insights": [{"type": "deviation", "metric": "hrv_last_night"}]}
    assert running.recommendations(s, tired) == []                                    # il recupero ha la precedenza
    assert running.recommendations(s, {"weather": {"best": {"dry": False}}, "insights": []}) == []


def test_no_recent_runs_gives_nothing():
    assert running.summarize(_runs([("2026-08-01", {})]), "2026-10-09") is None
    assert running.recommendations(None, {}) == []


def _eff_runs(shift_last=0.0, kind="running", n=9):
    """n uscite da 6 km con FC tra 138 e 166; velocità = relazione fissa con la FC, le ultime 3 spostate di shift_last."""
    spec = []
    for k in range(n):
        hr = 138 + 3.5 * ((k * 4) % 9)                        # intensità alternate, come in una settimana vera
        speed = 150 + 1.1 * (hr - 138)                         # m/min
        if k >= n - 3:
            speed *= 1 + shift_last
        spec.append((str(date(2026, 9, 1) + timedelta(days=3 * k)), {"km": 6.0, "minutes": 6000 / speed, "hr": hr,
                                                                      "kind": kind, "aid": 1000 + k}))
    return spec


def test_efficiency_not_fooled_by_easy_vs_hard_runs():
    # corse facili e dure alternate, ma sempre sulla stessa relazione velocità~FC: nessuna variazione
    eff = running.efficiency(_runs(_eff_runs(0.0)), date(2026, 10, 9))
    assert eff and abs(eff["change_pct"]) < 0.5


def test_efficiency_detects_real_shift():
    eff = running.efficiency(_runs(_eff_runs(0.06)), date(2026, 10, 9))
    assert eff and eff["change_pct"] > 3


def test_efficiency_needs_enough_runs_and_ignores_track():
    assert running.efficiency(_runs(_eff_runs(0.06, n=8)), date(2026, 10, 9)) is None
    assert running.efficiency(_runs(_eff_runs(0.06, kind="track_running")), date(2026, 10, 9)) is None


def test_lines_render():
    s = running.summarize(_runs([("2026-10-08", {"te": 3.4})]), "2026-10-09")
    text = "\n".join(running.report_lines(s))
    assert "CORSA" in text and "5:15/km" in text and "ieri" in text and "3,4" in text
