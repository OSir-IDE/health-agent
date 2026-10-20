"""Nodo 5 — Extras: meteo, frase del giorno, film/libro. Ogni parte è opzionale e non blocca il report."""
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from ..state import HealthState
from ..telegram_logs import load_user_suggestions
from ..weather import fetch_forecast, summarize

CONTENT = Path(__file__).resolve().parent.parent / "content"

# metrica problematica -> tema della frase
THEME_BY_METRIC = {
    "sleep_hours": "riposo", "sleep_score": "riposo", "hrv_last_night": "riposo",
    "resting_hr": "riposo", "body_battery_at_wake": "riposo",
    "stress_avg": "serenita", "steps": "cammino",
}
LOW_STATE = ("body_battery_at_wake", "sleep_hours")


def _theme(state: HealthState) -> str:
    alerts = [i["metric"] for i in state.get("insights", []) if i["type"] == "deviation"]
    if alerts:
        return THEME_BY_METRIC.get(alerts[0], "riflessione")
    if any(not t["improving"] for t in state.get("trends", [])):
        return "costanza"
    return ["riflessione", "tempo", "cammino"][date.fromisoformat(state["run_date"]).toordinal() % 3]


def pick_quote(state: HealthState) -> dict:
    """Frase del tema giusto usata meno di recente; stabile se rilanci nello stesso giorno."""
    quotes = json.loads((CONTENT / "quotes.json").read_text(encoding="utf-8"))
    day = state["run_date"]
    log_path = Path(os.getenv("QUOTES_LOG_PATH", "data/quotes_log.csv"))
    log = pd.read_csv(log_path, dtype={"date": str}) if log_path.exists() else pd.DataFrame(columns=["date", "text"])

    today = log[log["date"] == day]
    if not today.empty:
        found = next((q for q in quotes if q["text"] == today.iloc[-1]["text"]), None)
        if found:
            return found

    theme = _theme(state)
    pool = [q for q in quotes if theme in q["themes"]] or quotes
    last_used = dict(zip(log["text"], log["date"]))
    quote = min(pool, key=lambda q: last_used.get(q["text"], ""))   # mai usata, o la più vecchia

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([log, pd.DataFrame([{"date": day, "text": quote["text"]}])]).to_csv(log_path, index=False)
    return quote


def _feels_low(state: HealthState) -> bool:
    m = state.get("metrics", {})
    return any((m.get(k) or {}).get("good") is not None and m[k]["good"] <= -0.5 for k in LOW_STATE)


def pick_suggestion(state: HealthState, weather: dict | None) -> dict | None:
    d = date.fromisoformat(state["run_date"])
    if d.weekday() == 6:
        kind, reason = "libro", "per la settimana"
    elif weather and weather.get("evening_rain"):
        kind, reason = "film", "stasera è prevista pioggia"
    elif d.weekday() in (4, 5):
        kind, reason = "film", "per stasera"
    else:
        return None

    base = json.loads((CONTENT / "suggestions.json").read_text(encoding="utf-8"))[kind]
    items = load_user_suggestions().get(kind, []) + base      # i tuoi titoli hanno la precedenza
    log_path = Path(os.getenv("SUGGESTIONS_LOG_PATH", "data/suggestions_log.csv"))
    log = pd.read_csv(log_path, dtype={"date": str}) if log_path.exists() else pd.DataFrame(columns=["date", "kind", "title"])

    already = log[(log["date"] == str(d)) & (log["kind"] == kind)]
    if not already.empty:                                   # rilancio nello stesso giorno: stesso titolo
        item = next((i for i in items if i["title"] == already.iloc[-1]["title"]), None)
        return {"kind": kind, "reason": reason, **item} if item else None

    light = _feels_low(state)
    pool = [i for i in items if i.get("light") is not False] if light else items   # None = adatto a entrambi
    last_seen = {t: dt for t, dt in zip(log["title"], log["date"])}
    item = min(pool, key=lambda i: last_seen.get(i["title"], ""))   # mai proposto prima, o il più vecchio
    if light:
        reason += ", qualcosa di leggero"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([log, pd.DataFrame([{"date": str(d), "kind": kind, "title": item["title"]}])]).to_csv(log_path, index=False)
    return {"kind": kind, "reason": reason, **item}


def extras_node(state: HealthState) -> dict:
    out, errors = {}, []
    try:
        out["weather"] = summarize(fetch_forecast(state["run_date"]))
    except Exception as e:
        out["weather"] = None
        errors.append(f"meteo: {type(e).__name__}: {e}")
    for key, fn in (("quote", lambda: pick_quote(state)),
                    ("suggestion", lambda: pick_suggestion(state, out["weather"]))):
        try:
            out[key] = fn()
        except Exception as e:
            out[key] = None
            errors.append(f"{key}: {type(e).__name__}: {e}")
    return {**out, "errors": errors}
