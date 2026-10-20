"""Nodo 8 — Settimanale: grafici e riepilogo della settimana (di default la domenica mattina).

Parte dopo il giornale e non lo blocca: se fallisce, report e giornale sono già arrivati.
"""
import html
import os
import re
from datetime import date

import pandas as pd

from .. import running, weekly
from ..state import HealthState
from ..telegram_logs import send_message, send_photo


def is_weekly_day(run_date: str) -> bool:
    return date.fromisoformat(run_date).weekday() == int(os.getenv("WEEKLY_DAY", "6"))   # 0=lunedì … 6=domenica


def weekly_node(state: HealthState) -> dict:
    if os.getenv("WEEKLY_ENABLED", "true").strip().lower() not in ("1", "true", "si", "sì", "yes"):
        return {}
    if not (state.get("weekly") or is_weekly_day(state["run_date"])):
        return {}
    day = state["run_date"]
    try:
        hist = pd.read_csv(state["history_path"], dtype={"date": str}).sort_values("date")
        hist = hist[hist["date"] <= day]
        if len(hist) < 3:
            return {"errors": ["settimanale: storico troppo corto"]}
        png = weekly.render_chart(hist, day, os.getenv("WEEKLY_CHART_PATH", f"data/weekly_{day}.png"))
        text = weekly.build_text(hist, day, {"week_lines": running.week_lines(day)})
    except Exception as e:
        return {"errors": [f"settimanale: {type(e).__name__}: {e}"]}

    if state.get("dry_run", True):
        print("\n" + "─" * 40 + f"\n[grafico salvato in {png}]\n" + html.unescape(re.sub(r"</?(b|i|a|pre)\b[^>]*>", "", text)))
        return {"weekly_message": text}
    try:
        send_photo(str(png))
        send_message(text, html=True)
    except Exception as e:
        return {"weekly_message": text, "errors": [f"settimanale.invio: {type(e).__name__}: {e}"]}
    return {"weekly_message": text}
