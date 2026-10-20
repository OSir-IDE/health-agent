"""Nodo 3 — Analysis: confronta la mattina di oggi con la tua baseline personale.

Produce:
  - metrics:  per ogni metrica valore di oggi, media/dev.std della baseline, z-score, stato
  - trends:   tendenze sui 7 giorni via regressione lineare
  - highlight: il "punto forte" (metrica più migliorata)
  - focus:    la metrica più vicina alla soglia negativa (per il consiglio quando va tutto bene)
  - insights: allarmi e avvisi (usati anche dalle raccomandazioni)
"""
from datetime import date

import os

import numpy as np
import pandas as pd

from ..state import HealthState

BASELINE_DAYS = 7
MIN_VALID = 5          # giorni validi minimi per una baseline affidabile
Z_NOTABLE = 0.5        # oltre: ▲/▼
Z_ALERT = 1.0          # oltre, nel verso peggiore: ⚠
TREND_DAYS = 7
TREND_R2 = 0.5         # la retta deve spiegare almeno metà della variazione
TREND_MIN_SHIFT = 0.5  # spostamento settimanale in dev.std delle ultime 2 settimane
SLEEP_FLOOR_DEFAULT = 7.0  # ore: indicazione generale per gli adulti, modificabile con SLEEP_MIN_HOURS

# metrica -> (verso "peggiore", etichetta, blocco)
METRICS = {
    "sleep_hours":          ("low",  "Sonno",      "notte"),
    "sleep_score":          ("low",  "Punteggio",  "notte"),
    "hrv_last_night":       ("low",  "HRV",        "notte"),
    "resting_hr":           ("high", "FC riposo",  "notte"),
    "body_battery_at_wake": ("low",  "BB sveglia", "notte"),
    "stress_avg":           ("high", "Stress",     "giorno"),
    "steps":                ("low",  "Passi",      "giorno"),
}
NIGHT = [m for m, (_, _, b) in METRICS.items() if b == "notte"]


def _good_score(z: float, worse: str) -> float:
    """z-score orientato: positivo = meglio della media, negativo = peggio."""
    return z if worse == "low" else -z


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _trend(window: pd.DataFrame, col: str, ref_std: float) -> dict | None:
    """Regressione lineare sui giorni della finestra (gestisce giorni mancanti via data reale)."""
    s = _series(window, col)
    ok = s.notna()
    if ok.sum() < MIN_VALID or not ref_std:
        return None
    days = window.loc[ok, "date"].map(lambda d: date.fromisoformat(d).toordinal()).to_numpy(float)
    y = s[ok].to_numpy(float)
    slope, intercept = np.polyfit(days, y, 1)
    fitted = slope * days + intercept
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - fitted) ** 2).sum() / ss_tot if ss_tot else 0.0
    weekly = slope * 7
    if r2 >= TREND_R2 and abs(weekly) >= TREND_MIN_SHIFT * ref_std:
        return {"weekly_change": weekly, "r2": round(r2, 2)}
    return None


def analysis_node(state: HealthState) -> dict:
    hist = pd.read_csv(state["history_path"], dtype={"date": str}).sort_values("date")
    today = state["daily_record"]
    past = hist[hist["date"] < today["date"]]
    baseline = past.tail(BASELINE_DAYS)
    window = hist[hist["date"] <= today["date"]].tail(TREND_DAYS)   # include oggi
    longer = hist[hist["date"] <= today["date"]].tail(14)

    insights, metrics, trends = [], {}, []

    if all(today.get(m) is None for m in NIGHT):
        insights.append({"type": "warning",
                         "description": "Dati di stanotte non ancora disponibili: sincronizza l'orologio e rilancia."})
    if len(baseline) < MIN_VALID:
        insights.append({"type": "baseline",
                         "description": f"Storico di {len(baseline)} giorni: servono almeno {MIN_VALID} giorni "
                                        f"per i confronti (usa --backfill)."})

    for col, (worse, label, block) in METRICS.items():
        info = {"label": label, "block": block, "worse": worse, "today": today.get(col),
                "mean": None, "delta": None, "z": None, "status": "no_baseline"}
        values = _series(baseline, col).dropna()
        if info["today"] is None:
            info["status"] = "missing"
        elif len(values) >= MIN_VALID and values.std():
            mean, std = values.mean(), values.std()
            delta = float(info["today"]) - mean
            z = delta / std
            good = _good_score(z, worse)
            info.update(mean=mean, delta=delta, z=z, good=good,
                        status="alert" if good <= -Z_ALERT else "notable" if abs(z) >= Z_NOTABLE else "normal")
            if info["status"] == "alert":
                insights.append({"type": "deviation", "metric": col, "label": label, "block": block,
                                 "description": f"{label}: {info['today']} (media 7gg {mean:.1f})"})
        metrics[col] = info

        ref_std = _series(longer, col).std()
        t = _trend(window, col, ref_std)
        if t:
            improving = (t["weekly_change"] > 0) == (worse == "low")
            trends.append({"metric": col, "label": label, "improving": improving, **t})

    # Soglia assoluta: il confronto con la media non vede un sonno cronicamente corto
    floor = float(os.getenv("SLEEP_MIN_HOURS", SLEEP_FLOOR_DEFAULT))
    sleep_today = today.get("sleep_hours")
    if sleep_today is not None and float(sleep_today) < floor:
        last7 = _series(window, "sleep_hours").dropna()
        short = int((last7 < floor).sum())
        metrics["sleep_hours"]["below_floor"] = True
        insights.append({"type": "sleep_floor", "metric": "sleep_hours", "floor": floor,
                         "nights_below": short, "nights_total": len(last7),
                         "description": f"Sonno sotto le {floor:g} ore: {short} notti su {len(last7)} negli ultimi 7 giorni"})

    scored = {m: i for m, i in metrics.items() if i.get("good") is not None}
    highlight = None
    if scored:
        best = max(scored, key=lambda m: scored[m]["good"])
        if scored[best]["good"] >= Z_NOTABLE:
            highlight = best
    focus = min(scored, key=lambda m: scored[m]["good"]) if scored else None

    return {"insights": insights, "metrics": metrics, "trends": trends,
            "highlight": highlight, "focus": focus}
