"""Corsa: attività di corsa da Garmin, archivio locale (data/runs.csv) e sintesi per report e settimanale.

Tutto deterministico. Con poche uscite i confronti restano prudenti: sotto le soglie indicate
(MIN_RUNS_*) il modulo mostra i fatti (distanza, passo, FC) e non inventa tendenze.
I nomi dei campi Garmin sono letti con .get(): se un campo non c'è per le tue attività, resta vuoto.
"""
import html
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["activity_id", "date", "name", "type", "km", "minutes", "avg_hr", "max_hr", "cadence",
           "aerobic_te", "load", "vo2max", "elev_gain"]
MIN_REF_RUNS = 6               # uscite di riferimento (≥3 km, con FC, non in pista) prima delle ultime 3
JUMP_RATIO = 1.3               # volume della settimana oltre 1,3× la media delle 3 precedenti = aumento rapido
JUMP_MIN_KM = 3.0              # e almeno 3 km in più (evita falsi allarmi con poca base)
HARD_TE = 4.0                  # effetto allenante aerobico: da 4,0 in su = uscita impegnativa
IDLE_DAYS = 3                  # giorni senza corsa prima di suggerirne una


def runs_path() -> Path:
    return Path(os.getenv("RUNS_PATH", "data/runs.csv"))


# ---------------------------------------------------------------------------------- raccolta
def _activity_row(a: dict) -> dict | None:
    kind = ((a.get("activityType") or {}).get("typeKey") or "").lower()
    if "running" not in kind:                      # running, treadmill_running, trail_running, track_running…
        return None
    dist, dur = a.get("distance"), a.get("duration")
    if not dist or not dur:
        return None
    return {
        "activity_id": a.get("activityId"),
        "date": str(a.get("startTimeLocal") or "")[:10],
        "name": a.get("activityName") or "",
        "type": kind,
        "km": round(dist / 1000, 2),
        "minutes": round(dur / 60, 1),
        "avg_hr": a.get("averageHR"),
        "max_hr": a.get("maxHR"),
        "cadence": a.get("averageRunningCadenceInStepsPerMinute"),
        "aerobic_te": a.get("aerobicTrainingEffect"),
        "load": a.get("activityTrainingLoad"),
        "vo2max": a.get("vO2MaxValue"),
        "elev_gain": a.get("elevationGain"),
    }


def fetch_runs(api, start: str, end: str) -> list[dict]:
    """Corse tra due date (incluse). Tutte le attività, filtrate qui: non dipende dal parametro 'tipo' dell'API."""
    rows = [_activity_row(a) for a in api.get_activities_by_date(start, end)]
    return [r for r in rows if r and r["date"]]


def save_runs(rows: list[dict]) -> pd.DataFrame:
    """Aggiunge/aggiorna le corse in data/runs.csv (chiave: activity_id) e ritorna l'archivio completo."""
    path = runs_path()
    old = pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame(columns=COLUMNS)
    if rows:
        new = pd.DataFrame(rows, columns=COLUMNS)
        old = pd.concat([old, new], ignore_index=True) if len(old) else new
        old = old.drop_duplicates(subset="activity_id", keep="last").sort_values("date")
        path.parent.mkdir(parents=True, exist_ok=True)
        old.to_csv(path, index=False)
    return old


def load_runs() -> pd.DataFrame:
    path = runs_path()
    return pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame(columns=COLUMNS)


# ---------------------------------------------------------------------------------- formattazione
def pace(minutes: float, km: float) -> str:
    """5,2 km in 27 min -> '5:12/km'."""
    s = round(minutes * 60 / km)
    return f"{s // 60}:{s % 60:02d}/km"


def _km(v: float) -> str:
    return f"{v:.1f}".replace(".", ",") + " km"


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


# ---------------------------------------------------------------------------------- analisi
def _window(runs: pd.DataFrame, end: date, days: int, offset: int = 0) -> pd.DataFrame:
    """Corse negli `days` giorni che finiscono `offset` giorni prima di `end` (estremi inclusi)."""
    last = end - timedelta(days=offset)
    first = last - timedelta(days=days - 1)
    return runs[(runs["date"] >= str(first)) & (runs["date"] <= str(last))]


def _totals(w: pd.DataFrame) -> dict:
    if w.empty:
        return {"n": 0, "km": 0.0, "minutes": 0.0, "pace": None, "avg_hr": None}
    km, mins = float(w["km"].sum()), float(w["minutes"].sum())
    hr = w.dropna(subset=["avg_hr"])
    avg_hr = float(np.average(hr["avg_hr"], weights=hr["minutes"])) if len(hr) else None
    return {"n": len(w), "km": km, "minutes": mins, "pace": pace(mins, km), "avg_hr": avg_hr}


def efficiency(runs: pd.DataFrame, end: date) -> dict | None:
    """Le ultime 3 uscite sono più veloci o più lente del tuo solito, a parità di FC? (in %)

    Il confronto diretto "velocità/FC" tra uscite è sbagliato: uscite facili e dure hanno rapporti diversi.
    Qui si stima la tua relazione abituale velocità~FC sulle uscite precedenti degli ultimi 90 giorni
    (retta) e si guarda di quanto le ultime 3 se ne scostano. Escluse le sedute in pista (ripetute, FC e passo misti) e
    le uscite sotto 3 km. Servono almeno MIN_REF_RUNS uscite di riferimento con FC variabile e le ultime 3 dentro il loro campo di FC.
    """
    r = runs[(runs["date"] <= str(end)) & (runs["date"] > str(end - timedelta(days=90)))]
    r = r.dropna(subset=["avg_hr"]).sort_values(["date", "activity_id"])
    r = r[(r["avg_hr"] > 0) & (r["km"] >= 3) & ~r["type"].fillna("").str.contains("track")]
    ref, recent = r.iloc[:-3], r.iloc[-3:]
    if len(ref) < MIN_REF_RUNS or len(recent) < 3 or ref["avg_hr"].std() < 5:
        return None
    # la retta si stima SOLO sulle uscite precedenti: se includesse le ultime, le assorbirebbe
    slope, intercept = np.polyfit(ref["avg_hr"].to_numpy(float), (ref["km"] * 1000 / ref["minutes"]).to_numpy(float), 1)
    hr_new = recent["avg_hr"].to_numpy(float)
    if hr_new.min() < ref["avg_hr"].min() - 10 or hr_new.max() > ref["avg_hr"].max() + 10:
        return None                                    # FC fuori dal campo noto: sarebbe un'estrapolazione
    fitted = slope * hr_new + intercept
    if (fitted <= 0).any():
        return None
    speed = (recent["km"] * 1000 / recent["minutes"]).to_numpy(float)
    return {"change_pct": float(((speed - fitted) / fitted * 100).mean()), "n": len(r)}


def summarize(runs: pd.DataFrame, today: str) -> dict | None:
    """Sintesi per il report del mattino `today`. None se non c'è nessuna corsa recente."""
    end = date.fromisoformat(today)
    recent = runs[(runs["date"] <= today) & (runs["date"] > str(end - timedelta(days=21)))]
    if recent.empty:
        return None
    last = recent.sort_values(["date", "activity_id"]).iloc[-1]
    last_d = date.fromisoformat(last["date"])
    week, prev = _totals(_window(runs, end, 7)), _totals(_window(runs, end, 7, offset=7))
    base_weeks = [_totals(_window(runs, end, 7, offset=7 * k))["km"] for k in (1, 2, 3)]
    base = float(np.mean(base_weeks))
    jump = bool(base > 0 and week["km"] >= JUMP_RATIO * base and week["km"] - base >= JUMP_MIN_KM)
    return {
        "last": {"date": last["date"], "days_ago": (end - last_d).days, "km": float(last["km"]),
                 "pace": pace(last["minutes"], last["km"]), "avg_hr": _num(last["avg_hr"]),
                 "max_hr": _num(last["max_hr"]), "te": _num(last["aerobic_te"]), "name": last["name"]},
        "days_since": (end - last_d).days,
        "week": week, "prev_week": prev, "base_km": base,
        "jump": jump, "jump_pct": (week["km"] / base - 1) * 100 if base > 0 else None,
        "efficiency": efficiency(runs, end),
        "vo2max": _num(recent.dropna(subset=["vo2max"]).sort_values("date")["vo2max"].iloc[-1]) if recent["vo2max"].notna().any() else None,
    }


# ---------------------------------------------------------------------------------- testo
WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]


def _when(days_ago: int, iso: str) -> str:
    if days_ago == 0:
        return "oggi"
    if days_ago == 1:
        return "ieri"
    return f"{WEEKDAYS[date.fromisoformat(iso).weekday()]} {iso[8:]}/{iso[5:7]}"


def report_lines(s: dict) -> list[str]:
    """Blocco 'CORSA' del report del mattino (HTML già pronto)."""
    e = html.escape
    last, w, p = s["last"], s["week"], s["prev_week"]
    parts = [f"{_km(last['km'])} · {last['pace']}"]
    if last["avg_hr"]:
        parts.append(f"FC {last['avg_hr']:.0f}" + (f" (max {last['max_hr']:.0f})" if last["max_hr"] else ""))
    lines = ["", "🏃 <b>CORSA</b>", f"Ultima: {e(_when(last['days_ago'], last['date']))} · " + " · ".join(parts)]
    if last["te"]:
        lines.append(f"Effetto allenante aerobico {last['te']:.1f}".replace(".", ","))
    lines.append(f"7 giorni: {w['n']} {'uscita' if w['n'] == 1 else 'uscite'} · {_km(w['km'])}"
                 + (f" (prec. {_km(p['km'])})" if p["n"] else ""))
    if s["jump"]:
        lines.append(f"⚠ Volume +{s['jump_pct']:.0f}% sulla media delle 3 settimane precedenti")
    return lines


def week_lines(today: str) -> list[str]:
    """Riga per il settimanale (HTML). Vuota se non ci sono corse nella settimana."""
    runs = load_runs()
    if runs.empty:
        return []
    s = summarize(runs, today)
    if not s or not s["week"]["n"]:
        return []
    w, p = s["week"], s["prev_week"]
    line = f"🏃 Corsa: {w['n']} {'uscita' if w['n'] == 1 else 'uscite'} · {_km(w['km'])}"
    if p["n"]:
        line += f" (prec. {_km(p['km'])})"
    line += f" · passo medio {w['pace']}"
    if w["avg_hr"]:
        line += f" · FC {w['avg_hr']:.0f}"
    out = [line]
    if s["jump"]:
        out.append(f"⚠ Volume in aumento rapido: +{s['jump_pct']:.0f}% sulla media delle 3 settimane prima")
    eff = s["efficiency"]
    if eff and abs(eff["change_pct"]) >= 3:
        verso = "più veloce" if eff["change_pct"] > 0 else "più lento"
        out.append(f"A parità di FC le ultime 3 uscite sono {verso} del tuo solito ({eff['change_pct']:+.0f}%): "
                   f"indicativo, dipende anche da caldo e percorso")
    if s["vo2max"]:
        out.append(f"VO2max stimato dall'orologio: {s['vo2max']:.0f}")
    return out


# ---------------------------------------------------------------------------------- consigli a regole
def recommendations(s: dict | None, state: dict) -> list[str]:
    """Consigli di corsa a regole. Il recupero (sonno, HRV) ha sempre la precedenza."""
    if not s:
        return []
    recs = []
    if s["jump"]:
        recs.append(f"Il volume di corsa è salito del {s['jump_pct']:.0f}% rispetto alle ultime settimane: "
                    f"tieni questa settimana più leggera.")
    last = s["last"]
    if last["days_ago"] <= 1 and last["te"] and last["te"] >= HARD_TE:
        recs.append("L'ultima corsa è stata impegnativa: oggi recupero attivo o riposo.")
    insights = state.get("insights", [])
    tired = any(i["type"] in ("deviation", "sleep_floor") and i.get("metric") in
                ("hrv_last_night", "resting_hr", "sleep_hours", "body_battery_at_wake") for i in insights)
    best = (state.get("weather") or {}).get("best")
    if s["days_since"] >= IDLE_DAYS and not tired and not recs and best and best.get("dry"):
        recs.append(f"Sono {s['days_since']} giorni che non corri e il recupero è nella norma: "
                    f"buon momento per una corsa leggera tra le {best['start']} e le {best['end']}.")
    return recs


def context(s: dict | None) -> dict | None:
    """Sintesi per il contesto dell'LLM (solo numeri già calcolati)."""
    if not s:
        return None
    return {"ultima_corsa_giorni_fa": s["days_since"], "km_ultima": round(s["last"]["km"], 1),
            "km_ultimi_7gg": round(s["week"]["km"], 1), "km_7gg_precedenti": round(s["prev_week"]["km"], 1),
            "volume_in_aumento_rapido": s["jump"]}
