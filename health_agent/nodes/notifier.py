"""Nodo 5 — Notifier: compone il report e lo invia su Telegram (HTML, tabella a larghezza fissa)."""
import html
import os
import re
from datetime import date

from .. import feedback, running
from ..state import HealthState
from ..telegram_logs import send_message

WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
HRV_STATUS = {"BALANCED": "bilanciato", "UNBALANCED": "sbilanciato", "LOW": "basso", "POOR": "scarso"}


# --- formattazione valori --------------------------------------------------
def _hm(hours: float) -> str:
    m = round(hours * 60)
    return f"{m // 60}h{m % 60:02d}"


def _thousands(v: float, sign: bool = False) -> str:
    return (f"{v:+,.0f}" if sign else f"{v:,.0f}").replace(",", ".")


def _small(d: float) -> str:
    """Delta intero, ma con un decimale se piccolo (evita '-0')."""
    return f"{d:+.0f}" if abs(d) >= 1 else f"{d:+.1f}"


FORMAT = {  # metrica -> (valore, delta)
    "sleep_hours": (_hm, lambda d: f"{round(d * 60):+d}m"),
    "steps":       (_thousands, lambda d: _thousands(d, sign=True)),
    "hrv_last_night": (lambda v: f"{v:.0f}ms", _small),
    "resting_hr":  (lambda v: f"{v:.0f}bpm", _small),
}
DEFAULT_FMT = (lambda v: f"{v:.0f}", _small)


def _row(key: str, info: dict) -> str:
    vfmt, dfmt = FORMAT.get(key, DEFAULT_FMT)
    value = vfmt(float(info["today"])) if info["today"] is not None else "n/d"
    s = info["status"]
    if s in ("missing", "no_baseline"):
        tail = "–"
    elif s == "normal":
        tail = "≈"
    else:
        mark = " ⚠" if s == "alert" else " ✓" if info["good"] > 0 else ""
        tail = f"{'▲' if info['delta'] > 0 else '▼'} {dfmt(info['delta'])}{mark}"
    if info.get("below_floor") and "⚠" not in tail:
        tail = tail.replace(" ✓", "") + " ⚠"     # sopra la tua media ma comunque sotto soglia
    return f"{info['label']:<10}{value:>7}  {tail}"


def _table(state: HealthState, block: str) -> list[str]:
    metrics, rec = state.get("metrics", {}), state.get("daily_record", {})
    rows = []
    for key, info in metrics.items():
        if info["block"] != block:
            continue
        rows.append(_row(key, info))
        if key == "sleep_hours" and rec.get("deep_sleep_min") is not None:
            rows.append(f" prof {_hm(rec['deep_sleep_min'] / 60)} · REM {_hm((rec.get('rem_sleep_min') or 0) / 60)}")
        if key == "hrv_last_night" and rec.get("hrv_status"):
            rows.append(f" stato: {HRV_STATUS.get(rec['hrv_status'], rec['hrv_status'].lower())}")
    if block == "giorno":
        mod, vig = rec.get("moderate_intensity_min"), rec.get("vigorous_intensity_min")
        if mod is not None or vig is not None:
            mod, vig = int(mod or 0), int(vig or 0)
            rows.append(f"{'Intensità':<10}{f'{mod + vig}min':>7}")
            rows.append(f" mod {mod} · vig {vig}")
        if rec.get("body_battery_max") is not None:
            bb = f"{rec['body_battery_max']:.0f}/{(rec.get('body_battery_min') or 0):.0f}"
            rows.append(f"{'BB max/min':<10}{bb:>7}")
    return rows


# --- composizione ----------------------------------------------------------
def _weather_lines(w: dict) -> list[str]:
    e = html.escape
    city = e(os.getenv("WEATHER_CITY", "Roma"))
    hours, mins = divmod(int(w["daylight"].total_seconds() // 60), 60)
    b = w["best"]
    lines = ["", f"🌤 <b>METEO {city.upper()}</b>",
             f"{w['tmin']:.0f}° → {w['tmax']:.0f}° · {e(w['sky'])} · pioggia max {w['rain_max']:.0f}%",]
    if b["dry"]:
        lines.append(f"🏃 Momento migliore per uscire: {b['start']}–{b['end']} ({b['temp']}°, pioggia {b['rain']:.0f}%)")
    else:
        lines.append(f"🏃 Nessuna fascia asciutta oggi: la meno piovosa è {b['start']}–{b['end']} ({b['rain']:.0f}%), meglio un'attività al chiuso")
    for a, z in w["rain_ranges"]:
        lines.append(f"☔ Pioggia probabile {a}–{z}")
    for a, z in w["uv_ranges"]:
        lines.append(f"🧴 UV alto {a}–{z}: se esci, meglio fuori da questa fascia")
    lines.append(f"🌅 Alba {w['sunrise']} · tramonto {w['sunset']} ({hours}h{mins:02d} di luce)")
    return lines


def build_message(state: HealthState) -> str:
    e = html.escape
    d = date.fromisoformat(state["run_date"])
    insights = state.get("insights", [])
    metrics = state.get("metrics", {})
    lines = [f"📊 <b>Report {WEEKDAYS[d.weekday()]} {d:%d/%m}</b>"]
    lines += [f"⚠️ {e(i['description'])}" for i in insights if i["type"] in ("warning", "baseline")]

    if state.get("weather"):
        lines += _weather_lines(state["weather"])

    if not state.get("daily_record"):
        lines += ["", "Nessun dato disponibile — controlla sync Garmin/log."]
        lines += _closing(state)
        return "\n".join(lines)

    for block, title in (("notte", "🌙 <b>STANOTTE</b>"), ("giorno", "☀️ <b>IERI</b>")):
        rows = _table(state, block)
        if rows:
            lines += ["", title, "<pre>" + e("\n".join(rows)) + "</pre>"]
        if block == "notte":
            for i in insights:
                if i["type"] == "sleep_floor":
                    lines.append(f"😴 {e(i['description'])}")

    if state.get("running"):
        lines += running.report_lines(state["running"])

    lines += ["", "📈 <b>TENDENZE 7 GIORNI</b>"]
    trends = state.get("trends", [])
    if trends:
        for t in trends:
            _, dfmt = FORMAT.get(t["metric"], DEFAULT_FMT)
            arrow = "↗ in aumento" if t["weekly_change"] > 0 else "↘ in calo"
            verdict = "👍" if t["improving"] else "da tenere d'occhio"
            lines.append(f"• {e(t['label'])} {arrow} ({dfmt(t['weekly_change'])}/sett.) — {verdict}")
    else:
        lines.append("Nessuna tendenza marcata: valori stabili.")

    hl = state.get("highlight")
    if hl:
        i = metrics[hl]
        vfmt, _ = FORMAT.get(hl, DEFAULT_FMT)
        lines += ["", f"👍 <b>Punto forte:</b> {e(i['label'])} {vfmt(float(i['today']))} "
                      f"(media {vfmt(i['mean'])})"]

    for kind, title, status in state.get("log_added", []):
        if status == "added":
            lines.append(f"➕ Aggiunto alla lista {kind}: {e(title)}")
        elif status == "duplicate":
            lines.append(f"↩️ Già in lista {kind}: {e(title)}")
        else:
            lines.append(f"❓ #{kind} senza titolo — formato: #{kind} Titolo | nota")

    if state.get("recommendations"):
        lines += ["", "✅ <b>FOCUS DI OGGI</b>"]
        if state.get("day_summary"):
            lines.append(f"<i>{e(state['day_summary'])}</i>")
        lines += [f"{n}. {e(r)}" for n, r in enumerate(state["recommendations"], 1)]

    lines += _closing(state)
    errs = state.get("errors") or []
    if errs:
        lines += ["", "🔧 <b>Problemi tecnici</b> (il resto del report è valido)"]
        lines += [f"• {e(x[:120])}" for x in errs[:5]]
    lines += ["", "<i>▲▼ oltre ½ dev.std dalla tua media · ✓ verso migliore · ⚠ oltre 1 dev.std verso peggiore o sonno sotto soglia · – baseline insufficiente</i>"]
    return "\n".join(lines)


def _closing(state: HealthState) -> list[str]:
    e, out = html.escape, []
    sug = state.get("suggestion")
    if sug:
        icon = "🎬" if sug["kind"] == "film" else "📚"
        meta = ", ".join(str(x) for x in (sug.get("by"), sug.get("year")) if x)
        line = f"<b>{e(sug['title'])}</b>" + (f" ({e(meta)})" if meta else "") + (f": {e(sug['why'])}" if sug.get("why") else "")
        out += ["", f"{icon} <b>{sug['kind'].upper()}</b> — {e(sug['reason'])}", line]
    q = state.get("quote")
    if q:
        out += ["", f"💭 <i>«{e(q['text'])}»</i>", f"— {e(q['author'])}, {e(q['source'])}"]
    return out


def to_plain(msg: str) -> str:
    return html.unescape(re.sub(r"</?(b|i|pre)>", "", msg))


def notifier_node(state: HealthState) -> dict:
    msg = build_message(state)
    if state.get("dry_run", True):
        print(to_plain(msg))
        return {"message": msg}

    recs, day = state.get("recommendations") or [], state["run_date"]
    markup = feedback.keyboard("rec", day, len(recs), {}) if recs and feedback.enabled("rec") else None
    try:
        msg_id = send_message(msg, html=True, reply_markup=markup)
    except Exception as e:
        return {"message": msg, "errors": [f"telegram: {e}"]}
    errors = []
    if markup:
        try:
            feedback.record_sent(day, "rec", [{"idx": i, "label": r} for i, r in enumerate(recs, 1)], msg_id)
        except Exception as e:
            errors.append(f"feedback: {type(e).__name__}: {e}")
    return {"message": msg, "errors": errors}
