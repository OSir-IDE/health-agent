"""Report settimanale: confronto con la settimana prima, grafici, consigli seguiti e storie principali.

Tutto deterministico (nessun LLM). Una settimana = 7 mattine che finiscono nel giorno del report
(di default la domenica); la settimana di confronto è quella subito prima.
"""
import html
import json
import os
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                    # nessuna finestra: gira anche dal task
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import feedback
from .nodes.analysis import METRICS
from .nodes.notifier import DEFAULT_FMT, FORMAT

WEEKDAYS_SHORT = ["L", "M", "M", "G", "V", "S", "D"]
CONTENT = Path(__file__).resolve().parent / "content"

# metriche mostrate nei grafici e nella tabella, nell'ordine di lettura
PANELS = ["sleep_hours", "hrv_last_night", "resting_hr", "steps", "stress_avg", "body_battery_at_wake"]

# palette di riferimento (superficie chiara): enfasi = settimana corrente in blu, confronto in grigio
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, GRAY, RED = "#2a78d6", "#a3a29d", "#e34948"


# ---------------------------------------------------------------------------------- dati
def week_frames(hist: pd.DataFrame, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(questa settimana, settimana precedente), ognuna indicizzata sui 7 giorni, con NaN dove manca il dato."""
    end_d = date.fromisoformat(end)
    hist = hist.assign(date=hist["date"].astype(str)).set_index("date")

    def take(last: date) -> pd.DataFrame:
        days = [str(last - timedelta(days=k)) for k in range(6, -1, -1)]
        return hist.reindex(days).apply(pd.to_numeric, errors="coerce")

    return take(end_d), take(end_d - timedelta(days=7))


def compare(hist: pd.DataFrame, end: str) -> list[dict]:
    """Per ogni metrica: media di questa settimana e della precedente, differenza e giudizio."""
    cur, prev = week_frames(hist, end)
    recent = hist.assign(date=hist["date"].astype(str))
    recent = recent[recent["date"] <= end].tail(28)
    rows = []
    for key in PANELS:
        if key not in cur:
            continue
        worse, label, _ = METRICS[key]
        c, p = cur[key].mean(), prev[key].mean() if key in prev else np.nan
        sd = pd.to_numeric(recent[key], errors="coerce").std() if key in recent else np.nan
        verdict = None
        if not (np.isnan(c) or np.isnan(p) or not sd or np.isnan(sd)):
            z = (c - p) / sd
            if abs(z) >= 0.5:
                verdict = "good" if (z > 0) == (worse == "low") else "bad"
        rows.append({"key": key, "label": label, "cur": None if np.isnan(c) else float(c),
                     "prev": None if np.isnan(p) else float(p), "verdict": verdict,
                     "days": int(cur[key].notna().sum())})
    return rows


# ---------------------------------------------------------------------------------- grafico
def render_chart(hist: pd.DataFrame, end: str, path: str | Path) -> Path:
    """Sei riquadri piccoli (uno per metrica): settimana corrente in blu, precedente in grigio."""
    cur, prev = week_frames(hist, end)
    end_d = date.fromisoformat(end)
    labels = [WEEKDAYS_SHORT[(end_d - timedelta(days=k)).weekday()] for k in range(6, -1, -1)]
    floor = float(os.getenv("SLEEP_MIN_HOURS", "7"))
    x = np.arange(7)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), dpi=150, facecolor=SURFACE)
    for ax, key in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        _, label, _ = METRICS[key]
        vfmt = FORMAT.get(key, DEFAULT_FMT)[0]
        if key in prev:
            ax.plot(x, prev[key], color=GRAY, lw=2, marker="o", ms=5, zorder=2)
        if key in cur:
            ax.plot(x, cur[key], color=BLUE, lw=2, marker="o", ms=6, zorder=3)
            mean = cur[key].mean()
            title = f"{label}  ·  media {vfmt(float(mean))}" if not np.isnan(mean) else label
        else:
            title = label
        if key == "sleep_hours":
            ax.axhline(floor, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.text(0.99, 0.97, f"- - soglia {floor:g} h", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=INK_2)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(x, labels, fontsize=8.5, color=INK_2)
        ax.tick_params(axis="y", labelsize=8.5, labelcolor=INK_2, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        if key == "steps":
            ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        if key == "sleep_hours":
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, floor - 0.5), max(hi, floor + 0.5))

    fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=6, label="Questa settimana"),
                        plt.Line2D([], [], color=GRAY, lw=2, marker="o", ms=5, label="Settimana precedente")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"Settimana {end_d - timedelta(days=6):%d/%m} – {end_d:%d/%m}", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------- testo
def _delta_text(key: str, cur: float, prev: float) -> str:
    _, dfmt = FORMAT.get(key, DEFAULT_FMT)
    d = cur - prev
    return f"{'▲' if d > 0 else '▼'} {dfmt(d)}" if abs(d) > 1e-9 else "≈"


def build_text(hist: pd.DataFrame, end: str, running: dict | None = None) -> str:
    e = html.escape
    end_d = date.fromisoformat(end)
    start = str(end_d - timedelta(days=6))
    lines = [f"📅 <b>SETTIMANA {end_d - timedelta(days=6):%d/%m} – {end_d:%d/%m}</b>"]

    rows = []
    for r in compare(hist, end):
        vfmt = FORMAT.get(r["key"], DEFAULT_FMT)[0]
        if r["cur"] is None:
            rows.append(f"{r['label']:<10}{'n/d':>7}")
            continue
        tail = "" if r["prev"] is None else _delta_text(r["key"], r["cur"], r["prev"])
        mark = {"good": " ✓", "bad": " ⚠"}.get(r["verdict"], "")
        rows.append(f"{r['label']:<10}{vfmt(r['cur']):>7}  {tail}{mark}")
    if rows:
        lines += ["<pre>" + e("\n".join(rows)) + "</pre>",
                  "<i>medie della settimana e variazione sulla precedente · ✓/⚠ se oltre ½ dev.std dei 28 giorni</i>"]

    cur, _ = week_frames(hist, end)
    floor = float(os.getenv("SLEEP_MIN_HOURS", "7"))
    sleep = cur["sleep_hours"].dropna() if "sleep_hours" in cur else pd.Series(dtype=float)
    if len(sleep):
        short = int((sleep < floor).sum())
        best = sleep.idxmax()
        lines += ["", f"😴 Notti sotto le {floor:g} ore: {short} su {len(sleep)} · "
                      f"la migliore {WEEKDAY_NAMES[date.fromisoformat(best).weekday()]} "
                      f"({FORMAT['sleep_hours'][0](float(sleep.max()))})"]

    if running and running.get("week_lines"):
        lines += ["", *running["week_lines"]]

    ad = feedback.adherence(start, end)
    if ad["sent"]:
        answered = ad["done"] + ad["skipped"]
        lines += ["", f"✅ Consigli: {ad['done']} fatti, {ad['skipped']} no, {ad['sent'] - answered} senza risposta "
                      f"(su {ad['sent']})"]

    stories = feedback.top_stories(start, end)
    if stories:
        lines += ["", "📰 <b>LE STORIE DELLA SETTIMANA</b>"]
        for s in stories:
            href = html.escape(s["link"] or "", quote=True)
            lines.append(f'• <a href="{href}">{e(s["label"])}</a> <i>({s["day"][8:]}/{s["day"][5:7]}, '
                         f'{s["weight"]:.0f} testate)</i>')

    lines += _taste_lines(start, end)
    return "\n".join(lines)


WEEKDAY_NAMES = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]


def _taste_lines(start: str, end: str) -> list[str]:
    votes = [v for v in feedback.news_votes(start, end) if v["topic"] and v["section"] == "personal"]
    if not votes:
        return []
    labels = {t["id"]: t["label"] for t in json.loads((CONTENT / "interests.json").read_text(encoding="utf-8"))["topics"]}
    net: dict[str, int] = {}
    for v in votes:
        net[v["topic"]] = net.get(v["topic"], 0) + v["value"]
    ups = [f"{labels.get(t, t)} (+{n})" for t, n in sorted(net.items(), key=lambda kv: -kv[1]) if n > 0]
    downs = [f"{labels.get(t, t)} ({n})" for t, n in sorted(net.items(), key=lambda kv: kv[1]) if n < 0]
    out = [""]
    if ups:
        out.append("👍 Hai gradito: " + html.escape(", ".join(ups)))
    if downs:
        out.append("👎 Meno interessanti: " + html.escape(", ".join(downs)))
    factors = feedback.topic_factors()
    moved = [f"{labels.get(t, t)} {f:.2f}×" for t, f in sorted(factors.items()) if abs(f - 1) >= 0.05]
    if moved:
        out.append("<i>Pesi dei temi ora: " + html.escape(", ".join(moved)) + "</i>")
    return out
