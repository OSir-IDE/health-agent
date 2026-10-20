"""Nodo 7 — Giornale: secondo messaggio Telegram con narrazione e notizie del giorno.

Indipendente dal report di salute: se fallisce, il report è già partito.
"""
import html
import os
import re
from datetime import date

from .. import archive, embeddings, feedback, llm
from ..news import build_digest, remember_sent
from ..state import HealthState
from ..telegram_logs import send_message

WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
TELEGRAM_LIMIT = 4000


def _on(var: str, default: str = "true") -> bool:
    return os.getenv(var, default).strip().lower() in ("1", "true", "si", "sì", "yes")


def _esc(t: str) -> str:
    return html.escape(t, quote=False)


def _item(n: int, s: dict, summary: str | None, icon: str, show_sources: bool) -> list[str]:
    a = s["lead"]
    href = html.escape(a["link"], quote=True)
    lines = [f'{n}. {icon}<a href="{href}">{_esc(a["title"])}</a>']
    if summary:
        lines.append(f"   <i>{_esc(summary)}</i>")
    meta = a["source"] + (f" · ripresa da {s['n_sources']} testate" if show_sources and s["n_sources"] > 1 else "")
    lines.append(f"   {_esc(meta)}")
    return lines


def build_news_message(day: str, digest: dict, summaries: dict, narrative: str | None) -> str:
    d = date.fromisoformat(day)
    lines = [f"📰 <b>IL TUO GIORNALE</b> — {WEEKDAYS[d.weekday()]} {d:%d/%m}"]
    if narrative:
        lines += ["", "🧵 <b>IL FILO DEL GIORNO</b>", _esc(narrative)]
    n = 0
    if digest["top"]:
        lines += ["", "🔴 <b>IN PRIMO PIANO</b>"]
        for s in digest["top"]:
            lines += _item(n + 1, s, summaries.get(n), "", True)
            n += 1
    if digest["personal"]:
        lines += ["", "🎯 <b>PER TE</b>"]
        for s in digest["personal"]:
            lines += _item(n + 1, s, summaries.get(n), s["topic"]["icon"] + " ", False)
            n += 1
    if n == 0:
        lines += ["", "Nessuna notizia disponibile al momento."]

    foot = f"{digest['n_items']} articoli da {digest['n_sources_ok']} fonti · raggruppamento {digest['mode']}"
    if summaries or narrative:
        foot += " · testi generati in locale dall'LLM sulle sole fonti: per i dettagli apri gli articoli"
    lines += ["", f"<i>{_esc(foot)}</i>"]
    if digest["failed"]:
        lines.append(f"<i>⚠️ Fonti con problemi: {_esc('; '.join(digest['failed']))}</i>")

    msg = "\n".join(lines)
    while len(msg) > TELEGRAM_LIMIT and "\n" in msg:      # rete di sicurezza: taglia dal fondo
        msg = msg[:msg.rfind("\n", 0, TELEGRAM_LIMIT)]
    return msg


def _background(day: str, digest: dict, stories: list[dict], errors: list[str]) -> dict[int, list[dict]]:
    """Sfondo dall'archivio per ogni notizia, solo se l'archivio copre abbastanza giorni."""
    if digest["vectors"] is None or not _on("NEWS_ARCHIVE"):
        return {}
    model = embeddings.model()
    days = archive.days_available(day, model)
    if days < int(os.getenv("NEWS_ARCHIVE_MIN_DAYS", "7")):
        return {}
    thr = float(os.getenv("NEWS_ARCHIVE_SIMILARITY", "0.70"))
    out = {}
    for n, s in enumerate(stories, 1):
        try:
            rel = archive.related(digest["vectors"][s["lead_idx"]], day, model, threshold=thr)
        except Exception as e:
            errors.append(f"news.archivio: {type(e).__name__}: {e}")
            return {}
        if rel:
            out[n] = rel
    return out


def _sent_items(digest: dict) -> list[dict]:
    """Elementi del giornale con la stessa numerazione del messaggio (prima il primo piano, poi 'Per te')."""
    items, n = [], 0
    for section in ("top", "personal"):
        for s in digest[section]:
            n += 1
            items.append({"idx": n, "label": s["lead"]["title"], "section": section,
                          "topic": s["topic"]["id"] if s["topic"] else None, "source": s["lead"]["source"],
                          "link": s["lead"]["link"], "weight": s["n_sources"]})
    return items


def news_node(state: HealthState) -> dict:
    if not _on("NEWS_ENABLED"):
        return {}
    errors = []
    day = state["run_date"]
    try:
        digest = build_digest(day)
    except Exception as e:
        return {"errors": [f"news: {type(e).__name__}: {e}"]}
    if digest["warning"]:
        errors.append(f"news: {digest['warning']}")

    stories = digest["top"] + digest["personal"]
    summaries, narrative = {}, None
    if llm.enabled() and stories:
        try:
            summaries = llm.summarize_news([s["lead"] for s in stories])
        except Exception as e:
            errors.append(f"news.sintesi (solo titoli): {type(e).__name__}: {e}")
        if _on("NEWS_NARRATIVE"):
            try:
                bg = _background(day, digest, stories, errors)
                narrative = llm.write_narrative(
                    [{"titolo": s["lead"]["title"], "testo": s["lead"]["summary"],
                      "fonti": ", ".join(sorted({a["source"] for a in s["articles"]}))} for s in stories], bg)
            except Exception as e:
                errors.append(f"news.narrazione (omessa): {type(e).__name__}: {e}")

    if summaries or narrative:        # registro per la verifica: fonti esatte + testi generati
        try:
            llm.log_news(day, state.get("dry_run", True),
                         [{"n": n, "title": s["lead"]["title"], "text": s["lead"]["summary"], "link": s["lead"]["link"],
                           "source": s["lead"]["source"], "sources": sorted({a["source"] for a in s["articles"]}),
                           "articles": [{"source": a["source"], "title": a["title"], "text": a["summary"]}
                                        for a in s["articles"]]}
                          for n, s in enumerate(stories, 1)],
                         summaries, narrative, errors,
                         llm.narrative_warnings(narrative, [s["lead"]["title"] + " " + s["lead"]["summary"]
                                                            for s in stories]) if narrative else None)
        except Exception as e:
            errors.append(f"news.registro: {type(e).__name__}: {e}")

    # archivio: salva tutti gli articoli raccolti oggi (anche in prova), così lo sfondo cresce ogni giorno
    if digest["vectors"] is not None and _on("NEWS_ARCHIVE"):
        try:
            archive.save(digest["items"], digest["vectors"], day, embeddings.model())
        except Exception as e:
            errors.append(f"news.archivio: {type(e).__name__}: {e}")

    msg = build_news_message(day, digest, summaries, narrative)
    if state.get("dry_run", True):
        print("\n" + "─" * 40 + "\n" + html.unescape(re.sub(r"</?(b|i|a)\b[^>]*>", "", msg)))
        return {"news_message": msg, "errors": errors}
    n_items = sum(1 for _ in stories)
    markup = feedback.keyboard("news", day, n_items, {}) if n_items and feedback.enabled("news") else None
    try:
        msg_id = send_message(msg, html=True, reply_markup=markup)
        if n_items:       # le storie inviate si registrano sempre: servono al settimanale, anche senza pulsanti
            try:
                feedback.record_sent(day, "news", _sent_items(digest), msg_id if markup else None)
            except Exception as e:
                errors.append(f"feedback: {type(e).__name__}: {e}")
        remember_sent(digest["log_path"], day, [a["link"] for s in stories for a in s["articles"]])
    except Exception as e:
        errors.append(f"news.invio: {type(e).__name__}: {e}")
    return {"news_message": msg, "errors": errors}
