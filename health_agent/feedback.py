"""Feedback dai pulsanti Telegram: tastiere, voti salvati e apprendimento degli interessi.

Nessun processo sempre attivo: i tocchi restano in coda su Telegram e vengono letti dal task
(ogni 30 minuti, vedi telegram_logs.process_updates). Qui non c'è rete, solo SQLite e logica.

Tabelle (data/feedback.sqlite):
  sent   cosa è stato inviato con i pulsanti (notizie e consigli), con tema e fonte
  votes  l'ultimo voto per elemento (ripremere un altro pulsante sostituisce il voto)
  msgs   id del messaggio Telegram che porta la tastiera, per aggiornarla dopo il voto
"""
import os
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sent (
  day TEXT, kind TEXT, idx INTEGER, label TEXT, section TEXT, topic TEXT, source TEXT, link TEXT,
  weight REAL, PRIMARY KEY (day, kind, idx));
CREATE TABLE IF NOT EXISTS votes (
  day TEXT, kind TEXT, idx INTEGER, value INTEGER, ts TEXT, PRIMARY KEY (day, kind, idx));
CREATE TABLE IF NOT EXISTS msgs (
  day TEXT, kind TEXT, msg_id INTEGER, n INTEGER, PRIMARY KEY (day, kind));
"""

# callback_data (max 64 byte): "n|2026-10-09|3|1" = notizia 3 piaciuta; "r|2026-10-09|2|-1" = consiglio 2 non fatto
CODE = {"n": "news", "r": "rec"}
CALLBACK = re.compile(r"^([nr])\|(\d{4}-\d{2}-\d{2})\|(\d{1,2})\|(1|-1)$")
HALF_LIFE_DAYS = 30        # un voto vale la metà dopo 30 giorni
MAX_NET = 4.0              # voti netti massimi considerati per tema


def enabled(kind: str = "news") -> bool:
    """Pulsanti sotto i messaggi: spenti di default (FEEDBACK_BUTTONS=true li accende).

    Notizie 👍/👎 con FEEDBACK_BUTTONS; i ✅/✖ sui consigli richiedono anche FEEDBACK_REC_BUTTONS=true.
    """
    on = ("1", "true", "si", "sì", "yes")
    if os.getenv("FEEDBACK_BUTTONS", "false").strip().lower() not in on:
        return False
    return kind != "rec" or os.getenv("FEEDBACK_REC_BUTTONS", "false").strip().lower() in on


def db_path() -> Path:
    return Path(os.getenv("FEEDBACK_DB", "data/feedback.sqlite"))


def _connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------------- scrittura
def record_sent(day: str, kind: str, items: list[dict], msg_id: int | None) -> None:
    """items: [{"idx", "label", "section", "topic", "source", "link", "weight"}] (campi opzionali tranne idx e label)."""
    with closing(_connect()) as c, c:
        for it in items:
            c.execute("INSERT OR REPLACE INTO sent VALUES (?,?,?,?,?,?,?,?,?)",
                      (day, kind, it["idx"], it["label"], it.get("section"), it.get("topic"),
                       it.get("source"), it.get("link"), it.get("weight")))
        if msg_id is not None:
            c.execute("INSERT OR REPLACE INTO msgs VALUES (?,?,?,?)", (day, kind, msg_id, len(items)))


def handle_callback(data: str) -> tuple[str, str, int, int] | None:
    """Registra un tocco. Ritorna (tipo, giorno, indice, valore) o None se il dato non è nostro."""
    m = CALLBACK.match(data or "")
    if not m:
        return None
    kind, day, idx, value = CODE[m[1]], m[2], int(m[3]), int(m[4])
    with closing(_connect()) as c, c:
        if not c.execute("SELECT 1 FROM sent WHERE day=? AND kind=? AND idx=?", (day, kind, idx)).fetchone():
            return None                       # tastiera di una versione precedente o db ricreato
        c.execute("INSERT OR REPLACE INTO votes VALUES (?,?,?,?,?)",
                  (day, kind, idx, value, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return kind, day, idx, value


# ---------------------------------------------------------------------------------- tastiere
def votes_for(day: str, kind: str) -> dict[int, int]:
    with closing(_connect()) as c:
        return dict(c.execute("SELECT idx, value FROM votes WHERE day=? AND kind=?", (day, kind)).fetchall())


def keyboard(kind: str, day: str, n: int, votes: dict[int, int]) -> dict:
    """inline_keyboard Telegram. Il voto dato è marcato con ✓.

    Notizie: due storie per riga ("👍1 👎1 | 👍2 👎2"). Consigli: una riga per consiglio.
    """
    code = "n" if kind == "news" else "r"

    def btn(idx: int, value: int, text: str) -> dict:
        mark = "✓ " if votes.get(idx) == value else ""
        return {"text": f"{mark}{text}", "callback_data": f"{code}|{day}|{idx}|{value}"}

    if kind == "news":
        cells = [[btn(i, 1, f"👍{i}"), btn(i, -1, f"👎{i}")] for i in range(1, n + 1)]
        rows = [cells[k] + (cells[k + 1] if k + 1 < len(cells) else []) for k in range(0, len(cells), 2)]
    else:
        rows = [[btn(i, 1, f"✅ Fatto {i}"), btn(i, -1, f"✖ Non fatto {i}")] for i in range(1, n + 1)]
    return {"inline_keyboard": rows}


def stored_keyboard(day: str, kind: str) -> tuple[int, dict] | None:
    """(id messaggio, tastiera aggiornata) per un messaggio già inviato."""
    with closing(_connect()) as c:
        row = c.execute("SELECT msg_id, n FROM msgs WHERE day=? AND kind=?", (day, kind)).fetchone()
    return (row[0], keyboard(kind, day, row[1], votes_for(day, kind))) if row else None


# ---------------------------------------------------------------------------------- apprendimento
def topic_factors(today: date | None = None) -> dict[str, float]:
    """Moltiplicatore del peso di ogni tema, dai voti sulle notizie 'Per te'.

    voto netto = somma(±1 · 0,5^(età/30 giorni)), limitato a ±4. Fattore = 1 + 0,15 · netto → tra 0,4 e 1,6.
    I voti su 'In primo piano' non contano: lì decide la copertura, non i tuoi interessi.
    """
    today = today or date.today()
    step = float(os.getenv("FEEDBACK_STEP", "0.15"))
    with closing(_connect()) as c:
        rows = c.execute("SELECT s.topic, v.value, v.day FROM votes v JOIN sent s "
                         "ON s.day=v.day AND s.kind=v.kind AND s.idx=v.idx "
                         "WHERE v.kind='news' AND s.section='personal' AND s.topic IS NOT NULL").fetchall()
    net: dict[str, float] = {}
    for topic, value, day in rows:
        age = max((today - date.fromisoformat(day)).days, 0)
        net[topic] = net.get(topic, 0.0) + value * 0.5 ** (age / HALF_LIFE_DAYS)
    return {t: 1 + step * max(-MAX_NET, min(MAX_NET, v)) for t, v in net.items()}


def adjusted_topics(topics: list[dict], today: date | None = None) -> list[dict]:
    """Copia dei temi con i pesi corretti dai voti (interests.json non viene modificato)."""
    factors = topic_factors(today)
    return [{**t, "weight": t["weight"] * factors.get(t["id"], 1.0)} for t in topics]


# ---------------------------------------------------------------------------------- statistiche
def adherence(start: str, end: str) -> dict:
    """Consigli inviati e voti nell'intervallo [start, end] (date ISO, estremi inclusi)."""
    with closing(_connect()) as c:
        sent = c.execute("SELECT COUNT(*) FROM sent WHERE kind='rec' AND day BETWEEN ? AND ?", (start, end)).fetchone()[0]
        votes = c.execute("SELECT value, COUNT(*) FROM votes WHERE kind='rec' AND day BETWEEN ? AND ? GROUP BY value",
                          (start, end)).fetchall()
    v = dict(votes)
    return {"sent": sent, "done": v.get(1, 0), "skipped": v.get(-1, 0)}


def news_votes(start: str, end: str) -> list[dict]:
    with closing(_connect()) as c:
        rows = c.execute("SELECT s.topic, s.section, s.label, v.value FROM votes v JOIN sent s "
                         "ON s.day=v.day AND s.kind=v.kind AND s.idx=v.idx "
                         "WHERE v.kind='news' AND v.day BETWEEN ? AND ?", (start, end)).fetchall()
    return [{"topic": t, "section": s, "label": l, "value": v} for t, s, l, v in rows]


def top_stories(start: str, end: str, limit: int = 5) -> list[dict]:
    """Le storie 'In primo piano' della settimana più riprese (weight = numero di testate)."""
    with closing(_connect()) as c:
        rows = c.execute("SELECT day, label, source, link, weight FROM sent WHERE kind='news' AND section='top' "
                         "AND day BETWEEN ? AND ? ORDER BY weight DESC, day DESC", (start, end)).fetchall()
    return [{"day": d, "label": l, "source": s, "link": k, "weight": w} for d, l, s, k, w in rows[:limit]]
