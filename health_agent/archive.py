"""Archivio locale delle notizie (SQLite): ogni articolo raccolto con il suo embedding.

Serve a dare sfondo alla narrazione ("terzo giorno di tensione sui titoli di Stato") con fatti
recenti, invece della memoria del modello, che si ferma alla data del suo addestramento.
"""
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SCHEMA = """CREATE TABLE IF NOT EXISTS articles (
    link TEXT PRIMARY KEY, day TEXT, source TEXT, lang TEXT,
    title TEXT, summary TEXT, published TEXT, model TEXT, emb BLOB)"""


def _path() -> Path:
    return Path(os.getenv("NEWS_ARCHIVE_PATH", "data/news_archive.sqlite"))


def _connect() -> sqlite3.Connection:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute(SCHEMA)
    con.execute("CREATE INDEX IF NOT EXISTS idx_day ON articles(day)")
    return con


def known_embeddings(links: list[str], model: str) -> dict[str, np.ndarray]:
    """Embedding già calcolati (stesso modello): evita di ricalcolarli a ogni esecuzione."""
    if not _path().exists() or not links:
        return {}
    with _connect() as con:
        out = {}
        for i in range(0, len(links), 500):
            chunk = links[i:i + 500]
            q = f"SELECT link, emb FROM articles WHERE model = ? AND link IN ({','.join('?' * len(chunk))})"
            for link, emb in con.execute(q, [model, *chunk]):
                out[link] = np.frombuffer(emb, dtype=np.float32)
        return out


def save(items: list[dict], vectors: np.ndarray, day: str, model: str) -> int:
    rows = [(i["link"], day, i["source"], i["lang"], i["title"], i["summary"],
             i["published"].isoformat() if i["published"] else None, model, v.astype(np.float32).tobytes())
            for i, v in zip(items, vectors)]
    with _connect() as con:
        cur = con.executemany("INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?,?)", rows)
        return cur.rowcount


def days_available(before: str, model: str) -> int:
    if not _path().exists():
        return 0
    with _connect() as con:
        return con.execute("SELECT COUNT(DISTINCT day) FROM articles WHERE day < ? AND model = ?",
                           (before, model)).fetchone()[0]


def related(query: np.ndarray, before: str, model: str, k: int = 3, window_days: int = 30,
            threshold: float = 0.70) -> list[dict]:
    """Articoli dei giorni precedenti più simili a `query`, al massimo uno per giorno (per dare continuità)."""
    start = (date.fromisoformat(before) - timedelta(days=window_days)).isoformat()
    with _connect() as con:
        rows = con.execute("SELECT day, source, title, summary, emb FROM articles "
                           "WHERE day < ? AND day >= ? AND model = ?", (before, start, model)).fetchall()
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
    sims = mat @ query
    best_per_day = {}
    for idx in np.argsort(-sims):
        if sims[idx] < threshold:
            break
        day = rows[idx][0]
        if day not in best_per_day:
            best_per_day[day] = {"day": day, "source": rows[idx][1], "title": rows[idx][2],
                                 "summary": rows[idx][3][:250], "sim": float(sims[idx])}
        if len(best_per_day) == k:
            break
    return sorted(best_per_day.values(), key=lambda x: x["day"])
