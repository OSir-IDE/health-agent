"""Giornale personale: raccoglie i feed RSS, raggruppa le notizie uguali e le ordina.

Punteggio di ogni storia (gruppo di articoli sullo stesso fatto):
  - importanza: su quante testate DIVERSE compare
  - rilevanza:  corrispondenza con il profilo di interessi (content/interests.json)
  - freschezza: penalità per le notizie più vecchie

Similarità tra articoli:
  - modo "semantico" (predefinito): embedding multilingue bge-m3 via Ollama. Riconosce la stessa
    notizia anche con parole diverse o in un'altra lingua.
  - modo "caratteri" (riserva, se Ollama non risponde): TF-IDF su frammenti di 3-5 caratteri.
Le due misure hanno scale diverse, quindi soglie diverse (vedi _thresholds).
"""
import html
import json
import math
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import numpy as np
import requests
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import archive, embeddings, feedback

CONTENT = Path(__file__).resolve().parent / "content"
MAX_AGE_HOURS = 30
STALE_DAYS = 7             # un feed senza articoli da più di N giorni è "fermo"
REPEAT_DAYS = 3            # non riproporre un link già inviato negli ultimi N giorni
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36",
      "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"}


def _thresholds(mode: str) -> tuple[float, float]:
    """(stessa storia, troppo simili per comparire entrambe). Lette da .env al momento dell'uso."""
    if mode == "semantico":
        return (float(os.getenv("NEWS_EMB_SIMILARITY", "0.70")), float(os.getenv("NEWS_EMB_DIVERSITY", "0.62")))
    return (float(os.getenv("NEWS_SIMILARITY", "0.21")), float(os.getenv("NEWS_DIVERSITY", "0.12")))


# ---------------------------------------------------------------------------------- testo
def _plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _norm(text: str) -> str:
    """minuscolo e senza accenti: 'Università' -> 'universita'"""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _dedupe_sentences(text: str) -> str:
    """Alcuni feed ripetono la stessa frase due o tre volte nella descrizione."""
    for n in range(3, 1, -1):
        size = len(text) // n
        if size > 20 and text[:size] * n == text[:size * n]:
            return text[:size].strip()
    return text


def _doc(item: dict) -> str:
    return item["title"] + ". " + item["summary"][:200]


# ---------------------------------------------------------------------------------- raccolta
def fetch_feed(src: dict, now: datetime, per_source: int) -> dict:
    """Ritorna {"items", "error", "newest"}; i feed fermi sono segnalati in "error"."""
    try:
        r = requests.get(src["url"], headers=UA, timeout=20)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
    except Exception as e:
        return {"items": [], "error": f"{type(e).__name__}", "newest": None}
    items, newest = [], None
    for e in parsed.entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime(*t[:6], tzinfo=timezone.utc) if t else None
        if published and (newest is None or published > newest):
            newest = published
        if published and now - published > timedelta(hours=MAX_AGE_HOURS):
            continue
        title = _plain(e.get("title"))
        if not title or not e.get("link"):
            continue
        items.append({"title": title, "link": e["link"], "source": src["name"], "lang": src["lang"],
                      "general": src.get("general", False), "published": published,
                      "summary": _dedupe_sentences(_plain(e.get("summary") or e.get("description")))[:400]})
    # tetto per fonte: i più recenti, così un feed molto prolifico non domina la raccolta
    items.sort(key=lambda i: i["published"] or now, reverse=True)
    error = None
    if newest and now - newest > timedelta(days=STALE_DAYS):
        error = f"fermo dal {newest:%d/%m/%Y}"
    elif not parsed.entries:
        error = "feed vuoto"
    return {"items": items[:per_source], "error": error, "newest": newest}


def fetch_all(sources: list[dict], now: datetime) -> tuple[list[dict], list[str], dict]:
    per_source = int(os.getenv("NEWS_PER_SOURCE", "25"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda s: fetch_feed(s, now, per_source), sources))
    items, failed, seen, status = [], [], set(), {}
    for src, res in zip(sources, results):
        status[src["name"]] = res
        if res["error"]:
            failed.append(f"{src['name']}: {res['error']}")
        for it in res["items"]:
            if it["link"] not in seen:          # lo stesso articolo in due feed della stessa testata
                seen.add(it["link"])
                items.append(it)
    return items, failed, status


# ---------------------------------------------------------------------------------- similarità
def char_similarity(docs: list[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    return cosine_similarity(tfidf.fit_transform([_norm(d) for d in docs]))


def vectorize(items: list[dict]) -> tuple[np.ndarray | None, str, str | None]:
    """(vettori, modo, errore). Riusa dall'archivio gli embedding già calcolati."""
    if os.getenv("NEWS_SEMANTIC", "true").lower() not in ("1", "true", "si", "sì", "yes"):
        return None, "caratteri", None
    try:
        m = embeddings.model()
        known = archive.known_embeddings([i["link"] for i in items], m)
        missing = [i for i in items if i["link"] not in known]
        if missing:
            new = embeddings.embed([_doc(i) for i in missing])
            known.update({i["link"]: v for i, v in zip(missing, new)})
        return np.stack([known[i["link"]] for i in items]), "semantico", None
    except Exception as e:
        return None, "caratteri", f"embedding non disponibili, uso il confronto a caratteri ({str(e)[:140] or type(e).__name__})"


def similarity(items: list[dict], vectors: np.ndarray | None) -> np.ndarray:
    if vectors is not None:
        return np.clip(vectors @ vectors.T, -1, 1)
    return char_similarity([_doc(i) for i in items])


def cluster(items: list[dict], sim: np.ndarray, threshold: float) -> list[list[int]]:
    """Gruppi di indici. Legame medio: due gruppi si uniscono solo se SIMILI IN MEDIA,
    non basta una coppia (evita le catene A~B~C che univano notizie diverse)."""
    if len(items) < 2:
        return [[i] for i in range(len(items))]
    model = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                    distance_threshold=1 - threshold)
    labels = model.fit_predict(np.clip(1 - sim, 0, 2))
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, []).append(idx)
    return list(groups.values())


# ---------------------------------------------------------------------------------- punteggio
def _keyword_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for k in keywords if re.search(r"\b" + re.escape(k), text))


def score_story(idx: list[int], items: list[dict], topics: list[dict], exclude: list[str], now: datetime) -> dict:
    group = [items[i] for i in idx]
    testate = lambda arts: {a["source"].split()[0] for a in arts}   # "ANSA Economia" e "ANSA" = una testata
    # articolo rappresentativo: testata generalista italiana con descrizione, e tra queste la PIÙ RECENTE
    # (per le notizie in evoluzione: "30 morti" invece del primo lancio con "12 morti")
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    lead_pos = max(range(len(group)), key=lambda k: (group[k]["general"], group[k]["lang"] == "it",
                                                     bool(group[k]["summary"]), group[k]["published"] or oldest))
    text = _norm(" ".join(a["title"] + " " + a["title"] + " " + a["summary"] for a in group))
    newest = max((a["published"] for a in group if a["published"]), default=None)
    age_h = (now - newest).total_seconds() / 3600 if newest else 12

    best, best_score = None, 0.0
    for t in topics:
        hits = _keyword_hits(text, t["keywords"])
        if hits and t["weight"] > 0:
            s = t["weight"] * (1 + math.log(hits))
            if s > best_score:
                best, best_score = t, s
    lead = group[lead_pos]
    return {
        "lead": lead, "lead_idx": idx[lead_pos], "articles": group,
        "n_sources": len(testate(group)), "n_sources_it": len(testate([a for a in group if a["lang"] == "it"])),
        "age_h": age_h, "topic": best, "relevance": best_score,
        "excluded": _keyword_hits(_norm(lead["title"]), exclude) > 0,
        "freshness": math.exp(-age_h / 24),
    }


# ---------------------------------------------------------------------------------- selezione
def sent_recently(path: Path, now: datetime) -> set[str]:
    if not path.exists():
        return set()
    cutoff = (now - timedelta(days=REPEAT_DAYS)).date().isoformat()
    log = json.loads(path.read_text(encoding="utf-8"))
    return {link for day, links in log.items() if day >= cutoff for link in links}


def remember_sent(path: Path, day: str, links: list[str]) -> None:
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    log[day] = sorted(set(log.get(day, [])) | set(links))
    log = dict(sorted(log.items())[-30:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def select(stories: list[dict], sim: np.ndarray, diversity: float,
           n_top: int, n_personal: int, skip_links: set[str]) -> tuple[list, list]:
    fresh = [s for s in stories
             if not s["excluded"] and not any(a["link"] in skip_links for a in s["articles"])]
    # In primo piano: l'agenda italiana, cioè storie riprese da almeno 2 testate italiane
    top = sorted([s for s in fresh if s["n_sources_it"] >= 2],
                 key=lambda s: s["n_sources"] + s["freshness"], reverse=True)[:n_top]
    taken = {id(s) for s in top}
    # Per te: rilevanza, con un bonus per copertura e freschezza
    candidates = sorted([s for s in fresh if id(s) not in taken and s["relevance"] > 0],
                        key=lambda s: s["relevance"] + 0.5 * (s["n_sources"] - 1) + s["freshness"], reverse=True)
    chosen = [s["lead_idx"] for s in top]
    personal, per_topic = [], {}
    for t in top:                          # anche il primo piano conta nel limite di 2 storie per tema
        if t["topic"]:
            per_topic[t["topic"]["id"]] = per_topic.get(t["topic"]["id"], 0) + 1
    for s in candidates:
        tid = s["topic"]["id"]
        # varietà: scarta una storia troppo simile a una già scelta (es. tre articoli diversi sulla Borsa)
        if per_topic.get(tid, 0) >= 2 or any(sim[s["lead_idx"], j] >= diversity for j in chosen):
            continue
        personal.append(s)
        chosen.append(s["lead_idx"])
        per_topic[tid] = per_topic.get(tid, 0) + 1
        if len(personal) == n_personal:
            break
    return top, personal


def build_digest(day: str, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    sources = json.loads((CONTENT / "news_sources.json").read_text(encoding="utf-8"))
    prefs = json.loads((CONTENT / "interests.json").read_text(encoding="utf-8"))
    items, failed, _ = fetch_all(sources, now)
    vectors, mode, warn = vectorize(items)
    sim = similarity(items, vectors)
    same_story, diversity = _thresholds(mode)
    try:
        topics = feedback.adjusted_topics(prefs["topics"])       # i tuoi 👍/👎 spostano i pesi dei temi
    except Exception:
        topics = prefs["topics"]
    stories = [score_story(g, items, topics, prefs.get("exclude", []), now)
               for g in cluster(items, sim, same_story)]
    log_path = Path(os.getenv("NEWS_LOG_PATH", "data/news_log.json"))
    top, personal = select(stories, sim, diversity, int(os.getenv("NEWS_TOP", "3")),
                           int(os.getenv("NEWS_PERSONAL", "5")), sent_recently(log_path, now))
    return {"top": top, "personal": personal, "failed": failed, "n_items": len(items),
            "n_sources_ok": len(sources) - len(failed), "log_path": log_path,
            "items": items, "vectors": vectors, "mode": mode, "warning": warn}
