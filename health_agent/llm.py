"""Consigli scritti da un LLM locale (Ollama). I calcoli restano deterministici: l'LLM riceve
metriche già analizzate e si limita a trasformarle in consigli concreti.

Tutto è opzionale: se Ollama non risponde o l'output non supera la validazione,
il nodo recommendation usa i consigli a regole.
"""
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import requests

from . import grounding

SYSTEM_PROMPT = """Sei il coach personale di benessere di un ragazzo italiano. Ogni mattina ricevi un JSON
con le sue metriche già analizzate (sonno, HRV, frequenza a riposo, stress, passi),
confrontate con la sua media personale, più tendenze e meteo.

Scrivi da 1 a 3 consigli pratici per OGGI e una sintesi della giornata in una frase.

Regole:
- Usa SOLO i dati forniti. Non inventare numeri, eventi o attività.
- Priorità: prima gli allarmi, poi le tendenze in peggioramento, poi il mantenimento.
- Consigli concreti e fattibili oggi, collegati ai dati. Se utile, usa la fascia oraria migliore del meteo.
- Solo stile di vita (sonno, movimento, pause, luce, alimentazione generica, idratazione).
  MAI diagnosi, malattie, farmaci, integratori o dosaggi. Niente toni allarmistici.
- Italiano, seconda persona singolare, tono diretto e amichevole.
- Ogni consiglio al massimo 30 parole; la sintesi al massimo 20 parole.
- "suggerimenti_regole" sono spunti di un sistema a regole: puoi riformularli, unirli o migliorarli."""

SCHEMA = {
    "type": "object",
    "properties": {
        "sintesi": {"type": "string"},
        "consigli": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
    },
    "required": ["sintesi", "consigli"],
}
def _jsonable(o):
    """numpy (np.bool_, np.float32, np.int64…) -> tipi Python, che json sa scrivere."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"{type(o).__name__} non serializzabile")


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=_jsonable)


NUM_CTX = 8192   # il default di Ollama (4096) rischia di tagliare in silenzio l'inizio dei prompt lunghi
FORBIDDEN = re.compile(r"farmac|diagnos|malatti|integrator|\bmg\b|dosaggi|medicin|patologi", re.IGNORECASE)


def enabled() -> bool:
    return os.getenv("LLM_ENABLED", "false").strip().lower() in ("1", "true", "si", "sì", "yes")


def wait_for_ollama(max_wait: float | None = None, interval: float = 5.0, _get=None, _sleep=None, _clock=None) -> float | None:
    """Aspetta che il server Ollama risponda (dopo l'accensione o il risveglio il PC ci mette un po').

    Ritorna i secondi attesi se risponde, None se non risponde entro il limite (OLLAMA_WAIT_SECONDS, default 120).
    I parametri con underscore servono ai test. Non fa nulla se l'LLM è spento.
    """
    import time

    import requests
    get, sleep, clock = _get or requests.get, _sleep or time.sleep, _clock or time.monotonic
    if max_wait is None:
        max_wait = float(os.getenv("OLLAMA_WAIT_SECONDS", "120"))
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    start = clock()
    while True:
        try:
            if get(f"{url}/api/tags", timeout=5).status_code == 200:
                return clock() - start
        except Exception:
            pass
        if clock() - start >= max_wait:
            return None
        sleep(interval)


def build_context(state: dict, rule_recs: list[str]) -> dict:
    """Solo dati derivati e aggregati: niente nomi, email o storico completo."""
    d = date.fromisoformat(state["run_date"])
    metrics = {}
    for key, i in state.get("metrics", {}).items():
        if i.get("today") is None:
            continue
        metrics[i["label"]] = {
            "oggi": i["today"],
            "media_7gg": round(i["mean"], 1) if i.get("mean") is not None else None,
            "z_score": round(i["z"], 2) if i.get("z") is not None else None,
            "stato": i["status"],               # normal / notable / alert / no_baseline
            "peggio_se": "basso" if i["worse"] == "low" else "alto",
            "blocco": "stanotte" if i["block"] == "notte" else "ieri",
        }
    ctx = {
        "data": str(d),
        "giorno": ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"][d.weekday()],
        "metriche": metrics,
        "allarmi": [i["description"] for i in state.get("insights", []) if i["type"] in ("deviation", "sleep_floor")],
        "tendenze_7gg": [{"metrica": t["label"], "variazione_settimanale": round(t["weekly_change"], 1),
                          "in_miglioramento": t["improving"]} for t in state.get("trends", [])],
        "punto_forte": (state.get("metrics", {}).get(state.get("highlight")) or {}).get("label"),
        "metrica_piu_fragile": (state.get("metrics", {}).get(state.get("focus")) or {}).get("label"),
        "suggerimenti_regole": rule_recs,
    }
    from . import running
    if running.context(state.get("running")):
        ctx["corsa"] = running.context(state.get("running"))
    w = state.get("weather")
    if w:
        ctx["meteo"] = {"min": w["tmin"], "max": w["tmax"], "cielo": w["sky"],
                        "fascia_migliore_per_uscire": f"{w['best']['start']}-{w['best']['end']}" if w["best"]["dry"] else "nessuna (pioggia tutto il giorno)",
                        "pioggia_probabile": [f"{a}-{z}" for a, z in w["rain_ranges"]]}
    return ctx


def validate(out: dict) -> dict:
    recs = [r.strip() for r in out.get("consigli", []) if isinstance(r, str) and r.strip()]
    summary = str(out.get("sintesi", "")).strip()
    if not 1 <= len(recs) <= 3:
        raise ValueError(f"numero di consigli non valido: {len(recs)}")
    for text in recs + [summary]:
        if len(text) > 300:
            raise ValueError("testo troppo lungo")
        if FORBIDDEN.search(text):
            raise ValueError(f"contenuto non ammesso: {text[:60]}")
    return {"summary": summary, "recommendations": recs}


def ask_llm(context: dict) -> dict:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": dumps(context)}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": NUM_CTX},
    }
    r = requests.post(f"{url}/api/chat", json=payload, timeout=int(os.getenv("OLLAMA_TIMEOUT", "180")))
    r.raise_for_status()
    content = r.json()["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()   # modelli "reasoning"
    return validate(json.loads(content))


def log_run(day: str, rule_recs: list[str], result: dict | None, error: str | None, seconds: float,
            context: dict | None = None) -> None:
    """Registro per confrontare nel tempo consigli LLM e consigli a regole."""
    path = Path(os.getenv("LLM_LOG_PATH", "data/llm_log.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"date": day, "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"), "seconds": round(seconds, 1),
             "rules": rule_recs, "llm": result, "error": error, "context": context}
    with path.open("a", encoding="utf-8") as f:
        f.write(dumps(entry) + "\n")


def llm_recommendations(state: dict, rule_recs: list[str]) -> tuple[dict | None, str | None]:
    t0 = time.time()
    context = None
    try:
        context = build_context(state, rule_recs)
        result = ask_llm(context)
        log_run(state["run_date"], rule_recs, result, None, time.time() - t0, context)
        return result, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log_run(state["run_date"], rule_recs, None, err, time.time() - t0, context)
        return None, err


def log_news(day: str, dry_run: bool, stories: list[dict], summaries: dict, narrative: str | None,
             errors: list[str], warnings: list[dict] | None = None) -> None:
    """Registro del giornale: per ogni storia il testo ESATTO dato all'LLM, poi le sintesi e la narrazione.

    Serve a costruire un insieme di prova per un verificatore: frase generata + fonti = esempio da etichettare.
    """
    path = Path(os.getenv("NEWS_LLM_LOG_PATH", "data/news_llm_log.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"date": day, "dry_run": dry_run, "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
             "stories": stories, "summaries": summaries, "narrative": narrative, "errors": errors,
             "name_warnings": warnings or []}
    with path.open("a", encoding="utf-8") as f:
        f.write(dumps(entry) + "\n")


# --------------------------------------------------------------------------- giornale
NEWS_PROMPT = """Ricevi un elenco di notizie (titolo e testo del feed RSS, in italiano o inglese).
Per ciascuna scrivi UNA frase di sintesi in italiano, massimo 25 parole, che spieghi il fatto.

Regole:
- Usa SOLO le informazioni del titolo e del testo forniti. Non aggiungere fatti, cifre, nomi o contesto.
- Mantieni esattamente chi fa cosa, a chi, dove e con quale mezzo: non scambiare i ruoli
  (es. "aggredito sul monopattino dalla banda della minicar" NON diventa "aggredito da una banda in monopattino").
- Se il testo è vuoto o troppo scarno, riformula il titolo in modo chiaro, senza inventare.
- Tono neutro e informativo, nessun giudizio o opinione.
- Restituisci una sintesi per ogni id ricevuto, con lo stesso id."""

NEWS_SCHEMA = {
    "type": "object",
    "properties": {"sintesi": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "testo": {"type": "string"}},
        "required": ["id", "testo"]}}},
    "required": ["sintesi"],
}
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMBER.findall(text)}


def summarize_news(articles: list[dict]) -> dict[int, str]:
    """{indice: sintesi}. Le sintesi che non superano i controlli vengono scartate singolarmente."""
    payload_items = [{"id": i, "titolo": a["title"], "testo": a["summary"][:400]} for i, a in enumerate(articles)]
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    r = requests.post(f"{url}/api/chat", timeout=int(os.getenv("OLLAMA_TIMEOUT", "180")), json={
        "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
        "messages": [{"role": "system", "content": NEWS_PROMPT},
                     {"role": "user", "content": dumps(payload_items)}],
        "format": NEWS_SCHEMA, "stream": False, "options": {"temperature": 0.2, "num_ctx": NUM_CTX}})
    r.raise_for_status()
    content = re.sub(r"<think>.*?</think>", "", r.json()["message"]["content"], flags=re.DOTALL).strip()
    out = {}
    for s in json.loads(content).get("sintesi", []):
        i, text = s.get("id"), str(s.get("testo", "")).strip()
        if not isinstance(i, int) or not 0 <= i < len(articles) or not text or len(text) > 220:
            continue
        source = articles[i]["title"] + " " + articles[i]["summary"]
        if not _numbers(text) <= _numbers(source):        # cifre inventate -> scarta
            continue
        if grounding.ungrounded(text, source):            # "morti", "coma", "rialzo"… senza riscontro -> scarta
            continue
        out[i] = text
    return out


# --------------------------------------------------------------------------- narrazione
NARRATIVE_PROMPT = """Scrivi "Il filo del giorno": un breve testo in italiano (80-130 parole) che
introduca le notizie numerate che ricevi, come l'attacco di un notiziario.

Regole, da rispettare tutte:
- Usa SOLO le informazioni dei testi forniti. Non aggiungere fatti, cifre, nomi, date o contesto
  che non compaiono lì, nemmeno se li conosci: la tua memoria può essere vecchia.
- Ogni frase deve terminare con il riferimento alle notizie da cui viene, tra parentesi quadre: [1], [2][4].
- Niente frasi di cornice senza fatti ("Il filo del giorno segue...", "Ecco le notizie di oggi"):
  comincia direttamente dalla notizia più importante.
- Collega due notizie con un rapporto di causa SOLO se è scritto esplicitamente nei testi.
  Altrimenti accostale per tema ("sul fronte economico", "fuori dall'Italia", "intanto").
- "contesto_precedente" contiene notizie dei giorni scorsi, con la data: puoi usarle per dare
  continuità ("dopo le tensioni di martedì"), citando il giorno, mai come fatti di oggi.
- Tono neutro e informativo, nessuna opinione, nessuna previsione. Non serve citare ogni notizia:
  privilegia le più importanti (le prime) e un filo che le leghi.
- Restituisci solo il testo nel campo "testo"."""
NARRATIVE_SCHEMA = {"type": "object", "properties": {"testo": {"type": "string"}}, "required": ["testo"]}
CITATION = re.compile(r"\[(\d+)\]")
MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
          "settembre", "ottobre", "novembre", "dicembre"]


def _day_label(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]}"


def _sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) > 3]


def _cited_text(sentence: str, story_texts: list[str]) -> str:
    """Testo delle sole storie citate dalla frase ([1][3] -> storie 1 e 3)."""
    return " ".join(story_texts[int(c) - 1] for c in CITATION.findall(sentence) if 1 <= int(c) <= len(story_texts))


def validate_narrative(text: str, n_stories: int, story_texts: list[str]) -> str:
    """story_texts[n-1] = tutto ciò che il modello ha visto sulla storia n (titolo, testo, contesto precedente)."""
    text = re.sub(r"\s+", " ", text).strip()
    words = len(CITATION.sub("", text).split())
    if not 50 <= words <= 170:
        raise ValueError(f"lunghezza fuori misura: {words} parole")
    cited = {int(c) for c in CITATION.findall(text)}
    if not cited or not cited <= set(range(1, n_stories + 1)):
        raise ValueError(f"riferimenti non validi: {sorted(cited)}")
    sentences = _sentences(text)
    uncited = [x for x in sentences if not CITATION.search(x)]
    if uncited:
        raise ValueError(f"frase senza fonte: {uncited[0][:60]}")
    invented = _numbers(CITATION.sub("", text)) - _numbers(" ".join(story_texts))
    if invented:
        raise ValueError(f"cifre non presenti nelle fonti: {sorted(invented)}")
    for sent in sentences:                       # ogni frase contro le SOLE fonti che cita
        problems = grounding.ungrounded(CITATION.sub("", sent), _cited_text(sent, story_texts))
        if problems:
            raise ValueError(f"{problems[0]} (frase: {sent[:70]})")
    return text


def narrative_warnings(text: str, story_texts: list[str]) -> list[dict]:
    """Nomi propri della narrazione che le fonti citate non contengono: solo per il registro, non bloccano."""
    out = []
    for sent in _sentences(text):
        names = grounding.unseen_names(CITATION.sub("", sent), _cited_text(sent, story_texts))
        if names:
            out.append({"frase": sent, "nomi": names})
    return out


def write_narrative(stories: list[dict], background: dict[int, list[dict]]) -> str:
    """stories: [{"titolo","testo","fonti"}] in ordine di numerazione; background: {n: [articoli passati]}.

    Se il primo testo non supera i controlli, il modello riceve il motivo e ha un secondo tentativo.
    """
    payload = []
    for n, s in enumerate(stories, 1):
        entry = {"n": n, **s}
        if background.get(n):
            entry["contesto_precedente"] = [{"data": _day_label(b["day"]), "titolo": b["title"], "testo": b["summary"]}
                                            for b in background[n]]
        payload.append(entry)
    # tutto ciò che il modello vede su ciascuna storia (titolo, testo, contesto precedente con data)
    story_texts = [" ".join([e["titolo"], e["testo"]] +
                            [f'{b["data"]} {b["titolo"]} {b["testo"]}' for b in e.get("contesto_precedente", [])])
                   for e in payload]
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    messages = [{"role": "system", "content": NARRATIVE_PROMPT}, {"role": "user", "content": dumps(payload)}]
    for attempt in range(2):
        r = requests.post(f"{url}/api/chat", timeout=int(os.getenv("OLLAMA_TIMEOUT", "180")), json={
            "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"), "messages": messages,
            "format": NARRATIVE_SCHEMA, "stream": False, "options": {"temperature": 0.3, "num_ctx": NUM_CTX}})
        r.raise_for_status()
        content = re.sub(r"<think>.*?</think>", "", r.json()["message"]["content"], flags=re.DOTALL).strip()
        text = json.loads(content)["testo"]
        try:
            return validate_narrative(text, len(stories), story_texts)
        except ValueError as e:
            if attempt == 1:
                raise ValueError(f"{e} (anche al secondo tentativo)")
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Testo scartato: {e}. Riscrivilo rispettando tutte le regole."}]
