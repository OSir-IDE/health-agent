"""Embedding semantici locali via Ollama (default: bge-m3, multilingue).

Se Ollama non risponde, chi chiama ricade sul confronto a frammenti di caratteri.
Un errore del server (es. poca memoria perché un altro modello è ancora caricato) viene ritentato:
prima si scaricano gli altri modelli da Ollama, poi si riprova.
"""
import os
import time

import numpy as np
import requests

BATCH = 64
RETRIES = 2            # tentativi dopo il primo
RETRY_WAIT = 3         # secondi, raddoppia ad ogni tentativo


def model() -> str:
    return os.getenv("NEWS_EMBED_MODEL", "bge-m3")


def _url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _unload_others() -> None:
    """Scarica dalla memoria i modelli diversi da quello di embedding (di solito l'LLM del giro precedente)."""
    try:
        running = requests.get(f"{_url()}/api/ps", timeout=10).json().get("models", [])
        for m in running:
            name = m.get("name") or m.get("model")
            if name and name.split(":")[0] != model().split(":")[0]:
                requests.post(f"{_url()}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
    except Exception:
        pass                                    # solo un tentativo di aiuto: l'errore vero è quello del retry


def _post_batch(batch: list[str]) -> list:
    last = None
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(f"{_url()}/api/embed", timeout=120,
                              json={"model": model(), "input": batch, "keep_alive": "2m"})
            if r.ok:
                return r.json()["embeddings"]
            last = RuntimeError(f"Ollama {r.status_code}: {r.text[:150].strip()}")
        except requests.RequestException as e:
            last = RuntimeError(f"{type(e).__name__}: {str(e)[:150]}")
        if attempt < RETRIES:
            _unload_others()
            time.sleep(RETRY_WAIT * 2 ** attempt)
    raise last


def embed(texts: list[str]) -> np.ndarray:
    """Vettori normalizzati (n, d): il prodotto scalare tra due righe è la similarità del coseno."""
    out = []
    for i in range(0, len(texts), BATCH):
        out.extend(_post_batch(texts[i:i + BATCH]))
    v = np.asarray(out, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
