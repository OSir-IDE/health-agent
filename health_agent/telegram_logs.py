"""Comandi Telegram (#film / #libro), tocchi sui pulsanti e invio messaggi.

Nessun processo sempre attivo: Telegram conserva i messaggi non letti per 24 ore,
lo script li scarica al mattino e solo dopo li conferma a Telegram.
"""
import json
import os
import re
from pathlib import Path

import requests

from . import feedback

API = "https://api.telegram.org/bot{token}/{method}"

# "#film Titolo (2023) | perché vederlo"  /  "#libro Titolo | nota"
ADD = re.compile(r"^#(film|libro)\b\s*(.*)$", re.IGNORECASE | re.DOTALL)
YEAR = re.compile(r"\((\d{4})\)\s*$")
BASE_SUGGESTIONS = Path(__file__).resolve().parent / "content" / "suggestions.json"


def parse_addition(text: str) -> dict | None:
    """'#film Perfect Days (2023) | Wim Wenders a Tokyo' -> {'kind': 'film', 'title': 'Perfect Days', ...}"""
    m = ADD.match(text.strip())
    if not m:
        return None
    title, _, why = m[2].partition("|")
    title = title.strip()
    year = YEAR.search(title)
    if year:
        title = title[:year.start()].strip()
    return {"kind": m[1].lower(), "title": title, "by": "",
            "year": int(year[1]) if year else None, "light": None, "why": why.strip()}


def user_suggestions_path() -> Path:
    return Path(os.getenv("USER_SUGGESTIONS_PATH", "data/user_suggestions.json"))


def load_user_suggestions() -> dict:
    p = user_suggestions_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"film": [], "libro": []}


def add_suggestion(item: dict) -> str:
    """Aggiunge un titolo alla lista personale. Ritorna 'added', 'duplicate' o 'empty'."""
    if not item["title"]:
        return "empty"
    kind = item["kind"]
    base = json.loads(BASE_SUGGESTIONS.read_text(encoding="utf-8")).get(kind, [])
    user = load_user_suggestions()
    known = {i["title"].casefold() for i in base + user.get(kind, [])}
    if item["title"].casefold() in known:
        return "duplicate"
    user.setdefault(kind, []).append({k: v for k, v in item.items() if k != "kind"})
    p = user_suggestions_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(user, ensure_ascii=False, indent=1), encoding="utf-8")
    return "added"


def process_updates(token: str, chat_id: str, timeout: int = 0) -> dict:
    """Scarica gli aggiornamenti nuovi: comandi #film/#libro e tocchi sui pulsanti.

    Ritorna {"added": [(tipo, titolo, esito)], "votes": [(tipo, giorno, indice, valore)]}.
    Dopo i voti aggiorna la tastiera del messaggio (✓ sul pulsante scelto). Gli altri messaggi
    vengono ignorati e tutto viene confermato a Telegram solo a elaborazione finita.
    """
    r = requests.get(API.format(token=token, method="getUpdates"), params={"timeout": timeout}, timeout=timeout + 15)
    r.raise_for_status()
    updates = r.json().get("result", [])
    out = {"added": [], "votes": []}
    if not updates:
        return out
    touched = set()
    for u in updates:
        cq = u.get("callback_query")
        if cq:
            if str((cq.get("message") or {}).get("chat", {}).get("id")) == str(chat_id):
                vote = feedback.handle_callback(cq.get("data", ""))
                if vote:
                    out["votes"].append(vote)
                    touched.add((vote[0], vote[1]))
            continue
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(chat_id):
            continue
        addition = parse_addition((msg.get("text") or "").strip())
        if addition:
            out["added"].append((addition["kind"], addition["title"], add_suggestion(addition)))
    last = max(u["update_id"] for u in updates)
    requests.get(API.format(token=token, method="getUpdates"), params={"offset": last + 1, "timeout": 0}, timeout=15)
    for kind, day in touched:
        try:
            refresh_keyboard(token, chat_id, day, kind)
        except Exception:
            pass                                  # la tastiera non aggiornata non è un errore: il voto è salvato
    return out


def sync_updates(token: str, chat_id: str) -> list[tuple]:
    """Compatibilità: solo i comandi #film/#libro (i voti sono comunque registrati)."""
    return process_updates(token, chat_id)["added"]


def refresh_keyboard(token: str, chat_id: str, day: str, kind: str) -> None:
    stored = feedback.stored_keyboard(day, kind)
    if not stored:
        return
    msg_id, markup = stored
    r = requests.post(API.format(token=token, method="editMessageReplyMarkup"),
                      json={"chat_id": chat_id, "message_id": msg_id, "reply_markup": markup}, timeout=15)
    if r.status_code == 400 and "not modified" in r.text:
        return
    r.raise_for_status()


def sync_and_confirm() -> dict | None:
    """Per i giri del task a report già inviato: registra voti e comandi e conferma le aggiunte.

    Costa una sola chiamata a Telegram se non c'è niente di nuovo.
    """
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    res = process_updates(token, chat_id)
    labels = {"added": "➕ Aggiunto alla lista", "duplicate": "↩️ Già in lista"}
    lines = [f"{labels[st]} {kind}: {title}" for kind, title, st in res["added"] if st in labels]
    lines += [f"❓ #{kind} senza titolo — formato: #{kind} Titolo | nota" for kind, _, st in res["added"] if st == "empty"]
    if lines:
        send_message("\n".join(lines))
    return res


def send_message(text: str, html: bool = False, reply_markup: dict | None = None) -> int | None:
    """Invia un messaggio e ritorna il suo id (serve per aggiornare i pulsanti)."""
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    payload = {"chat_id": chat_id, "text": text, "link_preview_options": {"is_disabled": True}}
    if html:
        payload["parse_mode"] = "HTML"
    if reply_markup:
        payload["reply_markup"] = reply_markup
    r = requests.post(API.format(token=token, method="sendMessage"), json=payload, timeout=15)
    r.raise_for_status()
    return (r.json().get("result") or {}).get("message_id")


def send_photo(path: str, caption: str = "") -> None:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    with open(path, "rb") as f:
        r = requests.post(API.format(token=token, method="sendPhoto"),
                          data={"chat_id": chat_id, "caption": caption[:1000]}, files={"photo": f}, timeout=60)
    r.raise_for_status()
