"""Stato condiviso che scorre tra i nodi del grafo."""
import operator
from typing import Annotated, TypedDict


class HealthState(TypedDict, total=False):
    run_date: str                  # giorno analizzato (YYYY-MM-DD)
    dry_run: bool                  # True -> stampa invece di inviare su Telegram
    history_path: str
    sample_path: str               # opzionale: legge un JSON salvato invece di chiamare Garmin

    garmin_raw: dict               # campi Garmin già estratti (v. ingestion.extract_fields)
    log_added: list[tuple]         # titoli inviati con #film/#libro: (tipo, titolo, esito)

    runs_raw: list[dict]           # corse scaricate oggi da Garmin
    running: dict | None           # sintesi corsa (ultima uscita, 7 giorni, efficienza)

    daily_record: dict             # riga normalizzata del giorno
    insights: list[dict]           # allarmi e avvisi: [{"type", "metric", "label", "description"}]
    metrics: dict                  # per metrica: today, mean, delta, z, status (normal/notable/alert/...)
    trends: list[dict]             # tendenze 7gg: [{"metric", "label", "weekly_change", "r2", "improving"}]
    highlight: str | None          # metrica "punto forte"
    focus: str | None              # metrica più vicina alla soglia negativa
    recommendations: list[str]
    recommendations_source: str    # "regole" o "llm"
    day_summary: str               # sintesi della giornata (solo con LLM)
    weather: dict | None           # riepilogo meteo del giorno
    quote: dict | None             # frase del giorno
    suggestion: dict | None        # film/libro (solo in certi giorni)
    message: str
    news_message: str              # secondo messaggio: il giornale
    weekly: bool                   # forza il riepilogo settimanale anche fuori dal giorno previsto
    weekly_message: str            # riepilogo settimanale (domenica)

    # reducer: ogni nodo può aggiungere errori senza sovrascrivere quelli precedenti
    errors: Annotated[list[str], operator.add]
