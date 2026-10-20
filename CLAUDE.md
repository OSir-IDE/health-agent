# Health agent — istruzioni per Claude Code

Bot personale di Orfeo (Python/LangGraph) che ogni mattina invia su Telegram un report di salute da dati Garmin
e un giornale di notizie personalizzato. Gira in locale su Windows, lanciato da Task Scheduler ("HealthAgent").
Lingua: italiano (testi del bot, commenti, commit, risposte a Orfeo).

## Comandi

- Test (devono passare sempre prima di ogni commit): `python -m pytest tests -q`
- Prova senza inviare: `python main.py --dry-run` (aggiungi `--weekly` per il settimanale)
- Invio reale (solo se Orfeo lo chiede): `python main.py --force`
- Altro: `--backfill N` (storico Garmin), `--backfill-runs N` (corse), `tools/esplora_corsa.py`, `tools/diagnostica_notizie.py`
- Windows/PowerShell: usare `.venv\Scripts\python`; niente comandi bash.

## Architettura (health_agent/)

Grafo LangGraph in `graph.py`: ingestion → normalization → analysis → running → extras → recommendation →
notifier → news → weekly. Un nodo che fallisce non blocca il resto: aggiunge un errore a `state["errors"]`.

- `nodes/analysis.py`: baseline a 7 giorni, z-score, tendenze, soglia sonno (`SLEEP_MIN_HOURS`).
- `nodes/recommendation.py` + `llm.py`: i consigli a regole sono sempre calcolati; l'LLM locale (Ollama) li riformula.
- `news.py`, `nodes/news.py`: RSS → embedding bge-m3 → clustering (average linkage) → selezione; sintesi e "filo del giorno".
- `grounding.py`: controlli deterministici sulle frasi dell'LLM (termini delicati, direzione); nomi propri solo nel log.
- `running.py`: corse da Garmin in `data/runs.csv`; `weekly.py`: grafici e riepilogo della domenica.
- `feedback.py`: pulsanti Telegram (spenti di default, `FEEDBACK_BUTTONS`); registra comunque le storie inviate.

## Principi da non rompere

1. **L'analisi è deterministica; l'LLM riformula soltanto.** Ogni testo dell'LLM passa da validazione (forma, cifre,
   fonti citate, grounding) e ha un ripiego (regole o titolo del feed). Mai far decidere un fatto all'LLM.
2. **Meglio nessun numero che un numero falso.** Sotto le soglie di dati il bot mostra i fatti, non tendenze inventate.
3. **Idempotenza:** `data/last_report.txt` impedisce reinvii; i giri successivi del task leggono solo i comandi Telegram.
4. Il bot non dà indicazioni mediche: solo stile di vita (vedi `FORBIDDEN` in `llm.py`).

## Regole di lavoro

- **Mai leggere, stampare o committare `.env`, `data/`, `~/.garminconnect`.** Contengono token e dati di salute.
- Prima di inviare messaggi reali su Telegram o modificare il task di Windows, chiedere a Orfeo.
- Un commit per modifica coerente, test verdi, messaggio in italiano. `git add -A` (i file nuovi contano).
- Orfeo mette in discussione le scelte metodologiche: spiegare il perché, mostrare alternative e numeri, non imporre.
- Ogni metrica nuova va misurata su dati reali prima di fidarsene; distinguere sempre dati di sviluppo e di prova.

## Aperto / prossimi passi

- **Verificatore delle frasi** (`rag_verifier` in `C:\Users\siria\Documents\AI-Frontiers-Lab`): non copre le notizie così
  com'è (solo ancore: punteggi X/Y, k-range, KiB, tokens, polarità; inglese; Qwen richiede CUDA a 4 bit).
  Piano: insieme di prova da `data/news_llm_log.jsonl` (30-40 frasi etichettate da Orfeo), poi giudice sulle frasi
  (gemma3:12b via Ollama o Qwen2.5-7B 4 bit). GPU: RTX 5070 Laptop 8 GB: gemma e Qwen non stanno insieme, caricarli
  uno alla volta; il verificatore in un processo separato nel venv di AI Frontiers. Limite noto: inversione dei ruoli.
- Soglie del giornale (`NEWS_EMB_SIMILARITY` 0.65, `NEWS_EMB_DIVERSITY` 0.60) tarate su una sola mattina: ricontrollare.
- Report a PC spento: Garmin e meteo nel cloud (GitHub Actions), giornale e LLM in locale. Non ancora iniziato.
- Valutazione alla cieca consigli LLM contro regole da `data/llm_log.jsonl`.
