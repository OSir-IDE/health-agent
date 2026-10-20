"""Entry point.

Esempi:
  python main.py --dry-run                         # report di stamattina (notte + ieri), stampato
  python main.py --dry-run --sample data/garmin_sample_2026-09-23.json --date 2026-09-23
  python main.py --backfill 14                     # scarica gli ultimi 14 giorni nello storico
  python main.py --force                           # reinvia anche se oggi è già partito
"""
import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

from dotenv import load_dotenv

from health_agent.graph import build_graph


def too_early(now: datetime) -> bool:
    """Prima di REPORT_NOT_BEFORE (ora, predefinito 7) il giro esce senza fare nulla: i dati Garmin della notte
    non sono ancora sincronizzati e un task ripetuto a oltranza non deve mandare il report alle 3 di notte."""
    return now.hour < int(os.getenv("REPORT_NOT_BEFORE", "7"))


def backfill(days: int, history_path: str) -> None:
    """Ricostruisce lo storico delle ultime N mattine (notte di D + giornata di D-1)."""
    from health_agent.nodes.ingestion import build_record, fetch_garmin_raw, get_api
    from health_agent.nodes.normalization import normalization_node

    api = get_api()
    first = date.today() - timedelta(days=days)
    prev_raw, _ = fetch_garmin_raw(api, str(first - timedelta(days=1)))
    for i in range(days, 0, -1):
        d = str(date.today() - timedelta(days=i))
        raw, errors = fetch_garmin_raw(api, d)
        normalization_node({"run_date": d, "garmin_raw": build_record(raw, prev_raw),
                            "history_path": history_path})
        print(f"{d}  {'ok' if not errors else f'{len(errors)} endpoint in errore'}")
        prev_raw = raw
        time.sleep(1.5)  # gentile con i server Garmin


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):      # emoji anche dentro pipe o file
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(override=True)
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=str(date.today()),
                   help="Mattina del report (default: oggi). Notte di questa data + giornata del giorno prima")
    p.add_argument("--dry-run", action="store_true", help="Stampa invece di inviare")
    p.add_argument("--sample", help="JSON salvato dal notebook: niente chiamate a Garmin")
    p.add_argument("--backfill", type=int, metavar="N", help="Scarica gli ultimi N giorni e termina")
    p.add_argument("--backfill-runs", type=int, metavar="N", help="Scarica le corse degli ultimi N giorni e termina")
    p.add_argument("--weekly", action="store_true", help="Aggiunge il riepilogo settimanale anche se oggi non è domenica")
    p.add_argument("--force", action="store_true", help="Invia anche se il report di oggi è già partito")
    args = p.parse_args()

    history_path = os.getenv("HISTORY_PATH", "data/history.csv")

    if args.backfill_runs:
        from health_agent import running
        from health_agent.nodes.ingestion import get_api
        rows = running.fetch_runs(get_api(), str(date.today() - timedelta(days=args.backfill_runs)), str(date.today()))
        total = running.save_runs(rows)
        print(f"{len(rows)} corse scaricate, {len(total)} in archivio ({running.runs_path()})")
        raise SystemExit

    if args.backfill:
        backfill(args.backfill, history_path)
        raise SystemExit

    state = {"run_date": args.date, "dry_run": args.dry_run,
             "history_path": history_path, "errors": [], "weekly": args.weekly}
    if args.sample:
        state["sample_path"] = args.sample

    from runlog import beat, log

    # Il task prova più volte al giorno: invia solo se il report di oggi non è ancora partito
    from pathlib import Path
    marker = Path("data/last_report.txt")
    if not args.dry_run and not args.force and args.date == str(date.today()) and too_early(datetime.now()):
        beat("presto")
        raise SystemExit
    if not args.dry_run and not args.force and marker.exists() and marker.read_text().strip() == args.date:
        # report già inviato: il giro serve solo a leggere i tocchi sui pulsanti e i comandi #film/#libro
        beat("gia-inviato")
        try:
            from health_agent.telegram_logs import sync_and_confirm
            res = sync_and_confirm()
            if res and (res["votes"] or res["added"]):
                log("sync", f"{len(res['votes'])} voti, {len(res['added'])} comandi")
        except Exception as e:
            log("sync", f"errore {type(e).__name__}: {e}")
        raise SystemExit
    if not args.dry_run:
        beat("avvio-report")
    from health_agent import llm
    if llm.enabled():
        waited = llm.wait_for_ollama()                  # dopo accensione/risveglio Ollama può non essere ancora pronto
        if waited is None:
            log("ollama", "non risponde dopo l'attesa: si va con i ripieghi")
        elif waited >= 1:
            log("ollama", f"pronto dopo {waited:.0f}s")
    try:
        result = build_graph().invoke(state)
    except Exception as e:
        # Il grafo si è rotto: prova almeno ad avvisarti su Telegram
        msg = f"{type(e).__name__}: {e}"
        log("report", f"CRASH {msg}")
        beat("crash")
        print("❌ Report non generato:", msg)
        if not args.dry_run:
            try:
                from health_agent.telegram_logs import send_message
                send_message(f"⚠️ Report di oggi non generato.\n{msg[:300]}\n\nDettagli in data/run.log")
            except Exception:
                pass
        raise SystemExit(1)

    # Segna come inviato solo se il report su Telegram non ha dato errore
    if not args.dry_run and not any(str(e).startswith("telegram") for e in result.get("errors", [])):
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(args.date)

    errs = result.get("errors") or []
    if not args.dry_run:
        beat("fine-errori" if errs else "fine-ok")
    source = result.get("recommendations_source", "regole")
    log("report", f"ok (consigli: {source})" + (f" | {len(errs)} avvisi: {'; '.join(errs)[:300]}" if errs else ""))
    if errs:
        print("⚠️ Errori:", errs)
