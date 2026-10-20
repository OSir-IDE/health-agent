"""Nodo 1 — Ingestion: raccoglie dati Garmin e comandi Telegram.

Ogni record corrisponde alla MATTINA del report (run_date = D):
  - blocco notte  -> endpoint Garmin con data D   (sonno, HRV, Body Battery al risveglio)
  - blocco giorno -> endpoint Garmin con data D-1 (stress, passi, attività di ieri)

Percorsi dei campi verificati su un campione reale (v. notebooks/01_esplorazione_garmin.ipynb).
"""
import json
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from garminconnect import Garmin

from .. import running
from ..state import HealthState
from ..telegram_logs import sync_updates

TOKENSTORE = "~/.garminconnect"

# Endpoint chiamati: il formato del dict restituito è lo stesso del campione salvato dal notebook
ENDPOINTS = {
    "sleep":        lambda api, d: api.get_sleep_data(d),
    "hrv":          lambda api, d: api.get_hrv_data(d),
    "stats":        lambda api, d: api.get_stats(d),
    "body_battery": lambda api, d: api.get_body_battery(d, d),
}


def _dig(obj, *path):
    """Naviga dict/list annidati; ritorna None se il percorso non esiste."""
    for p in path:
        try:
            obj = obj[p]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


def _to_hours(seconds):
    return round(seconds / 3600, 2) if seconds else None


def _to_minutes(seconds):
    return round(seconds / 60) if seconds else None


@lru_cache(maxsize=1)
def get_api() -> Garmin:
    """Login una sola volta per processo (utile nel backfill). Usa i token salvati."""
    api = Garmin(os.getenv("GARMIN_EMAIL"), os.getenv("GARMIN_PASSWORD"))
    api.login(tokenstore=TOKENSTORE)
    return api


def fetch_garmin_raw(api: Garmin, run_date: str) -> tuple[dict, list[str]]:
    """Chiama ogni endpoint separatamente: se uno fallisce, gli altri arrivano comunque."""
    raw, errors = {}, []
    for name, call in ENDPOINTS.items():
        try:
            raw[name] = call(api, run_date)
        except Exception as e:
            raw[name] = None
            errors.append(f"garmin.{name}: {type(e).__name__}: {e}")
    return raw, errors


def extract_night(raw: dict) -> dict:
    """Blocco NOTTE: dati Garmin della data D = la notte che termina la mattina di D."""
    dto = _dig(raw, "sleep", "dailySleepDTO") or {}
    hrv = _dig(raw, "hrv", "hrvSummary") or {}
    stats = raw.get("stats") or {}
    return {
        "sleep_hours": _to_hours(dto.get("sleepTimeSeconds")),
        "deep_sleep_min": _to_minutes(dto.get("deepSleepSeconds")),
        "rem_sleep_min": _to_minutes(dto.get("remSleepSeconds")),
        "sleep_score": _dig(dto, "sleepScores", "overall", "value"),
        "hrv_last_night": hrv.get("lastNightAvg"),
        "hrv_weekly_avg": hrv.get("weeklyAvg"),
        "hrv_status": hrv.get("status"),
        # FC a riposo misurata durante il sonno; fallback sul valore giornaliero
        "resting_hr": _dig(raw, "sleep", "restingHeartRate") or stats.get("restingHeartRate"),
        "body_battery_at_wake": stats.get("bodyBatteryAtWakeTime"),
    }


def extract_day(raw: dict) -> dict:
    """Blocco GIORNO: dati Garmin della data D-1 = la giornata di ieri, completa."""
    stats = raw.get("stats") or {}
    bb0 = _dig(raw, "body_battery", 0) or {}
    return {
        "stress_avg": stats.get("averageStressLevel"),
        "body_battery_max": stats.get("bodyBatteryHighestValue"),
        "body_battery_min": stats.get("bodyBatteryLowestValue"),
        "bb_charged": bb0.get("charged"),
        "bb_drained": bb0.get("drained"),
        "steps": stats.get("totalSteps"),
        "active_kcal": stats.get("activeKilocalories"),
        "moderate_intensity_min": stats.get("moderateIntensityMinutes"),
        "vigorous_intensity_min": stats.get("vigorousIntensityMinutes"),
    }


def build_record(raw_night: dict, raw_day: dict) -> dict:
    """Record della mattina D: notte da raw(D) + giornata da raw(D-1)."""
    return {**extract_night(raw_night or {}), **extract_day(raw_day or {})}


def previous_day(d: str) -> str:
    return str(date.fromisoformat(d) - timedelta(days=1))


def fetch_telegram_commands() -> list[tuple]:
    """Legge i comandi #film/#libro inviati al bot."""
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    return sync_updates(token, chat_id) if token and chat_id else []


def ingestion_node(state: HealthState) -> dict:
    run_date = state["run_date"]          # mattina del report
    prev = previous_day(run_date)         # giornata di ieri
    errors = []

    try:
        if state.get("sample_path"):
            # modalità sviluppo: lo stesso file alimenta entrambi i blocchi
            with open(state["sample_path"], encoding="utf-8") as f:
                raw_night = raw_day = json.load(f)
        else:
            api = get_api()
            raw_night, e1 = fetch_garmin_raw(api, run_date)
            raw_day, e2 = fetch_garmin_raw(api, prev)
            errors += e1 + e2
        fields = build_record(raw_night, raw_day)
        garmin = fields if any(v is not None for v in fields.values()) else {}
    except Exception as e:
        garmin = {}
        errors.append(f"garmin: {type(e).__name__}: {e}")

    runs = []
    if not state.get("sample_path") and os.getenv("RUNNING_ENABLED", "true").strip().lower() in ("1", "true", "si", "sì", "yes"):
        try:
            runs = running.fetch_runs(get_api(), str(date.fromisoformat(run_date) - timedelta(days=8)), run_date)
        except Exception as e:
            errors.append(f"garmin.corse: {type(e).__name__}: {e}")

    try:
        added = fetch_telegram_commands()
    except Exception as e:
        added = []
        errors.append(f"comandi_bot: {type(e).__name__}: {e}")

    return {"garmin_raw": garmin, "runs_raw": runs, "log_added": added, "errors": errors}
