"""Quali dati di corsa fornisce davvero il tuo Garmin? Esegui: python tools/esplora_corsa.py [giorni]

Stampa tipi di attività, quante corse hanno ciascun campo e una corsa di esempio completa
(salvata anche in data/corsa_esempio.json, senza dati personali di profilo). Serve per tarare
health_agent/running.py sui campi che il tuo account espone.
"""
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(override=True)
from health_agent import running                      # noqa: E402
from health_agent.nodes.ingestion import get_api      # noqa: E402

days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
api = get_api()
acts = api.get_activities_by_date(str(date.today() - timedelta(days=days)), str(date.today()))
types = Counter(((a.get("activityType") or {}).get("typeKey") or "?") for a in acts)
print(f"{len(acts)} attività negli ultimi {days} giorni:", dict(types))

runs = [a for a in acts if "running" in (((a.get("activityType") or {}).get("typeKey")) or "")]
print(f"\n{len(runs)} corse. Campi presenti (n. corse con valore non vuoto), solo quelli usati o utili:")
interesting = ["distance", "duration", "averageHR", "maxHR", "averageRunningCadenceInStepsPerMinute",
               "aerobicTrainingEffect", "anaerobicTrainingEffect", "activityTrainingLoad", "vO2MaxValue",
               "elevationGain", "avgStrideLength", "avgGroundContactTime", "avgVerticalOscillation", "avgPower",
               "hrTimeInZone_1", "hrTimeInZone_2", "hrTimeInZone_3", "trainingEffectLabel", "differenceBodyBattery"]
for k in interesting:
    n = sum(1 for a in runs if a.get(k) not in (None, "", 0))
    print(f"  {k:<42}{n:>3}/{len(runs)}")

print("\nCorse lette da health_agent/running.py:")
rows = running.fetch_runs(type("A", (), {"get_activities_by_date": lambda self, s, e: acts})(), "", "")
for r in rows[:10]:
    print(" ", {k: r[k] for k in ("date", "km", "minutes", "avg_hr", "aerobic_te", "load", "vo2max")})

if runs:
    keep = {k: v for k, v in runs[0].items() if not isinstance(v, (dict, list)) and "owner" not in k.lower()
            and "userProfile" not in k and k not in ("ownerFullName", "ownerDisplayName", "deviceId")}
    out = Path("data/corsa_esempio.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(keep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nEsempio salvato in {out}")
