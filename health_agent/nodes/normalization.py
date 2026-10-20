"""Nodo 2 — Normalization: unisce le fonti in una riga giornaliera e aggiorna lo storico."""
from pathlib import Path

import pandas as pd

from ..state import HealthState


def normalization_node(state: HealthState) -> dict:
    record = {"date": state["run_date"], **state.get("garmin_raw", {})}


    path = Path(state["history_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    hist = hist.drop_duplicates(subset="date", keep="last").sort_values("date")
    hist.to_csv(path, index=False)

    return {"daily_record": record}
