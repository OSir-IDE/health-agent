"""Nodo 3b — Corsa: aggiorna l'archivio delle corse e prepara la sintesi per report e consigli."""
from .. import running
from ..state import HealthState


def running_node(state: HealthState) -> dict:
    try:
        runs = running.save_runs(state.get("runs_raw") or [])
        return {"running": running.summarize(runs, state["run_date"])}
    except Exception as e:
        return {"running": None, "errors": [f"corsa: {type(e).__name__}: {e}"]}
