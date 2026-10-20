"""Registro minimale delle esecuzioni (utile perché con pythonw non si vede il terminale)."""
import os
from datetime import datetime
from pathlib import Path

MAX_LINES = 500


def log(job: str, message: str) -> None:
    path = Path(os.getenv("RUN_LOG_PATH", "data/run.log"))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines.append(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {job:<8} {message}")
    path.write_text("\n".join(lines[-MAX_LINES:]) + "\n", encoding="utf-8")


def beat(outcome: str, keep: int = 120) -> None:
    """Un rigo per OGNI avvio del task, anche quando non fa nulla: mostra se Windows lo lancia e cosa decide lo script.

    data/heartbeat.txt (ultimi `keep` righe). Esiti: presto, gia-inviato, avvio-report, fine-ok, fine-errori, crash.
    """
    import os
    from datetime import datetime
    path = Path(os.getenv("HEARTBEAT_PATH", "data/heartbeat.txt"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        lines.append(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {outcome}")
        path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")
    except Exception:
        pass                              # il registro non deve mai rompere il report
