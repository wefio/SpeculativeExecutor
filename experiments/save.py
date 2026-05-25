"""Save experiment raw records with timestamp."""
import os
import json
from datetime import datetime


EXPERIMENT_DIR = os.path.dirname(__file__)


def save(name: str, data: dict) -> str:
    """Save raw experiment data. Returns filepath."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(EXPERIMENT_DIR, f"{name}_{ts}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: experiments/{name}_{ts}.json")
    return filepath


def list_experiments(name: str = None) -> list[str]:
    """List experiment files, optionally filtered by name prefix."""
    if not os.path.exists(EXPERIMENT_DIR):
        return []
    files = sorted(os.listdir(EXPERIMENT_DIR))
    if name:
        files = [f for f in files if f.startswith(name)]
    return [f for f in files if f.endswith(".json")]
