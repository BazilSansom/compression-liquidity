# paper/io.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def rows_to_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    ensure_dir(out_csv.parent)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
