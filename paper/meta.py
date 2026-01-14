# paper/meta.py
from __future__ import annotations

import datetime
import subprocess
import sys
from typing import Any, Dict
from dataclasses import asdict, is_dataclass


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def git_info() -> Dict[str, Any]:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = (
            subprocess.call(["git", "diff", "--quiet"]) != 0
            or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0
        )
        return {"hash": h, "dirty": bool(dirty)}
    except Exception:
        return {"hash": None, "dirty": None}


def make_run_id(exp: str, tag: str) -> str:
    gi = git_info()
    short = (gi["hash"] or "nogit")[:8]
    ts = utc_now_iso().replace(":", "-")
    return f"{ts}_{short}_{exp}_{tag}"


def python_info() -> str:
    return sys.version


def spec_to_jsonable(spec: Any) -> Dict[str, Any]:
    """
    JSON-safe dump of ExperimentSpec-like objects (handles nested dataclasses + callables).
    """
    def conv(x: Any) -> Any:
        if is_dataclass(x):
            return {k: conv(v) for k, v in asdict(x).items()}
        if callable(x):
            mod = getattr(x, "__module__", "")
            name = getattr(x, "__qualname__", getattr(x, "__name__", repr(x)))
            return f"{mod}:{name}"
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    return conv(spec)
