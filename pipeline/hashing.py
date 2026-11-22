"""Content hashing and the build manifest.

Every artefact the pipeline writes is recorded with a sha256 so that a
rebuild can be proved identical. This underpins the reproducibility claim
in docs/ingestion-plan.md section 7.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from . import paths

MANIFEST = paths.REPORTS / "manifest.json"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Order-insensitive-by-column, order-sensitive-by-row frame hash.

    Columns are sorted so that column order alone cannot change the hash,
    but row order is preserved deliberately: for matrices aligned to
    arm_index, row order is load-bearing.
    """
    h = hashlib.sha256()
    for col in sorted(df.columns):
        h.update(str(col).encode("utf-8"))
        series = df[col]
        try:
            digest = pd.util.hash_pandas_object(series, index=False).values.tobytes()
        except TypeError:
            # List/dict cells (e.g. raw_skills) are not hashable by pandas;
            # fall back to a stable string rendering.
            digest = series.map(
                lambda v: "|".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
            ).str.encode("utf-8").sum() or b""
            digest = bytes(digest)
        h.update(digest)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load() -> Dict[str, Any]:
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as fh:
            return json.load(fh)
    return {"dataset_version": paths.dataset_version(), "artifacts": {}, "stages": {}}


def _jsonable(value: Any) -> Any:
    """Coerce numpy/pandas scalars so the manifest stays serialisable.

    pandas value_counts() yields numpy int64 keys and values, which json
    refuses. The manifest is the reproducibility record, so it must never
    fail to write because of a dtype detail.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save(man: Dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(man), fh, indent=2, sort_keys=True)
    tmp.replace(MANIFEST)


def record_artifact(
    path: Path,
    rows: Optional[int] = None,
    frame_hash: Optional[str] = None,
    **extra: Any,
) -> None:
    man = _load()
    rel = paths.rel(path)
    entry: Dict[str, Any] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "written_at": now_iso(),
    }
    if rows is not None:
        entry["rows"] = int(rows)
    if frame_hash is not None:
        entry["frame_sha256"] = frame_hash
    entry.update(extra)
    man["artifacts"][rel] = entry
    _save(man)


def record_stage(stage: str, **fields: Any) -> None:
    man = _load()
    man.setdefault("stages", {})[stage] = dict(fields, completed_at=now_iso())
    _save(man)


def write_parquet(df: pd.DataFrame, path: Path, **extra: Any) -> str:
    """Write a frame and register it in the manifest. Returns the frame hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = sha256_frame(df)
    df.to_parquet(path, index=False)
    record_artifact(path, rows=len(df), frame_hash=fh, **extra)
    return fh
