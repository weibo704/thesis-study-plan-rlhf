"""Stage 1 -- fetch.

Downloads the source datasets and records provenance: sha256, row count,
licence, terms URL and retrieval date. Nothing downstream may read a source
that is not registered here, and the `redistributable` flag recorded in this
stage gates the export step.

Kaggle credentials are optional. If the API is unavailable the stage falls
back to whatever has been placed in data/raw/<source_id>/ by hand and tells
the user exactly what is missing, so the pipeline is runnable either way.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from . import hashing, paths

DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{ref}"
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"


def _resolve_token() -> Optional[str]:
    """Find a KGAT-style Kaggle API token.

    Kaggle's newer tokens (KGAT_...) are bearer tokens and are NOT understood
    by the kaggle python client, which only does basic auth with a
    username/key pair from kaggle.json. Since the bearer token works
    directly against the REST API, prefer it and skip the client entirely.
    """
    env = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env:
        return env
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    return None


def _download_via_token(ref: str, dest_dir: Path, token: str) -> Tuple[bool, str]:
    """Download and unzip a dataset using bearer auth.

    The token is only ever placed in a request header; it is never logged,
    echoed, or written into any artefact.
    """
    try:
        resp = requests.get(
            DOWNLOAD_URL.format(ref=ref),
            headers={"Authorization": f"Bearer {token}"},
            timeout=180,
            allow_redirects=True,
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"request failed: {type(exc).__name__}"

    if resp.status_code == 401:
        return False, "token rejected (401)"
    if resp.status_code == 403:
        return False, "token lacks access (403) - accept the dataset terms on Kaggle"
    if resp.status_code == 404:
        return False, "dataset not found (404)"
    if resp.status_code != 200:
        return False, f"http {resp.status_code}"

    payload = resp.content
    if not payload:
        return False, "empty response"

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(dest_dir)
        return True, "downloaded"
    except zipfile.BadZipFile:
        # A single-file dataset can come back raw rather than zipped.
        ctype = resp.headers.get("content-type", "")
        if "csv" in ctype or payload[:1] not in (b"P", b"<"):
            (dest_dir / "download.csv").write_bytes(payload)
            return True, "downloaded (raw csv)"
        return False, f"not a zip (content-type={ctype[:40]})"


def _kaggle_api():
    """Import and authenticate Kaggle lazily.

    The kaggle package authenticates at import time and raises OSError when
    no credentials exist, so this must never be a module-level import.
    """
    try:
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:  # noqa: BLE001 - any failure means "no API"
        return exc


def _count_rows(path: Path, encoding_hint: str = "utf-8") -> Optional[int]:
    """CSV record count excluding the header.

    Counting file lines is wrong here: descriptions and syllabi contain
    embedded newlines, which inflates a line count by ~3x on the edX export.
    That number lands in the provenance table, so it has to be records.
    """
    for enc in (encoding_hint, "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc, errors="strict", newline="") as fh:
                return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
        except (UnicodeDecodeError, LookupError):
            continue
        except csv.Error:
            break
    with open(path, encoding="latin-1", errors="replace", newline="") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def fetch_source(source_id: str, spec: Dict, api, token: Optional[str]) -> Tuple[bool, str]:
    dest_dir = paths.raw_dir(source_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = paths.raw_file(source_id)

    if target.exists():
        return True, "already present"

    status = "no auth available"
    if token:
        ok, status = _download_via_token(spec["kaggle_ref"], dest_dir, token)
    elif not isinstance(api, Exception):
        try:
            api.dataset_download_files(spec["kaggle_ref"], path=str(dest_dir), unzip=True)
            ok, status = True, "downloaded"
        except Exception as exc:  # noqa: BLE001
            ok, status = False, f"download failed: {type(exc).__name__}"
    else:
        ok = False
        status = f"kaggle client unavailable ({type(api).__name__})"

    if target.exists():
        return True, status
    found = sorted(p.name for p in dest_dir.glob("*.csv"))
    if ok and found:
        return False, f"expected {spec['filename']!r}, found {found}"
    return False, status


def run(source_ids: Optional[List[str]] = None) -> pd.DataFrame:
    paths.ensure_dirs()
    sources = paths.enabled_sources()
    if source_ids:
        sources = {k: v for k, v in sources.items() if k in source_ids}

    token = _resolve_token()
    api = None if token else _kaggle_api()
    if token:
        print("[fetch] using Kaggle API token (bearer auth)")
    elif isinstance(api, Exception):
        print(f"[fetch] Kaggle client unavailable: {type(api).__name__}", file=sys.stderr)

    rows, missing = [], []
    for sid, spec in sources.items():
        ok, status = fetch_source(sid, spec, api, token)
        target = paths.raw_file(sid)
        print(f"[fetch] {sid:14} {status}")
        if not ok:
            missing.append((sid, spec, status))
            continue
        rows.append(
            {
                "source_id": sid,
                "name": spec["name"],
                "url": spec["url"],
                "platform": spec["platform"],
                "license": spec["license"],
                "terms_url": spec["terms_url"],
                "redistributable": bool(spec["redistributable"]),
                "role": spec["role"],
                "snapshot": str(spec.get("snapshot", "")),
                "filename": spec["filename"],
                "file_sha256": hashing.sha256_file(target),
                "row_count": _count_rows(target),
                "retrieved_at": hashing.now_iso(),
                "notes": " ".join(str(spec.get("notes", "")).split()),
            }
        )

    if missing:
        print("\n[fetch] Missing sources. Provide auth via either:")
        print("    ~/.kaggle/access_token   (KGAT_... bearer token)")
        print("    ~/.kaggle/kaggle.json    (classic username/key pair)")
        print("  or download by hand and place each CSV at the path shown:\n")
        for sid, spec, status in missing:
            print(f"    {spec['url']}")
            print(f"      -> {paths.rel(paths.raw_file(sid))}   [{status}]")
        print()

    if not rows:
        raise SystemExit("[fetch] no sources available; nothing to record")

    df = pd.DataFrame(rows).sort_values("source_id").reset_index(drop=True)
    out = paths.CATALOG / "sources.parquet"
    hashing.write_parquet(df, out, stage="fetch")
    hashing.record_stage(
        "fetch",
        sources_ok=list(df["source_id"]),
        sources_missing=[m[0] for m in missing],
    )
    print(f"[fetch] wrote {paths.rel(out)} ({len(df)} sources)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: fetch source datasets")
    ap.add_argument("--only", nargs="*", help="restrict to these source ids")
    args = ap.parse_args()
    df = run(args.only)
    cols = ["source_id", "license", "redistributable", "row_count", "file_sha256"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 20):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
