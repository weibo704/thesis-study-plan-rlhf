"""Stage 2 -- parse.

Reads each raw CSV under a detected encoding, repairs mojibake, and writes a
typed staging parquet that preserves the source column names.

Encoding is handled explicitly rather than left to pandas' default because
the Coursera 2021 export is not clean UTF-8: non-ASCII institution names and
curly apostrophes arrive corrupted. Silently accepting replacement characters
would corrupt dedup and embeddings, and would do so *non-uniformly* -- the
damage concentrates in non-English institutions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import hashing, paths
from .textnorm import repair_text

# Ordered by strictness. utf-8 first so a clean file is read losslessly;
# cp1252 before latin-1 because it is the likelier Windows-export origin and
# decodes the smart-quote range that latin-1 maps to control characters.
ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

REPLACEMENT = "�"


def detect_encoding(path: Path) -> Tuple[str, bool]:
    """Return (encoding, decoded_cleanly).

    The first encoding that decodes WITHOUT RAISING wins. An earlier version
    also required the decoded text to contain no U+FFFD, which was actively
    harmful: the Coursera 2021 export is valid UTF-8 that already contains
    baked-in U+FFFD from the original scrape. Rejecting UTF-8 on that basis
    fell through to cp1252, which cannot raise (every byte maps to some
    character) and so "succeeded" by mis-decoding the whole file. ftfy then
    undid most of the damage, so the output looked right while the declared
    encoding was wrong.

    Pre-existing U+FFFD is a data-quality fact to report, not a reason to
    try a different codec -- the underlying bytes are already lost.
    """
    raw = path.read_bytes()
    for enc in ENCODINGS:
        try:
            raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        return enc, True
    return "latin-1", False


def parse_source(source_id: str) -> Dict[str, object]:
    src = paths.raw_file(source_id)
    if not src.exists():
        raise FileNotFoundError(f"{source_id}: missing {src}")

    encoding, lossless = detect_encoding(src)
    df = pd.read_csv(src, encoding=encoding, dtype=str, keep_default_na=False,
                     na_values=[""], low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    repaired_cells = 0
    replacement_cells = 0
    for col in df.columns:
        if df[col].dtype != object:
            continue
        original = df[col]
        fixed = original.map(repair_text)
        changed = (original.fillna("") != fixed.fillna("")).sum()
        repaired_cells += int(changed)
        # Count U+FFFD AFTER repair. Counting before is misleading: ftfy can
        # turn a mojibake sequence back INTO a replacement char, so the
        # pre-repair count understates the damage that actually lands in the
        # catalog. These characters are unrecoverable -- the bytes are gone.
        replacement_cells += int(
            fixed.fillna("").astype(str).str.contains(REPLACEMENT, regex=False).sum()
        )
        df[col] = fixed

    df.insert(0, "_source_id", source_id)
    df.insert(1, "_row_index", range(len(df)))

    out = paths.staging_file(source_id)
    hashing.write_parquet(df, out, stage="parse", encoding=encoding)

    return {
        "source_id": source_id,
        "rows": len(df),
        "columns": len(df.columns) - 2,
        "encoding": encoding,
        "lossless_decode": lossless,
        "cells_repaired": repaired_cells,
        "cells_with_replacement_char": replacement_cells,
    }


def run(source_ids: Optional[List[str]] = None) -> pd.DataFrame:
    paths.ensure_dirs()
    ids = source_ids or [
        sid for sid in paths.enabled_sources() if paths.raw_file(sid).exists()
    ]
    if not ids:
        raise SystemExit("[parse] no raw files present; run stage 1 first")

    report = pd.DataFrame([parse_source(sid) for sid in ids])
    path = paths.REPORTS / "encoding_report.csv"
    report.to_csv(path, index=False)
    hashing.record_stage("parse", sources=ids)

    for row in report.itertuples():
        flag = "" if row.lossless_decode else "  <-- LOSSY DECODE"
        print(
            f"[parse] {row.source_id:14} {row.rows:>6} rows  "
            f"{row.encoding:<10} repaired={row.cells_repaired:<6} "
            f"replacement_chars={row.cells_with_replacement_char}{flag}"
        )
    print(f"[parse] wrote {paths.rel(path)}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: parse and repair encoding")
    ap.add_argument("--only", nargs="*")
    ap.parse_args()
    run(ap.parse_args().only)


if __name__ == "__main__":
    main()
