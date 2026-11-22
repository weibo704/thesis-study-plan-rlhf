"""Catalog coverage report.

Standalone so the real numbers can be regenerated after any re-ingest.
Stage 11 (validate) will absorb this once it exists.
"""
from __future__ import annotations

import pandas as pd

from . import paths

REPL = "�"


def _pct(series: pd.Series) -> str:
    return f"{series.mean():.1%}"


def run() -> str:
    d = pd.read_parquet(paths.STAGING / "deduped.parquet")
    can = d[d["is_canonical"]].copy()
    src = pd.read_parquet(paths.CATALOG / "sources.parquet")

    lines = [
        "# Catalog coverage report", "",
        f"Dataset version: `{paths.dataset_version()}`", "",
        "## Sources", "",
        "| source | licence | redistributable | records |", "|---|---|---|---|",
    ]
    for r in src.itertuples():
        lines.append(f"| {r.source_id} | {r.license} | {r.redistributable} | {r.row_count:,} |")

    lines += [
        "", "## Arm pool (pre-scope-filter)", "",
        f"- spine rows: **{len(d):,}**",
        f"- canonical courses after dedup: **{len(can):,}**",
        f"- collapsed as duplicates: **{len(d) - len(can):,}**", "",
        "By source:", "",
    ]
    for sid, n in can["source_id"].value_counts().items():
        lines.append(f"- `{sid}`: {n:,}")

    lines += ["", "## Field coverage (canonical rows)", "",
              "| field | coverage |", "|---|---|"]
    for label, series in [
        ("description", can["description"].notna()),
        ("raw skill text", can["raw_skills_text"].notna()),
        ("syllabus", can["syllabus"].notna()),
        ("difficulty level", can["level_canonical"].notna()),
        ("rating", can["rating_mean"].notna()),
        ("effort hours (total)", can["effort_hours_total"].notna()),
        ("any duration signal",
         can[["duration_lower_bound", "calendar_weeks_lo"]].notna().any(axis=1)),
    ]:
        lines.append(f"| {label} | {_pct(series)} |")

    lines += ["", "## Difficulty provenance", "", "| level_source | rows |", "|---|---|"]
    for k, v in can["level_source"].value_counts().items():
        lines.append(f"| `{k}` | {v:,} |")
    dist = can["level_canonical"].value_counts().sort_index()
    lines += ["", "Distribution: " + ", ".join(
        f"{int(k)}={int(v):,}" for k, v in dist.items()) +
        "  (0=beginner, 1=intermediate, 2=advanced)"]

    lines += ["", "## Known data damage", ""]
    for col in ("title_normalized", "description", "institution"):
        dmg = can[col].fillna("").astype(str).str.contains(REPL, regex=False)
        lines.append(f"- `{col}`: **{int(dmg.sum()):,}** rows ({_pct(dmg)}) contain U+FFFD")
    lines += [
        "",
        "These replacement characters are baked into the Coursera 2021 export",
        "at scrape time. The original bytes are gone, so they are unrecoverable",
        "here. They matter because they sit inside description text that feeds",
        "embeddings; treat as a documented limitation.",
    ]

    lines += ["", "## Language", "", "| language | rows |", "|---|---|"]
    for k, v in can["language"].value_counts().head(6).items():
        lines.append(f"| {k} | {v:,} |")

    out = paths.REPORTS / "catalog_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {paths.rel(out)}")
    return "\n".join(lines)


if __name__ == "__main__":
    run()
