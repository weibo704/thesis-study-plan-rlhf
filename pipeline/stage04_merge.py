"""Stage 4 -- fill gaps in the spine from donor sources.

A donor NEVER creates a course row. It only fills fields the spine lacks,
and every donated value is recorded in `field_provenance` -- canonical rows
are merges of several snapshots with no shared identifier, so "where did
this level come from" has to be answerable.

Two donors, ordered by match precision:

  coursera_hours  joined on url_slug EXACTLY. The only source of real effort
                  hours for Coursera. 57.4% slug overlap with the spine.
  coursera2025    joined on normalised title + institution. Lower precision
                  and only ~20% overlap, because the 2021 and 2024 snapshots
                  genuinely share about a quarter of their courses.

The slug donor is applied first so the higher-precision source wins any
contested field.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd
import yaml

from . import hashing, paths
from .textnorm import merge_key

SPINE_SOURCES = ("coursera2021", "edx")
DONORS = (
    ("coursera_hours", "url_slug"),
    ("coursera2025", "title_institution"),
)

DURATION_PAYLOAD = ("duration_lower_bound", "duration_upper_bound",
                    "calendar_weeks_lo", "calendar_weeks_hi", "effort_hours_total")
SCALAR_GAPS = ("content_granularity", "rating_count", "subject_raw")
DECLARED = ("declared", "declared_vocab_mapped")


def _cfg() -> Dict[str, Any]:
    with open(paths.CONFIG / "thresholds.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["merge"]


def _isnull(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and value != value


def _inst_key(value: Any) -> str:
    return merge_key(value) if value else ""


def _note(provenance: List[Dict[str, Any]], course_id: str, field: str, donor_id: str) -> None:
    provenance.append({"course_id": course_id, "field_name": field,
                       "source_id": donor_id, "transform": "donor_fill"})


def _donate(rec: Dict[str, Any], cand: Dict[str, Any], donor_id: str,
            counts: Dict[str, int], provenance: List[Dict[str, Any]]) -> None:
    """Fill genuine gaps in one spine row from one donor record."""
    # --- difficulty ---------------------------------------------------
    # A DECLARED donor level beats a spine level that stage 3 only inferred
    # from wording; otherwise text inference permanently shadows real data.
    donor_level = cand.get("level_canonical")
    if not _isnull(donor_level):
        spine_declared = rec.get("level_source") in DECLARED
        donor_declared = cand.get("level_source") in DECLARED
        if _isnull(rec.get("level_canonical")) or (donor_declared and not spine_declared):
            rec["level_canonical"] = donor_level
            rec["level_source"] = cand.get("level_source") or "declared"
            rec["level_is_imputed"] = bool(cand.get("level_is_imputed"))
            rec["level_raw"] = cand.get("level_raw")
            counts["level_canonical"] = counts.get("level_canonical", 0) + 1
            _note(provenance, rec["course_id"], "level_canonical", donor_id)

    # --- duration -----------------------------------------------------
    # Donated as one unit: the metadata (source, unit, imputed flag) must
    # travel with whichever payload arrived. Keying it off effort hours
    # alone loses every calendar-only bucket, which by design carries none.
    spine_has = any(not _isnull(rec.get(f)) for f in DURATION_PAYLOAD)
    donor_has = any(not _isnull(cand.get(f)) for f in DURATION_PAYLOAD)
    if donor_has and not spine_has:
        for field in DURATION_PAYLOAD:
            if not _isnull(cand.get(field)):
                rec[field] = cand.get(field)
                counts[field] = counts.get(field, 0) + 1
        rec["duration_source"] = cand.get("duration_source") or "bucket"
        rec["duration_is_imputed"] = bool(cand.get("duration_is_imputed"))
        rec["duration_unit_original"] = cand.get("duration_unit_original")
        _note(provenance, rec["course_id"], "duration", donor_id)

    # --- clean delimited skills ---------------------------------------
    # Both donors carry parseable skill lists; the 2021 spine carries an
    # undelimited blob. This is the highest-value field transfer available.
    if _isnull(rec.get("raw_skills")) and not _isnull(cand.get("raw_skills")):
        rec["raw_skills"] = cand.get("raw_skills")
        rec["skills_delimited"] = True
        counts["raw_skills"] = counts.get("raw_skills", 0) + 1
        _note(provenance, rec["course_id"], "raw_skills", donor_id)

    # --- remaining scalar gaps ----------------------------------------
    for field in SCALAR_GAPS:
        if _isnull(cand.get(field)) or not _isnull(rec.get(field)):
            continue
        rec[field] = cand.get(field)
        counts[field] = counts.get(field, 0) + 1
        _note(provenance, rec["course_id"], field, donor_id)


def _donor_index(frame: pd.DataFrame, strategy: str) -> Dict[Any, Dict[str, Any]]:
    frame = frame.copy()
    frame["_completeness"] = frame.notna().sum(axis=1)
    if strategy == "url_slug":
        frame = frame[frame["url_slug"].notna()]
        keys = ["url_slug"]
    else:
        frame["_inst"] = frame["institution"].map(_inst_key)
        keys = ["title_key", "_inst"]
    # Keep the most complete record per key; ties broken deterministically.
    frame = (frame.sort_values(["_completeness", "course_id"], ascending=[False, True])
                  .drop_duplicates(subset=keys, keep="first"))
    out: Dict[Any, Dict[str, Any]] = {}
    for r in frame.to_dict("records"):
        key = r["url_slug"] if strategy == "url_slug" else (r["title_key"], r["_inst"])
        out[key] = r
    return out


def run() -> pd.DataFrame:
    _cfg()  # validates the config is present/parseable
    norm = pd.read_parquet(paths.STAGING / "normalized.parquet")
    spine = norm[norm["source_id"].isin(SPINE_SOURCES)].copy().reset_index(drop=True)

    provenance: List[Dict[str, Any]] = []
    for rec in spine.to_dict("records"):
        for col in ("title_raw", "description", "institution", "level_canonical",
                    "duration_source", "rating_mean", "url"):
            if not _isnull(rec.get(col)):
                provenance.append({"course_id": rec["course_id"], "field_name": col,
                                   "source_id": rec["source_id"], "transform": "adapter"})

    counts: Dict[str, int] = {}
    per_donor = {sid: 0 for sid, _ in DONORS}
    indexes = {sid: _donor_index(norm[norm["source_id"] == sid], strat)
               for sid, strat in DONORS}
    strategies = dict(DONORS)

    spine["_inst"] = spine["institution"].map(_inst_key)
    records = spine.to_dict("records")
    for rec in records:
        if rec["source_id"] != "coursera2021":
            continue  # donors are Coursera-only; never cross platforms
        for donor_id, _ in DONORS:
            strat = strategies[donor_id]
            key = rec["url_slug"] if strat == "url_slug" else (rec["title_key"], rec["_inst"])
            if key is None or (isinstance(key, float) and key != key):
                continue
            cand = indexes[donor_id].get(key)
            if cand is None:
                continue
            per_donor[donor_id] += 1
            _donate(rec, cand, donor_id, counts, provenance)

    spine = pd.DataFrame(records).drop(columns=["_inst"], errors="ignore")
    n_coursera = int((spine["source_id"] == "coursera2021").sum())

    prov = pd.DataFrame(provenance).drop_duplicates()
    hashing.write_parquet(spine, paths.STAGING / "merged.parquet", stage="merge")
    hashing.write_parquet(prov, paths.CATALOG / "field_provenance.parquet", stage="merge")

    lines = ["# Donor merge report", "",
             f"- spine rows: **{len(spine):,}** "
             f"(coursera2021={n_coursera:,}, edx={int((spine['source_id']=='edx').sum()):,})",
             "", "| donor | join | spine rows matched | share of coursera2021 |",
             "|---|---|---|---|"]
    for sid, strat in DONORS:
        share = per_donor[sid] / n_coursera if n_coursera else 0
        lines.append(f"| `{sid}` | {strat} | {per_donor[sid]:,} | {share:.1%} |")
    lines += ["", "## Fields donated into a gap", "", "| field | count |", "|---|---|"]
    for field, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{field}` | {count:,} |")
    lines += ["", "A donor never creates a course. A donated value only fills a genuine",
              "null; the spine is never overwritten. The slug-joined donor is applied",
              "first, so the higher-precision source wins any contested field."]
    (paths.REPORTS / "merge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("merge", matched=per_donor, spine_rows=len(spine))
    print(f"[merge] spine={len(spine):,} matched={per_donor}")
    print(f"[merge] provenance rows: {len(prov):,}")
    return spine


def main() -> None:
    argparse.ArgumentParser(description="Stage 4: donor merge").parse_args()
    run()


if __name__ == "__main__":
    main()
