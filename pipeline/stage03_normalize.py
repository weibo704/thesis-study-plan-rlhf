"""Stage 3 -- normalize.

Maps each source's idiosyncratic columns onto the canonical `courses`
schema. One adapter per source; everything they share lives in parsers.py.

Design rules carried from docs/ingestion-plan.md section 4:
  * every imputed value carries a *_is_imputed flag and a *_source label
  * sentinels become nulls, never imputed modes
  * effort hours and calendar weeks never collapse into one another
  * raw skill text is preserved verbatim; tokenising it is stage 6's job
"""
from __future__ import annotations

import argparse
import hashlib
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from . import hashing, paths, parsers
from .textnorm import merge_key, normalize_title, repair_text, url_slug

CANONICAL_COLUMNS: List[str] = [
    "course_id", "source_id", "platform", "url", "url_slug", "id_strategy",
    "title_raw", "title_normalized", "title_key",
    "description", "syllabus", "institution",
    "content_granularity",
    "level_raw", "level_canonical", "level_source", "level_is_imputed",
    "effort_hours_total", "duration_lower_bound", "duration_upper_bound",
    "calendar_weeks_lo", "calendar_weeks_hi",
    "duration_unit_original", "duration_source", "duration_is_imputed",
    "access_model", "price_usd", "certification",
    "rating_mean", "rating_count", "rating_is_missing", "enrollment_count",
    "language", "language_is_imputed", "pace",
    "subject_raw", "raw_skills_text", "raw_skills", "skills_delimited",
    "category_slugs", "skills_body",
    "program_membership",
    "dataset_version",
]


# Explicit dtypes: without them, a source whose column is entirely null
# infers a different dtype than the same column elsewhere, and the concat
# silently changes types between runs. Nullable extension dtypes are used
# so that "missing" stays distinguishable from 0 / False.
CANONICAL_DTYPES: Dict[str, str] = {
    "course_id": "string", "source_id": "string", "platform": "string",
    "url": "string", "url_slug": "string", "id_strategy": "string",
    "title_raw": "string", "title_normalized": "string", "title_key": "string",
    "description": "string", "syllabus": "string", "institution": "string",
    "content_granularity": "string",
    "level_raw": "string", "level_canonical": "Int8", "level_source": "string",
    "level_is_imputed": "boolean",
    "effort_hours_total": "Float64",
    "duration_lower_bound": "Float64", "duration_upper_bound": "Float64",
    "calendar_weeks_lo": "Float64", "calendar_weeks_hi": "Float64",
    "duration_unit_original": "string", "duration_source": "string",
    "duration_is_imputed": "boolean",
    "access_model": "string", "price_usd": "Float64", "certification": "boolean",
    "rating_mean": "Float64", "rating_count": "Int64",
    "rating_is_missing": "boolean", "enrollment_count": "Int64",
    "language": "string", "language_is_imputed": "boolean", "pace": "string",
    "subject_raw": "string", "raw_skills_text": "string",
    "skills_delimited": "boolean", "program_membership": "string",
    "skills_body": "string",
    "dataset_version": "string",
    # raw_skills stays object: it holds a list per row.
}


def _apply_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in CANONICAL_DTYPES.items():
        if col not in frame.columns:
            continue
        try:
            frame[col] = frame[col].astype(dtype)
        except (TypeError, ValueError):
            frame[col] = pd.array([None] * len(frame), dtype=dtype)
    return frame


def _hash_id(*parts: Optional[str]) -> str:
    blob = "|".join((p or "") for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _base_row(source_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "platform": spec["platform"],
        "dataset_version": paths.dataset_version(),
        "url": None, "url_slug": None, "syllabus": None,
        "content_granularity": None, "certification": None,
        "price_usd": None, "access_model": None,
        "rating_mean": None, "rating_count": None, "rating_is_missing": True,
        "enrollment_count": None, "pace": None, "subject_raw": None,
        "raw_skills_text": None, "raw_skills": None, "skills_delimited": False,
        "category_slugs": None, "skills_body": None,
        "program_membership": None,
        "effort_hours_total": None,
        "duration_lower_bound": None, "duration_upper_bound": None,
        "calendar_weeks_lo": None, "calendar_weeks_hi": None,
        "duration_unit_original": None, "duration_source": "unknown",
        "duration_is_imputed": True,
    }


def _apply_level(row: Dict[str, Any], raw: Optional[str]) -> None:
    level, source, is_container = parsers.parse_level(raw)
    row["level_raw"] = raw
    if is_container and not row.get("content_granularity"):
        # "Mixed" means multi-level container, not a difficulty.
        row["content_granularity"] = "program"
    if level is None:
        level, source = parsers.infer_level_from_text(
            row.get("title_raw"), row.get("description")
        )
    row["level_canonical"] = level
    row["level_source"] = source
    # A remapped declared value is still declared, not imputed.
    row["level_is_imputed"] = source not in ("declared", "declared_vocab_mapped")


def _apply_duration(row: Dict[str, Any], dur: parsers.Duration) -> None:
    row["duration_lower_bound"] = dur.effort_hours_lo
    row["duration_upper_bound"] = dur.effort_hours_hi
    row["calendar_weeks_lo"] = dur.calendar_weeks_lo
    row["calendar_weeks_hi"] = dur.calendar_weeks_hi
    row["duration_unit_original"] = dur.unit_original
    row["duration_source"] = dur.source
    # A point estimate only where the interval is a genuine point.
    if dur.source == "derived_effort_x_weeks" and dur.effort_hours_lo is not None:
        row["effort_hours_total"] = dur.effort_hours_lo
    elif dur.effort_hours_lo is not None and dur.effort_hours_lo == dur.effort_hours_hi:
        row["effort_hours_total"] = dur.effort_hours_lo
    else:
        row["effort_hours_total"] = None
    row["duration_is_imputed"] = dur.source in ("unknown", "bucket")


# Coursera 2021 appends its own category identifiers to the skills blob as
# hyphenated lowercase tokens ("computer-science software-development").
# They are trivially separable and present on 91.8% of rows, which makes
# them the most reliable domain signal in that source.
_SLUG_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b")


def _extract_category_slugs(text: Optional[str]):
    """Return (slugs, body_without_slugs)."""
    if not text:
        return None, None
    slugs = _SLUG_RE.findall(text)
    if not slugs:
        return None, text
    body = " ".join(_SLUG_RE.sub(" ", text).split())
    # Preserve order, drop repeats.
    seen, ordered = set(), []
    for sl in slugs:
        if sl not in seen:
            seen.add(sl); ordered.append(sl)
    return ordered, (body or None)


def _split_skills(text: Optional[str]):
    """Split only when a real delimiter exists.

    Coursera 2021 space-joins multi-word skill phrases with no delimiter
    ("Drama Comedy peering screenwriting film ..."), so splitting on
    whitespace would fabricate tokens. Those rows keep the raw blob and are
    flagged skills_delimited=False for stage 6 to handle.
    """
    if not text:
        return None, False
    for delim in (";", ",", "|"):
        if delim in text:
            parts = [p.strip() for p in text.split(delim)]
            parts = [p for p in parts if p]
            if len(parts) > 1:
                return parts, True
    return None, False


def adapt_coursera2021(df: pd.DataFrame, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for rec in df.to_dict("records"):
        row = _base_row("coursera2021", spec)
        row["title_raw"] = repair_text(rec.get("Course Name"))
        row["title_normalized"] = normalize_title(row["title_raw"])
        row["title_key"] = merge_key(row["title_normalized"])
        row["institution"] = repair_text(rec.get("University"))
        row["description"] = repair_text(rec.get("Course Description"))
        row["url"] = repair_text(rec.get("Course URL"))
        row["url_slug"] = url_slug(row["url"])
        row["id_strategy"] = "url_slug" if row["url_slug"] else "title_institution_hash"
        row["course_id"] = (
            f"coursera2021:{row['url_slug']}" if row["url_slug"]
            else f"coursera2021:h{_hash_id(row['title_key'], row['institution'])}"
        )
        # /learn/ is a course, /specializations/ a container.
        if row["url"] and "/specializations/" in row["url"]:
            row["content_granularity"] = "program"
        elif row["url_slug"]:
            row["content_granularity"] = "course"
        _apply_level(row, repair_text(rec.get("Difficulty Level")))
        rating, missing = parsers.parse_rating(rec.get("Course Rating"))
        row["rating_mean"], row["rating_is_missing"] = rating, missing
        row["raw_skills_text"] = repair_text(rec.get("Skills"))
        row["raw_skills"], row["skills_delimited"] = _split_skills(row["raw_skills_text"])
        row["category_slugs"], row["skills_body"] = _extract_category_slugs(row["raw_skills_text"])
        row["language"], row["language_is_imputed"] = "English", True
        row["access_model"] = "audit_free_cert_paid"
        _apply_duration(row, parsers.EMPTY_DURATION)
        rows.append(row)
    return rows


def adapt_coursera2025(df: pd.DataFrame, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for rec in df.to_dict("records"):
        row = _base_row("coursera2025", spec)
        row["title_raw"] = repair_text(rec.get("Title"))
        row["title_normalized"] = normalize_title(row["title_raw"])
        row["title_key"] = merge_key(row["title_normalized"])
        row["institution"] = repair_text(rec.get("Institution"))
        row["subject_raw"] = repair_text(rec.get("Subject"))
        # No URL column in this export, so the key must be derived.
        row["id_strategy"] = "title_institution_hash"
        row["course_id"] = f"coursera2025:h{_hash_id(row['title_key'], row['institution'])}"
        row["content_granularity"] = parsers.parse_granularity(
            repair_text(rec.get("Learning Product"))
        )
        _apply_level(row, repair_text(rec.get("Level")))
        _apply_duration(row, parsers.parse_duration_bucket(repair_text(rec.get("Duration"))))
        rating, missing = parsers.parse_rating(rec.get("Rate"))
        row["rating_mean"], row["rating_is_missing"] = rating, missing
        row["rating_count"] = parsers.parse_count(rec.get("Reviews"))
        row["raw_skills_text"] = repair_text(rec.get("Gained Skills"))
        row["raw_skills"], row["skills_delimited"] = _split_skills(row["raw_skills_text"])
        row["category_slugs"], row["skills_body"] = _extract_category_slugs(row["raw_skills_text"])
        row["language"], row["language_is_imputed"] = "English", True
        row["access_model"] = "audit_free_cert_paid"
        rows.append(row)
    return rows


def adapt_edx(df: pd.DataFrame, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for rec in df.to_dict("records"):
        row = _base_row("edx", spec)
        row["title_raw"] = repair_text(rec.get("title"))
        row["title_normalized"] = normalize_title(row["title_raw"])
        row["title_key"] = merge_key(row["title_normalized"])
        row["institution"] = repair_text(rec.get("institution"))
        row["subject_raw"] = repair_text(rec.get("subject"))
        row["description"] = repair_text(rec.get("course_description")) or repair_text(rec.get("summary"))
        row["syllabus"] = repair_text(rec.get("course_syllabus"))
        row["url"] = repair_text(rec.get("course_url"))
        row["url_slug"] = url_slug(row["url"])
        row["id_strategy"] = "url_slug" if row["url_slug"] else "title_institution_hash"
        row["course_id"] = (
            f"edx:{row['url_slug']}" if row["url_slug"]
            else f"edx:h{_hash_id(row['title_key'], row['institution'])}"
        )
        row["content_granularity"] = "course"
        # course_type conflates pace with programme membership.
        course_type = repair_text(rec.get("course_type"))
        row["pace"] = parsers.parse_pace(course_type)
        if course_type and "micromasters" in course_type.casefold():
            row["program_membership"] = "MicroMasters"
        _apply_level(row, repair_text(rec.get("Level")))
        _apply_duration(
            row,
            parsers.effort_from_effort_and_length(
                parsers.parse_hours_per_week(rec.get("course_effort")),
                parsers.parse_weeks(rec.get("course_length")),
            ),
        )
        row["enrollment_count"] = parsers.parse_count(rec.get("n_enrolled"))
        price, access = parsers.parse_price(rec.get("price"))
        row["price_usd"] = price
        row["access_model"] = access or "audit_free_cert_paid"
        lang = repair_text(rec.get("language"))
        row["language"], row["language_is_imputed"] = (lang or "English"), lang is None
        rows.append(row)
    return rows


def adapt_coursera_hours(df: pd.DataFrame, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Donor carrying real effort hours and a course URL.

    The URL is what makes this source valuable: it allows an exact slug
    join instead of matching drifting titles across snapshots.
    """
    import ast

    rows = []
    for rec in df.to_dict("records"):
        row = _base_row("coursera_hours", spec)
        row["title_raw"] = repair_text(rec.get("Course Title"))
        row["title_normalized"] = normalize_title(row["title_raw"])
        row["title_key"] = merge_key(row["title_normalized"])
        row["institution"] = repair_text(rec.get("Offered By"))
        row["url"] = repair_text(rec.get("Course Url"))
        row["url_slug"] = url_slug(row["url"])
        row["id_strategy"] = "url_slug" if row["url_slug"] else "title_institution_hash"
        row["course_id"] = (
            f"coursera_hours:{row['url_slug']}" if row["url_slug"]
            else f"coursera_hours:h{_hash_id(row['title_key'], row['institution'])}"
        )
        row["content_granularity"] = parsers.parse_granularity(
            repair_text(rec.get("Course Type"))) or "course"
        _apply_level(row, repair_text(rec.get("Level")))
        _apply_duration(row, parsers.parse_duration_bucket(repair_text(rec.get("Duration"))))
        rating, missing = parsers.parse_rating(rec.get("Rating"))
        row["rating_mean"], row["rating_is_missing"] = rating, missing
        row["rating_count"] = parsers.parse_count(rec.get("Review"))
        # "Skill gain" is a python-literal list, so it parses cleanly --
        # unlike the 2021 spine's undelimited blob.
        raw = repair_text(rec.get("Skill gain"))
        if raw:
            try:
                items = ast.literal_eval(raw)
                if isinstance(items, (list, tuple)):
                    clean = [" ".join(str(x).split()).strip() for x in items]
                    clean = [c for c in clean if c]
                    if clean:
                        row["raw_skills"] = clean
                        row["skills_delimited"] = True
                        row["raw_skills_text"] = ", ".join(clean)
            except (ValueError, SyntaxError):
                row["raw_skills_text"] = raw
        row["language"], row["language_is_imputed"] = "English", True
        row["access_model"] = "audit_free_cert_paid"
        rows.append(row)
    return rows


ADAPTERS = {
    "coursera2021": adapt_coursera2021,
    "coursera2025": adapt_coursera2025,
    "edx": adapt_edx,
    "coursera_hours": adapt_coursera_hours,
}


def _enforce_unique_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """course_id must be one-per-source-row.

    Two rows in the same source can share a url_slug (the Coursera 2021
    export contains genuine repeats). Left alone, that makes course_id
    non-unique, which silently breaks canonical-row selection in stage 5 and
    would later corrupt arm identity. Collisions get a stable ordinal
    suffix; the dedup stage is what decides they are the same course.
    """
    dupe_mask = frame["course_id"].duplicated(keep=False)
    if not dupe_mask.any():
        frame["course_id_is_disambiguated"] = False
        return frame
    ordinal = frame.groupby("course_id").cumcount()
    suffixed = frame["course_id"].astype("string") + "#" + (ordinal + 1).astype(str)
    frame["course_id"] = frame["course_id"].where(~dupe_mask, suffixed)
    frame["course_id_is_disambiguated"] = dupe_mask
    n = int(dupe_mask.sum())
    print(f"[normalize] disambiguated {n} colliding course_id(s)")
    return frame


def run(source_ids: Optional[List[str]] = None) -> pd.DataFrame:
    paths.ensure_dirs()
    specs = paths.enabled_sources()
    ids = source_ids or [s for s in specs if paths.staging_file(s).exists()]
    if not ids:
        raise SystemExit("[normalize] no staging files; run stage 2 first")

    frames = []
    for sid in ids:
        if sid not in ADAPTERS:
            print(f"[normalize] {sid}: no adapter, skipped")
            continue
        staged = pd.read_parquet(paths.staging_file(sid))
        rows = ADAPTERS[sid](staged, specs[sid])
        frame = pd.DataFrame(rows)
        for col in CANONICAL_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        frames.append(_apply_dtypes(frame[CANONICAL_COLUMNS].copy()))
        declared = int(frame["level_source"].isin(
            ("declared", "declared_vocab_mapped")).sum())
        print(
            f"[normalize] {sid:14} {len(frame):>6} rows  "
            f"level_declared={declared:<6} "
            f"effort_total={int(frame['effort_hours_total'].notna().sum()):<6} "
            f"desc={int(frame['description'].notna().sum())}"
        )

    out = _apply_dtypes(pd.concat(frames, ignore_index=True))
    out = _enforce_unique_ids(out)
    path = paths.STAGING / "normalized.parquet"
    hashing.write_parquet(out, path, stage="normalize")
    hashing.record_stage("normalize", sources=ids, rows=len(out))
    print(f"[normalize] wrote {paths.rel(path)} ({len(out)} rows)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3: normalize to canonical schema")
    ap.add_argument("--only", nargs="*")
    run(ap.parse_args().only)


if __name__ == "__main__":
    main()
