"""Stage 7 -- skill extraction (deterministic pass).

Produces `course_skills` (relation='teaches') plus the candidate vocabulary
that stage 6's merge cascade consumes.

The two Coursera exports disagree about format in a way that turns out to be
useful rather than merely annoying:

  * 2025 lists skills **comma-delimited and clean** -- 1,754 distinct
    phrases, mean 2.18 words. That is a ready-made canonical vocabulary.
  * 2021 concatenates tags into an undelimited blob with no reliable casing
    rule ("Drama Comedy peering screenwriting film Document Review ...").
    Splitting on whitespace would fabricate tokens, so it is never done.

So: rows matched to a 2025 donor take its delimited skills directly. The
rest are segmented by longest-match against the 2025 vocabulary. Whatever
the dictionary cannot cover is preserved verbatim as `residue` for the
LLM-backed pass, rather than silently dropped.

NOT DONE HERE: LLM candidate generation and the adjudication band of the
merge cascade (plan section 3b/3d). Both need an API key. What this stage
emits is a deterministic floor, not the frozen taxonomy.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from . import hashing, paths
from .textnorm import merge_key

MAX_PHRASE_WORDS = 5
MIN_PHRASE_CHARS = 3

# Tokens that are pure noise as standalone skills.
_STOP_PHRASES = {"other", "general", "introduction", "overview", "basics",
                 "fundamentals", "principles", "concepts", "tools"}


def build_vocabulary(normalized: pd.DataFrame) -> pd.DataFrame:
    """Canonical skill phrases harvested from the delimited 2025 export."""
    counter: Counter = Counter()
    donor = normalized[normalized["source_id"] == "coursera2025"]
    for blob in donor["raw_skills_text"].dropna():
        for part in str(blob).split(","):
            phrase = " ".join(part.split()).strip()
            if len(phrase) >= MIN_PHRASE_CHARS and len(phrase.split()) <= MAX_PHRASE_WORDS:
                counter[phrase] += 1

    rows = []
    for phrase, support in counter.items():
        key = merge_key(phrase)
        if not key or phrase.casefold() in _STOP_PHRASES:
            continue
        rows.append({
            "phrase": phrase,
            "phrase_key": key,
            "phrase_cf": phrase.casefold(),
            "support": support,
            "n_words": len(phrase.split()),
            "origin": "coursera2025_delimited",
        })
    vocab = pd.DataFrame(rows).sort_values(["support", "phrase"],
                                           ascending=[False, True])
    # Collapse phrases that share a merge key (word-order variants).
    vocab = vocab.drop_duplicates(subset=["phrase_key"], keep="first").reset_index(drop=True)
    return vocab


def _segment(body: str, lookup: Dict[str, str]) -> Tuple[List[str], str]:
    """Longest-match segmentation. Returns (matched phrases, residue).

    Greedy longest-first because short phrases are substrings of longer ones
    ("data" inside "data analysis"); matching short-first would shred the
    real tags.
    """
    words = body.split()
    matched: List[str] = []
    residue: List[str] = []
    i = 0
    while i < len(words):
        hit = None
        for length in range(min(MAX_PHRASE_WORDS, len(words) - i), 0, -1):
            cand = " ".join(words[i:i + length]).casefold().strip(" ()[]")
            if len(cand) >= MIN_PHRASE_CHARS and cand in lookup:
                hit = (lookup[cand], length)
                break
        if hit:
            matched.append(hit[0])
            i += hit[1]
        else:
            residue.append(words[i])
            i += 1
    return matched, " ".join(residue)


def run() -> pd.DataFrame:
    normalized = pd.read_parquet(paths.STAGING / "normalized.parquet")
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    live = courses[courses["in_scope"] == True].copy()  # noqa: E712

    vocab = build_vocabulary(normalized)
    lookup = {r.phrase_cf: r.phrase for r in vocab.itertuples()}
    # Word-order variants resolve to the same canonical phrase.
    for r in vocab.itertuples():
        lookup.setdefault(r.phrase_key.replace("-", " "), r.phrase)

    records: List[Dict] = []
    residues: List[Dict] = []
    stats = Counter()

    for row in live.itertuples():
        cid = row.course_id
        delimited = row.raw_skills
        if delimited is not None and len(delimited) > 0:
            stats["rows_delimited"] += 1
            for phrase in delimited:
                phrase = " ".join(str(phrase).split()).strip()
                if len(phrase) < MIN_PHRASE_CHARS or phrase.casefold() in _STOP_PHRASES:
                    continue
                records.append({
                    "course_id": cid, "skill_phrase": phrase,
                    "skill_key": merge_key(phrase), "relation": "teaches",
                    "tag_source": "source_field", "tag_confidence": 1.0,
                })
            continue

        body = row.skills_body
        if not body:
            stats["rows_no_skill_text"] += 1
            continue
        matched, residue = _segment(str(body), lookup)
        if matched:
            stats["rows_dictionary_matched"] += 1
        else:
            stats["rows_unmatched"] += 1
        for phrase in dict.fromkeys(matched):
            records.append({
                "course_id": cid, "skill_phrase": phrase,
                "skill_key": merge_key(phrase), "relation": "teaches",
                "tag_source": "dictionary_matched", "tag_confidence": 0.6,
            })
        if residue.strip():
            residues.append({"course_id": cid, "residue": residue.strip(),
                             "residue_tokens": len(residue.split())})

    course_skills = pd.DataFrame(records).drop_duplicates(
        subset=["course_id", "skill_key"]).reset_index(drop=True)
    residue_df = pd.DataFrame(residues)

    # Vocabulary support recomputed against actual catalog usage -- this is
    # the number the stage 6 prune threshold should key on, not the donor's
    # internal frequency.
    used = course_skills.groupby("skill_key").agg(
        courses=("course_id", "nunique"),
        phrase=("skill_phrase", "first"),
    ).reset_index().sort_values("courses", ascending=False)

    hashing.write_parquet(course_skills, paths.CATALOG / "course_skills.parquet",
                          stage="skills")
    hashing.write_parquet(used, paths.CATALOG / "skill_candidates.parquet",
                          stage="skills")
    if len(residue_df):
        hashing.write_parquet(residue_df, paths.REPORTS / "skill_residue.parquet",
                              stage="skills")

    covered = course_skills["course_id"].nunique()
    lines = [
        "# Skill extraction report", "",
        f"- in-scope courses: **{len(live):,}**",
        f"- courses with >=1 extracted skill: **{covered:,}** "
        f"({covered / len(live):.1%})",
        f"- skill assignments: **{len(course_skills):,}**",
        f"- distinct skill phrases in use: **{used['skill_key'].nunique():,}**",
        f"- donor vocabulary size: **{len(vocab):,}** phrases", "",
        "## How each course was handled", "", "| path | courses |", "|---|---|",
    ]
    for key, label in [("rows_delimited", "clean delimited skills (2025 donor)"),
                       ("rows_dictionary_matched", "segmented via donor vocabulary"),
                       ("rows_unmatched", "skill text present, no dictionary hit"),
                       ("rows_no_skill_text", "no skill text at all")]:
        lines.append(f"| {label} | {stats[key]:,} |")

    if len(residue_df):
        lines += [
            "", "## Residue (unmatched by the dictionary)", "",
            f"- courses with residue: **{len(residue_df):,}**",
            f"- mean unmatched tokens per course: "
            f"**{residue_df['residue_tokens'].mean():.1f}**", "",
            "Residue is preserved verbatim in `reports/skill_residue.parquet`.",
            "It is the undelimited 2021 tag text the 2025 vocabulary does not",
            "cover, and it is the input for the LLM extraction pass.",
        ]

    lines += [
        "", "## Not done in this stage", "",
        "- **LLM candidate generation** (plan 3b): the residue above needs it.",
        "- **Adjudication band** of the merge cascade (plan 3d): the",
        "  deterministic auto-merge and auto-reject bands can run without a",
        "  model, but the middle band cannot.",
        "- **Freezing the taxonomy** (plan 3f): premature until both land.",
        "",
        "Both require an API key, which is not configured. What this stage",
        "emits is a deterministic floor, not the frozen taxonomy.",
    ]
    (paths.REPORTS / "skill_extraction.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")

    hashing.record_stage("skills", assignments=len(course_skills),
                         courses_covered=int(covered),
                         distinct_phrases=int(used["skill_key"].nunique()))
    print(f"[skills] vocabulary={len(vocab):,} phrases")
    print(f"[skills] assignments={len(course_skills):,} over {covered:,}/{len(live):,} "
          f"courses ({covered/len(live):.1%})")
    print(f"[skills] paths: {dict(stats)}")
    return course_skills


def main() -> None:
    argparse.ArgumentParser(description="Stage 7: deterministic skill extraction").parse_args()
    run()


if __name__ == "__main__":
    main()
