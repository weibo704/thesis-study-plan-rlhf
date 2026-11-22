"""Stage 8 -- scope classification and the attrition funnel.

Decides which canonical courses belong to the AI/software-development item
space. Two independent signals, deliberately kept separable so a reviewer
can see which one carried a decision:

  1. weighted family-keyword evidence over title / skills / description
  2. TF-IDF cosine against a per-family profile document

Nothing is deleted. Out-of-scope rows stay in `courses.parquet` with
`arm_status='out_of_scope'` and a `scope_reason`, so the funnel is
reconstructable and the decision is revisitable.

Runs before the taxonomy stage, so it sees raw skill text rather than
mapped skills. Re-run it after stage 7 to use the cleaned tags.
"""
from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from . import hashing, paths


def _cfg() -> Dict[str, Any]:
    with open(paths.CONFIG / "thresholds.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["scope"]


def _compile(terms: List[str]) -> List[Tuple[str, re.Pattern]]:
    out = []
    for term in terms:
        esc = re.escape(term.casefold()).replace(r"\ ", r"\s+")
        # Word-boundary anchored so "ai" does not match "said" or "detail".
        out.append((term, re.compile(rf"(?<!\w){esc}(?!\w)")))
    return out


def _family_profiles() -> Tuple[List[str], Dict[str, List[Tuple[str, re.Pattern]]], Dict[str, str]]:
    fams = paths.load_families()["families"]
    ids = [f["id"] for f in fams]
    patterns = {f["id"]: _compile(f.get("keywords_include", [])) for f in fams}
    # Profile doc: scope prose + positive examples + keywords. Deliberately
    # excludes out_examples, which would pull the vector toward what the
    # family is NOT.
    docs = {
        f["id"]: " ".join([
            f.get("label", ""), f.get("scope", ""),
            " ".join(f.get("in_examples", []) or []),
            " ".join(f.get("keywords_include", []) or []),
        ]).casefold()
        for f in fams
    }
    return ids, patterns, docs


def _out_hint_patterns() -> List[Tuple[str, re.Pattern]]:
    hints = paths.load_families().get("out_of_scope_hints", []) or []
    flat = [term for group in hints for term in group]
    return _compile(flat)


def _score_keywords(row: pd.Series, patterns, cfg) -> Tuple[Dict[str, float], List[str]]:
    title = (row.get("title_normalized") or "").casefold()
    skills = (row.get("raw_skills_text") or "").casefold()
    desc = (row.get("description") or "").casefold()
    scores: Dict[str, float] = {}
    hits: List[str] = []
    for fam, pats in patterns.items():
        score = 0.0
        for term, pat in pats:
            if pat.search(title):
                score += cfg["weight_title"]
                hits.append(f"title:{term}")
            if pat.search(skills):
                score += cfg["weight_skills"]
                hits.append(f"skills:{term}")
            elif pat.search(desc):
                score += cfg["weight_description"]
                hits.append(f"desc:{term}")
        if score:
            scores[fam] = score
    return scores, hits


def run() -> pd.DataFrame:
    cfg = _cfg()
    fam_ids, patterns, docs = _family_profiles()
    out_pats = _out_hint_patterns()

    deduped = pd.read_parquet(paths.STAGING / "deduped.parquet")
    total_rows = len(deduped)
    df = deduped[deduped["is_canonical"]].copy().reset_index(drop=True)

    funnel: List[Tuple[str, int, str]] = [
        ("spine rows", total_rows, "all normalized source rows"),
        ("canonical after dedup", len(df), "one row per dedup cluster"),
    ]

    df["in_scope"] = True
    df["scope_reason"] = ""
    df["domain_family"] = pd.NA
    df["scope_band"] = "in"

    def _drop(mask: pd.Series, reason: str, label: str) -> None:
        active = df["in_scope"] & mask
        df.loc[active, "in_scope"] = False
        df.loc[active, "scope_reason"] = reason
        df.loc[active, "scope_band"] = "out"
        funnel.append((label, int(df["in_scope"].sum()), f"-{int(active.sum())}"))

    # --- structural filters ------------------------------------------------
    _drop(df["content_granularity"].fillna("") == "program",
          "container_not_a_course", "after dropping containers")
    _drop(df["language"].fillna("English") != "English",
          "not_english", "after dropping non-English")
    has_text = df["description"].notna() | df["raw_skills_text"].notna()
    _drop(~has_text, "no_text", "after dropping text-less rows")

    # --- subject classification -------------------------------------------
    live = df[df["in_scope"]].copy()
    blob = (
        live["title_normalized"].fillna("") + " \n " +
        live["raw_skills_text"].fillna("") + " \n " +
        live["description"].fillna("")
    ).str.casefold()

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=2, max_features=60000, sublinear_tf=True)
    course_vecs = vec.fit_transform(blob.tolist())
    fam_vecs = vec.transform([docs[f] for f in fam_ids])
    sims = (course_vecs @ fam_vecs.T).toarray()  # tf-idf rows are L2-normalised

    kw_scores, kw_hits, out_hits = [], [], []
    for _, row in live.iterrows():
        scores, hits = _score_keywords(row, patterns, cfg)
        kw_scores.append(scores)
        kw_hits.append(hits[:6])
        text = " ".join(str(row.get(c) or "") for c in
                        ("title_normalized", "raw_skills_text", "description")).casefold()
        out_hits.append(sum(1 for _, p in out_pats if p.search(text)))

    decisions, families, bands, reasons, best_scores = [], [], [], [], []
    for i, scores in enumerate(kw_scores):
        sim_row = sims[i]
        sim_best_idx = int(np.argmax(sim_row))
        sim_best = float(sim_row[sim_best_idx])

        kw_fam, kw_best = (max(scores.items(), key=lambda kv: kv[1])
                           if scores else (None, 0.0))

        # Keyword evidence alone is sufficient; similarity may promote a weak
        # hit or rescue a keyword-less match, but never vetoes.
        strong_kw = kw_best >= cfg["min_keyword_score"]
        promoted = (kw_best >= cfg["promote_keyword_score"]
                    and sim_best >= cfg["min_similarity_supporting"])
        strong_sim = sim_best >= cfg["min_similarity_alone"]
        weak = (kw_best >= cfg["ambiguous_keyword_score"]
                or sim_best >= cfg["ambiguous_similarity"])

        # Out-of-subject hints demote only when family evidence is weak.
        vetoed = (out_hits[i] >= cfg["out_hint_veto_ratio"] * max(kw_best, 0.5)
                  and not strong_kw)

        if vetoed:
            decisions.append(False); bands.append("out")
            reasons.append(f"out_of_subject_hints({out_hits[i]})")
        elif strong_kw or promoted or strong_sim:
            decisions.append(True); bands.append("in")
            reasons.append(f"kw={kw_best:.1f},sim={sim_best:.2f}")
        elif weak:
            decisions.append(True); bands.append("ambiguous")
            reasons.append(f"ambiguous kw={kw_best:.1f},sim={sim_best:.2f}")
        else:
            decisions.append(False); bands.append("out")
            reasons.append(f"no_subject_signal kw={kw_best:.1f},sim={sim_best:.2f}")

        # A family is only meaningful for a course we believe is in scope.
        # Assigning argmax-similarity to an out-of-scope row produced noise
        # like "Singing Popular Music" -> mlops-deployment.
        if decisions[-1]:
            families.append(kw_fam or fam_ids[sim_best_idx])
        else:
            families.append(None)
        best_scores.append(round(max(kw_best, 0.0), 2))

    live["in_scope"] = decisions
    live["scope_band"] = bands
    live["scope_reason"] = reasons
    live["domain_family"] = families
    live["scope_keyword_score"] = best_scores
    live["scope_similarity"] = [round(float(np.max(sims[i])), 4) for i in range(len(live))]
    live["scope_keyword_hits"] = ["; ".join(h) for h in kw_hits]

    for col in ("in_scope", "scope_band", "scope_reason", "domain_family",
                "scope_keyword_score", "scope_similarity", "scope_keyword_hits"):
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[live.index, col] = live[col]

    funnel.append(("after subject classification", int(df["in_scope"].sum()),
                   f"-{int((~live['in_scope']).sum())}"))

    df["arm_status"] = np.where(df["in_scope"], "active", "out_of_scope")

    # --- outputs -----------------------------------------------------------
    out_path = paths.CATALOG / "courses.parquet"
    hashing.write_parquet(df, out_path, stage="scope")

    ambiguous = df[df["scope_band"] == "ambiguous"]
    audit = pd.concat([
        ambiguous.head(150),
        df[df["scope_band"] == "in"].sample(min(75, int((df["scope_band"] == "in").sum())),
                                            random_state=0),
        df[df["scope_band"] == "out"].sample(min(75, int((df["scope_band"] == "out").sum())),
                                             random_state=0),
    ])[["course_id", "title_normalized", "institution", "domain_family",
        "scope_band", "scope_keyword_score", "scope_similarity",
        "scope_keyword_hits", "scope_reason"]]
    audit = audit.assign(human_label="")
    audit.to_csv(paths.REPORTS / "scope_audit_sample.csv", index=False)

    in_scope = df[df["in_scope"]]
    lines = ["# Scope attrition funnel", "",
             "| step | remaining | delta |", "|---|---|---|"]
    for label, remaining, delta in funnel:
        lines.append(f"| {label} | {remaining:,} | {delta} |")
    lines += ["", f"**In-scope arm pool: {len(in_scope):,}**", "",
              "## Band breakdown", "", "| band | rows |", "|---|---|"]
    for band, n in df["scope_band"].value_counts().items():
        lines.append(f"| {band} | {n:,} |")
    lines += ["", "## Assigned family (in-scope only)", "",
              "| family | rows |", "|---|---|"]
    for fam, n in in_scope["domain_family"].value_counts().items():
        lines.append(f"| `{fam}` | {n:,} |")
    lines += [
        "", "## Audit status", "",
        "`reports/scope_audit_sample.csv` holds the full ambiguous band plus a",
        "random 75 from each of the in and out bands, with an empty",
        "`human_label` column. Precision and recall are **not reported yet**:",
        "they require that column to be filled in. Until then these counts are",
        "an unvalidated classifier's output, not a measured result.",
        "", "Nothing is deleted. Out-of-scope rows remain in",
        "`data/catalog/courses.parquet` with `arm_status='out_of_scope'`.",
    ]
    (paths.REPORTS / "attrition_funnel.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")

    hashing.record_stage("scope", in_scope=int(len(in_scope)),
                         bands=dict(df["scope_band"].value_counts()))
    print(f"[scope] canonical={len(df):,} -> in_scope={len(in_scope):,}")
    print(f"[scope] bands: {dict(df['scope_band'].value_counts())}")
    print(f"[scope] wrote {paths.rel(out_path)} and reports/attrition_funnel.md")
    return df


def main() -> None:
    argparse.ArgumentParser(description="Stage 8: scope classification").parse_args()
    run()


if __name__ == "__main__":
    main()
