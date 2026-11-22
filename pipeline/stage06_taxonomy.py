"""Stage 6 -- build and freeze the skill taxonomy.

Phases mirror docs/ingestion-plan.md section 3:

  extract  (3b) LLM segmentation of the residue the dictionary could not cover
  merge    (3c/3d) normalise, block, three-band cascade, union-find closure
  prune    (3e) support threshold, breadth test, per-family caps
  freeze   (3f) noisy-tag filter, then write immutable skills + tag_crosswalk

Batch, not incremental: candidates are generated over the whole catalog
first, then a single global merge pass runs. Order-independence is a build
requirement, not an accident -- an incremental design would make the
taxonomy depend on the order courses were processed.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz

from . import hashing, llm, paths
from .textnorm import UnionFind, merge_key

EXTRACT_SYSTEM = (
    "You segment noisy course skill-tag text into individual skill phrases.\n"
    "The input is a concatenation of tags with no delimiters, scraped from a "
    "course catalog. Word order is unreliable and tags vary in length.\n\n"
    "You also receive the course title as context.\n\n"
    "Rules:\n"
    "1. Ground every phrase in the tag text or the title. Never invent a "
    "skill that appears in neither. You MAY use the title to complete a "
    "truncated tag ('model canvas' + title -> 'Business Model Canvas').\n"
    "2. Each phrase is 1-4 words, a noun phrase naming a skill, technology, "
    "method or topic.\n"
    "3. Drop fragments that are not skills (stray words, numbers, artefacts).\n"
    "4. Prefer the longest coherent phrase: 'data structures' not 'data'.\n"
    "5. Do not expand abbreviations and do not rephrase.\n"
    'Return JSON: {"skills": ["phrase", ...]}. Empty list if nothing usable.'
)


def _cfg() -> Dict[str, Any]:
    with open(paths.CONFIG / "thresholds.yaml", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("taxonomy", {})


# ---------------------------------------------------------------- extract
def phase_extract(limit: Optional[int] = None) -> pd.DataFrame:
    cfg = llm.config()
    model = cfg["models"]["extract"]
    budget = llm.Budget(cfg["limits"]["max_extract_calls"], "extract")

    residue = pd.read_parquet(paths.REPORTS / "skill_residue.parquet")
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet").set_index("course_id")
    if limit:
        residue = residue.head(limit)

    rows: List[Dict[str, Any]] = []
    misses = 0
    for i, rec in enumerate(residue.itertuples(), 1):
        title = str(courses.loc[rec.course_id, "title_normalized"] or "")
        user = json.dumps({"course_title": title, "tag_text": rec.residue},
                          ensure_ascii=False)
        out = llm.chat_json(
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": user}],
            model=model, budget=budget,
        )
        if out is None:
            misses += 1
            continue
        for phrase in (out.get("skills") or [])[:12]:
            phrase = " ".join(str(phrase).split()).strip()
            if 2 < len(phrase) <= 60 and len(phrase.split()) <= 4:
                # Record whether the phrase is verbatim in the tag text.
                # Title-completed phrases are usually better skills, but the
                # distinction has to stay auditable rather than implied.
                rows.append({
                    "course_id": rec.course_id, "phrase": phrase,
                    "origin": "llm_residue",
                    "in_residue": phrase.casefold() in str(rec.residue).casefold(),
                })
        if i % 200 == 0:
            print(f"[taxonomy:extract] {i}/{len(residue)} (api calls {budget.used})")

    df = pd.DataFrame(rows).drop_duplicates()
    hashing.write_parquet(df, paths.STAGING / "llm_candidates.parquet",
                          stage="taxonomy_extract")
    print(f"[taxonomy:extract] {len(df):,} candidate phrases from "
          f"{residue['course_id'].nunique():,} courses "
          f"(api calls {budget.used}, misses {misses})")
    return df


# ------------------------------------------------------------------ merge
def _candidate_pool() -> pd.DataFrame:
    """Every candidate phrase with its catalog support."""
    existing = pd.read_parquet(paths.CATALOG / "course_skills.parquet")
    pool = existing[["course_id", "skill_phrase"]].rename(
        columns={"skill_phrase": "phrase"})
    pool["origin"] = "deterministic"

    llm_path = paths.STAGING / "llm_candidates.parquet"
    if llm_path.exists():
        extra = pd.read_parquet(llm_path)[["course_id", "phrase", "origin"]]
        pool = pd.concat([pool, extra], ignore_index=True)

    pool["phrase"] = pool["phrase"].map(lambda p: " ".join(str(p).split()).strip())
    pool = pool[pool["phrase"].str.len() > 2]
    pool["phrase_key"] = pool["phrase"].map(merge_key)
    pool = pool[pool["phrase_key"] != ""]
    return pool.drop_duplicates(subset=["course_id", "phrase_key"])


def phase_merge() -> pd.DataFrame:
    cfg = _cfg()
    lcfg = llm.config()
    budget = llm.Budget(lcfg["limits"]["max_adjudicate_calls"], "adjudicate")

    pool = _candidate_pool()
    # One row per distinct phrase, with catalog support and a display form.
    agg = pool.groupby("phrase_key").agg(
        phrase=("phrase", lambda s: s.value_counts().index[0]),
        support=("course_id", "nunique"),
        origins=("origin", lambda s: ",".join(sorted(set(s)))),
    ).reset_index()
    agg = agg.sort_values(["support", "phrase_key"], ascending=[False, True])
    agg = agg[agg["support"] >= cfg["min_support_for_merge"]].reset_index(drop=True)
    print(f"[taxonomy:merge] {len(agg):,} distinct phrases entering the cascade")

    vecs = llm.embed(agg["phrase"].tolist(), lcfg["models"]["embed"])
    with warnings.catch_warnings():
        # numpy 2.x on macOS Accelerate emits spurious divide/overflow
        # warnings from BLAS matmul. Verified numerically: the result
        # matches a manual dot product and contains no NaN or Inf.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1.0)

    k = min(cfg["neighbours"], len(agg) - 1)
    nn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

    uf = UnionFind(agg["phrase_key"].tolist())
    decisions: List[Dict[str, Any]] = []
    bands = Counter()
    seen = set()

    for i in range(len(agg)):
        for j in nn_idx[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            cos = float(sims[a, b])
            if cos < cfg["min_cosine_candidate"]:
                continue
            pa, pb = agg.at[a, "phrase"], agg.at[b, "phrase"]
            tsr = float(fuzz.token_set_ratio(pa.casefold(), pb.casefold()))

            if cos >= cfg["auto_merge_cosine"] and tsr >= cfg["auto_merge_tsr"]:
                band = "auto_merge"
            elif cos >= cfg["adjudicate_cosine"]:
                band = "adjudicate"
            else:
                band = "auto_reject"
            bands[band] += 1
            decisions.append({"phrase_a": pa, "phrase_b": pb, "key_a": agg.at[a, "phrase_key"],
                              "key_b": agg.at[b, "phrase_key"], "cosine": round(cos, 4),
                              "tsr": tsr, "band": band, "merged": band == "auto_merge"})
            if band == "auto_merge":
                uf.union(agg.at[a, "phrase_key"], agg.at[b, "phrase_key"])

    # Adjudicate the middle band with the LLM.
    adj = [d for d in decisions if d["band"] == "adjudicate"]
    print(f"[taxonomy:merge] bands {dict(bands)} -> adjudicating {len(adj)}")
    for n, d in enumerate(adj, 1):
        out = llm.chat_json([
            {"role": "system", "content":
             "You decide whether two course-skill phrases name the SAME skill "
             "and should be merged into one taxonomy entry.\n"
             "Merge only if a learner would consider them interchangeable, or "
             "one is plainly a spelling/word-order variant of the other.\n"
             "Do NOT merge a general skill with a specific one (e.g. "
             "'machine learning' vs 'deep learning'), related-but-distinct "
             "tools, or different stages of a workflow.\n"
             'Return JSON: {"same": true|false, "canonical": "preferred phrase", '
             '"reason": "max 15 words"}'},
            {"role": "user", "content": json.dumps(
                {"phrase_a": d["phrase_a"], "phrase_b": d["phrase_b"]})},
        ], model=lcfg["models"]["adjudicate"], budget=budget)
        if out and bool(out.get("same")):
            uf.union(d["key_a"], d["key_b"])
            d["merged"] = True
            d["reason"] = str(out.get("reason", ""))[:120]
        elif out:
            d["reason"] = str(out.get("reason", ""))[:120]
        if n % 200 == 0:
            print(f"[taxonomy:merge]   adjudicated {n}/{len(adj)}")

    agg["cluster"] = [uf.find(k) for k in agg["phrase_key"]]
    dec_df = pd.DataFrame(decisions)
    hashing.write_parquet(agg, paths.STAGING / "taxonomy_clusters.parquet",
                          stage="taxonomy_merge")
    if len(dec_df):
        hashing.write_parquet(dec_df, paths.REPORTS / "taxonomy_merge_decisions.parquet",
                              stage="taxonomy_merge")
    merged = len(agg) - agg["cluster"].nunique()
    print(f"[taxonomy:merge] {len(agg):,} phrases -> {agg['cluster'].nunique():,} "
          f"clusters (collapsed {merged:,}); api calls {budget.used}")
    return agg


# ------------------------------------------------------------------ prune
def _family_of_courses(course_ids, course_family: Dict[str, str]) -> Counter:
    return Counter(course_family.get(c) for c in course_ids
                   if course_family.get(c))


def phase_prune() -> pd.DataFrame:
    """Support threshold, breadth test, per-family caps (plan 3e)."""
    cfg = _cfg()
    clusters = pd.read_parquet(paths.STAGING / "taxonomy_clusters.parquet")
    pool = _candidate_pool()
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    course_family = dict(zip(courses["course_id"], courses["domain_family"]))
    family_labels = {merge_key(f["label"]): f["id"]
                     for f in paths.load_families()["families"]}
    family_ids = {merge_key(f["id"].replace("-", " ")): f["id"]
                  for f in paths.load_families()["families"]}

    key_to_cluster = dict(zip(clusters["phrase_key"], clusters["cluster"]))
    pool = pool[pool["phrase_key"].isin(key_to_cluster)].copy()
    pool["cluster"] = pool["phrase_key"].map(key_to_cluster)

    rows = []
    for cluster, grp in pool.groupby("cluster"):
        members = clusters[clusters["cluster"] == cluster]
        # Canonical label: highest-support member, shortest name as tiebreak.
        canon = members.sort_values(
            ["support", "phrase"], ascending=[False, True]
        ).iloc[0]["phrase"]
        course_ids = grp["course_id"].unique()
        fam_counts = _family_of_courses(course_ids, course_family)
        total = sum(fam_counts.values())
        if total:
            top_fam, top_n = fam_counts.most_common(1)[0]
            concentration = top_n / total
            course_fam = top_fam
            # Shannon entropy over families, normalised to [0,1]. A leaf
            # spread evenly across many families is a generic term, not a
            # skill -- this is the breadth test.
            probs = np.array([v / total for v in fam_counts.values()])
            ent = float(-(probs * np.log(probs + 1e-12)).sum())
            max_ent = float(np.log(len(paths.load_families()["families"])))
            breadth = ent / max_ent if max_ent else 0.0
        else:
            top_fam, concentration, breadth, course_fam = None, 0.0, 1.0, None

        key = merge_key(canon)
        is_family_label = key in family_labels or key in family_ids
        rows.append({
            "cluster": cluster, "label": canon, "skill_key": key,
            "support": len(course_ids), "course_majority_family": course_fam,
            "family_concentration": round(concentration, 3),
            "breadth": round(breadth, 3),
            "n_variants": len(members),
            "is_family_label": is_family_label,
        })

    cand = pd.DataFrame(rows)

    # Family assignment by direct semantic fit to the family profile, NOT by
    # majority vote of the courses carrying the skill. Inheriting the course
    # majority compounds stage 8's argmax-keyword errors -- it put "back end"
    # and "front end" under deep-learning, and "FinTech" under security.
    fams = paths.load_families()["families"]
    profiles = [" ".join([f["label"], f.get("scope", ""),
                          " ".join(f.get("in_examples", []) or []),
                          " ".join(f.get("keywords_include", []) or [])])
                for f in fams]
    lcfg = llm.config()
    fvec = llm.embed(profiles, lcfg["models"]["embed"])
    svec = llm.embed(cand["label"].tolist(), lcfg["models"]["embed"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = svec @ fvec.T
    best = fit.argmax(axis=1)
    cand["family"] = [fams[i]["id"] for i in best]
    cand["family_fit"] = [round(float(fit[r, best[r]]), 4) for r in range(len(cand))]
    cand["family_agrees_with_courses"] = (
        cand["family"] == cand["course_majority_family"])

    cand["drop_reason"] = None
    cand.loc[cand["support"] < cfg["min_support"], "drop_reason"] = "low_support"
    cand.loc[cand["drop_reason"].isna() & cand["is_family_label"],
             "drop_reason"] = "is_a_family_label"
    cand.loc[cand["drop_reason"].isna() & (cand["breadth"] > cfg["max_breadth"]),
             "drop_reason"] = "too_broad"
    cand.loc[cand["drop_reason"].isna() & cand["family"].isna(),
             "drop_reason"] = "no_family"
    # A label needs EITHER a confident semantic fit to its family, OR a
    # weaker fit corroborated by the courses that carry it. A flat fit floor
    # alone has false negatives: short proper nouns like "flask" are real
    # skills but score low against family prose, while genuine junk like
    # "persona" and "market segmentation" scores low *and* has no course
    # support in that family.
    confident = cand["family_fit"] >= cfg["min_family_fit_alone"]
    corroborated = ((cand["family_fit"] >= cfg["min_family_fit_corroborated"])
                    & cand["family_agrees_with_courses"])
    cand.loc[cand["drop_reason"].isna() & ~(confident | corroborated),
             "drop_reason"] = "weak_family_fit"

    keep = cand[cand["drop_reason"].isna()].copy()
    keep = keep.sort_values(["family", "support", "label"],
                            ascending=[True, False, True])
    keep["rank_in_family"] = keep.groupby("family").cumcount()
    over = keep["rank_in_family"] >= cfg["max_leaves_per_family"]
    cand.loc[keep[over].index, "drop_reason"] = "family_cap"
    keep = keep[~over].drop(columns=["rank_in_family"])

    hashing.write_parquet(cand, paths.STAGING / "taxonomy_candidates.parquet",
                          stage="taxonomy_prune")
    print(f"[taxonomy:prune] {len(cand):,} clusters -> {len(keep):,} leaves")
    print("[taxonomy:prune] drops:",
          dict(cand["drop_reason"].value_counts(dropna=True)))
    return keep


# ----------------------------------------------------------------- freeze
def phase_freeze() -> pd.DataFrame:
    """Noisy-tag filter, then write the immutable taxonomy (plan 3f)."""
    cfg = _cfg()
    lcfg = llm.config()
    keep = phase_prune()
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    pool = _candidate_pool()
    clusters = pd.read_parquet(paths.STAGING / "taxonomy_clusters.parquet")

    key_to_cluster = dict(zip(clusters["phrase_key"], clusters["cluster"]))
    cluster_to_leaf = dict(zip(keep["cluster"], keep["skill_key"]))
    label_of = dict(zip(keep["skill_key"], keep["label"]))

    pool = pool[pool["phrase_key"].isin(key_to_cluster)].copy()
    pool["cluster"] = pool["phrase_key"].map(key_to_cluster)
    pool["skill_key"] = pool["cluster"].map(cluster_to_leaf)
    assigned = pool[pool["skill_key"].notna()].copy()

    # --- noisy-tag filter -------------------------------------------------
    # The upstream tags are genuinely noisy: a Git course carries a real
    # "Mergers & Acquisitions" tag. Keep an assignment only when the skill
    # phrase is semantically close to the course's own text.
    ctext = (courses["title_normalized"].fillna("") + ". " +
             courses["description"].fillna("").str.slice(0, 600))
    cvec = llm.embed(ctext.tolist(), lcfg["models"]["embed"])
    cidx = {cid: i for i, cid in enumerate(courses["course_id"])}

    leaf_keys = list(label_of)
    lvec = llm.embed([label_of[k] for k in leaf_keys], lcfg["models"]["embed"])
    lidx = {k: i for i, k in enumerate(leaf_keys)}

    sims = []
    for r in assigned.itertuples():
        ci, li = cidx.get(r.course_id), lidx.get(r.skill_key)
        sims.append(float(cvec[ci] @ lvec[li]) if ci is not None and li is not None else 0.0)
    assigned["similarity"] = sims
    before = len(assigned)
    kept = assigned[assigned["similarity"] >= cfg["min_tag_course_similarity"]].copy()
    print(f"[taxonomy:freeze] noisy-tag filter: {before:,} -> {len(kept):,} "
          f"assignments (dropped {before - len(kept):,})")

    # --- frozen artefacts -------------------------------------------------
    # --- difficulty and time per skill -----------------------------------
    # Difficulty is derivable: every skill maps to courses and 99.1% of
    # courses carry a declared level.
    # Time is NOT. Skills come only from Coursera (edX has no skills column
    # at source) while effort hours come only from edX, so the two are
    # perfectly disjoint. Calendar weeks is the only time signal that
    # overlaps skill-bearing courses, at ~30% coverage, and it measures
    # schedule span rather than learner effort -- recorded as a clearly
    # labelled proxy, not as "time required".
    lvl = courses.set_index("course_id")["level_canonical"]
    wks = courses.set_index("course_id")["calendar_weeks_lo"]
    enrich = kept.assign(
        level=kept["course_id"].map(lvl),
        weeks=kept["course_id"].map(wks),
    ).groupby("skill_key").agg(
        difficulty_mean=("level", "mean"),
        difficulty_median=("level", "median"),
        difficulty_coverage=("level", lambda s: float(s.notna().mean())),
        typical_weeks_median=("weeks", "median"),
        weeks_coverage=("weeks", lambda s: float(s.notna().mean())),
    )

    fam_of = dict(zip(keep["skill_key"], keep["family"]))
    skills = pd.DataFrame({
        "skill_id": ["SKILL_" + k.upper().replace("-", "_") for k in keep["skill_key"]],
        "skill_slug": keep["skill_key"].values,
        "label": keep["label"].values,
        "family": keep["family"].values,
        "tier": "skill",
        "support_count": keep["support"].values,
        "family_concentration": keep["family_concentration"].values,
        "family_fit": keep["family_fit"].values,
        "breadth": keep["breadth"].values,
        "difficulty_mean": keep["skill_key"].map(enrich["difficulty_mean"]).round(3).values,
        "difficulty_median": keep["skill_key"].map(enrich["difficulty_median"]).values,
        "difficulty_coverage": keep["skill_key"].map(enrich["difficulty_coverage"]).round(3).values,
        "typical_weeks_median": keep["skill_key"].map(enrich["typical_weeks_median"]).round(2).values,
        "weeks_coverage": keep["skill_key"].map(enrich["weeks_coverage"]).round(3).values,
    }).sort_values(["family", "label"]).reset_index(drop=True)

    crosswalk = clusters.merge(
        keep[["cluster", "skill_key", "label"]], on="cluster", how="left"
    ).rename(columns={"phrase": "raw_tag", "label": "canonical_label"})
    crosswalk["action"] = np.where(crosswalk["skill_key"].notna(), "map", "drop")
    crosswalk["skill_id"] = crosswalk["skill_key"].map(
        lambda k: "SKILL_" + k.upper().replace("-", "_") if isinstance(k, str) else None)
    crosswalk = crosswalk[["raw_tag", "phrase_key", "skill_id", "canonical_label",
                           "action", "support", "origins"]]

    course_skills = kept.assign(
        skill_id=kept["skill_key"].map(
            lambda k: "SKILL_" + k.upper().replace("-", "_")),
        relation="teaches",
        tag_confidence=kept["similarity"].round(4),
    )[["course_id", "skill_id", "skill_key", "relation", "origin",
       "tag_confidence"]].drop_duplicates(subset=["course_id", "skill_id"])

    hashing.write_parquet(skills, paths.CATALOG / "skills.parquet", stage="taxonomy_freeze")
    hashing.write_parquet(crosswalk, paths.CATALOG / "tag_crosswalk.parquet",
                          stage="taxonomy_freeze")
    hashing.write_parquet(course_skills, paths.CATALOG / "course_skills_final.parquet",
                          stage="taxonomy_freeze")

    lines = ["# Skill taxonomy (frozen)", "",
             f"- leaves: **{len(skills):,}** across "
             f"**{skills['family'].nunique()}** families",
             f"- assignments: **{len(course_skills):,}** over "
             f"**{course_skills['course_id'].nunique():,}** courses",
             f"- crosswalk rows: **{len(crosswalk):,}** "
             f"({int((crosswalk['action']=='map').sum()):,} map, "
             f"{int((crosswalk['action']=='drop').sum()):,} drop)", "",
             "## Leaves per family", "", "| family | leaves |", "|---|---|"]
    for fam, n in skills["family"].value_counts().items():
        lines.append(f"| `{fam}` | {n} |")
    lines += ["", "## Noisy-tag filter", "",
              f"- assignments before: **{before:,}**",
              f"- after: **{len(kept):,}** "
              f"(dropped {before - len(kept):,}, "
              f"{(before - len(kept)) / before:.1%})", "",
              "An assignment survives only if the leaf label is semantically",
              "close to the course's own title and description. This is what",
              "removes tags like `Mergers & Acquisitions` on a Git course,",
              "which are genuinely present in the upstream data."]
    lines += ["", "## Audit status", "",
              "Thresholds were calibrated against this catalog over three",
              "passes and then deliberately frozen: further tuning against",
              "individually inspected examples risks overfitting the filter",
              "to the cases that happened to be looked at.",
              "",
              "Known residual errors found by inspection:",
              "",
              "- `FinTech` survives at family_fit 0.303, just over the 0.30",
              "  floor, and is assigned to `web-frontend`. It is a domain,",
              "  not a skill.",
              "- `flask` was dropped as a false negative: a real framework",
              "  whose short proper-noun name scores low against family prose",
              "  and whose courses did not corroborate the family.",
              "",
              "`reports/taxonomy_audit.csv` lists every leaf with an empty",
              "`human_label` column. Precision is **not reported**: it needs",
              "that column filled in. Until then this is an unvalidated",
              "vocabulary, not a measured result.",
              ]
    audit = skills[["skill_id", "label", "family", "support_count",
                    "family_fit", "breadth"]].copy()
    audit["human_label"] = ""
    audit.sort_values(["family", "label"]).to_csv(
        paths.REPORTS / "taxonomy_audit.csv", index=False)

    (paths.REPORTS / "taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("taxonomy_freeze", leaves=len(skills),
                         assignments=len(course_skills))
    print(f"[taxonomy:freeze] {len(skills)} leaves, {len(course_skills):,} assignments")
    return skills


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6: build the skill taxonomy")
    ap.add_argument("--phase",
                    choices=["extract", "merge", "prune", "freeze", "all"],
                    default="all")
    ap.add_argument("--limit", type=int, help="cap courses processed (piloting)")
    args = ap.parse_args()
    if args.phase in ("extract", "all"):
        phase_extract(args.limit)
    if args.phase in ("merge", "all"):
        phase_merge()
    if args.phase == "prune":
        phase_prune()
    if args.phase in ("freeze", "all"):
        phase_freeze()


if __name__ == "__main__":
    main()
