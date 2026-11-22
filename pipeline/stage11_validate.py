"""Stage 11 -- validation, manifest, datasheet, and a downstream smoke test.

Every assertion here is a build gate: the stage exits non-zero on failure so
a broken catalog cannot silently ship.

The smoke test at the end is NOT the thesis recommender. It is a trivial
nearest-neighbour scorer over the frozen arm embeddings, run against a few
hand-written profiles. Its only job is to catch a corrupted text pipeline —
if distinct profiles return near-identical top-10s, the embeddings or the
row alignment are broken, and that is far cheaper to discover here than
after building a bandit on top.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from . import hashing, llm, paths

PROFILES: Dict[str, str] = {
    "absolute_beginner_python":
        "I have never programmed before. I want to learn Python basics, "
        "variables, loops and functions from scratch.",
    "ml_engineer_deep_learning":
        "I know Python and statistics well. I want to train neural networks, "
        "CNNs and transformers using PyTorch and TensorFlow.",
    "frontend_web_developer":
        "I want to build websites with HTML, CSS, JavaScript and React, and "
        "make responsive user interfaces.",
    "data_analyst_sql":
        "I need to query relational databases with SQL, model data and build "
        "dashboards and visualizations for business reporting.",
    "security_engineer":
        "I want to learn network security, cryptography, penetration testing "
        "and how to defend systems against attacks.",
    "devops_cloud":
        "I want to containerize applications with Docker, orchestrate with "
        "Kubernetes and deploy to AWS cloud infrastructure.",
}


class Failures:
    def __init__(self) -> None:
        self.items: List[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")
            self.items.append(f"{label}{(' — ' + detail) if detail else ''}")
        return ok


def _is_acyclic(edges: pd.DataFrame) -> Tuple[bool, List[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for r in edges.itertuples():
        adj[r.prereq_skill_id].add(r.skill_id)
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = defaultdict(int)

    for start in list(adj):
        if colour[start] != WHITE:
            continue
        stack = [(start, iter(adj[start]))]
        colour[start] = GREY
        path = [start]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                colour[node] = BLACK
                stack.pop()
                path.pop()
                continue
            if colour[nxt] == GREY:
                return False, path + [nxt]
            if colour[nxt] == WHITE:
                colour[nxt] = GREY
                path.append(nxt)
                stack.append((nxt, iter(adj[nxt])))
    return True, []


def smoke_test(top_k: int = 10) -> pd.DataFrame:
    """Nearest-neighbour recommendations for hand-written profiles."""
    emb = pd.read_parquet(paths.AGENT / "embeddings.parquet")
    index = pd.read_parquet(paths.AGENT / "arm_index.parquet")
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    feats = pd.read_parquet(paths.AGENT / "arm_features.parquet")[["arm_index", "course_id"]]
    meta = feats.merge(courses[["course_id", "title_normalized", "domain_family",
                                "level_canonical", "platform"]], on="course_id")

    arm_vecs = emb.drop(columns=["arm_index"]).to_numpy(np.float32)
    # Re-project the profile text through the SAME frozen SVD basis; a fresh
    # projection would put profiles and arms in different spaces.
    components = np.load(paths.AGENT / "svd_projection_svd64-v1.npy")
    raw = llm.embed(list(PROFILES.values()), llm.config()["models"]["embed"])
    prof = raw @ components.T

    def unit(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.clip(n, 1e-9, None)

    sims = unit(prof) @ unit(arm_vecs).T
    rows = []
    for pi, name in enumerate(PROFILES):
        order = np.argsort(-sims[pi])[:top_k]
        for rank, ai in enumerate(order, 1):
            arm = int(emb.iloc[ai]["arm_index"])
            m = meta[meta["arm_index"] == arm]
            rows.append({
                "profile": name, "rank": rank, "arm_index": arm,
                "score": round(float(sims[pi, ai]), 4),
                "title": m["title_normalized"].iloc[0] if len(m) else "?",
                "family": m["domain_family"].iloc[0] if len(m) else None,
            })
    return pd.DataFrame(rows)


def run() -> bool:
    f = Failures()
    print("=== validation ===")

    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    skills = pd.read_parquet(paths.CATALOG / "skills.parquet")
    cs = pd.read_parquet(paths.CATALOG / "course_skills_final.parquet")
    crosswalk = pd.read_parquet(paths.CATALOG / "tag_crosswalk.parquet")
    sources = pd.read_parquet(paths.CATALOG / "sources.parquet")
    index = pd.read_parquet(paths.AGENT / "arm_index.parquet")
    feats = pd.read_parquet(paths.AGENT / "arm_features.parquet")
    prereqs = pd.read_parquet(paths.WORLD / "skill_prereqs.parquet")
    spec = yaml.safe_load((paths.AGENT / "arm_feature_spec.yaml").read_text())

    # identity
    f.check(index["arm_index"].is_unique and index["dedup_cluster_id"].is_unique,
            "arm_index is a bijection onto dedup clusters")
    active = index[index["arm_status"] == "active"]
    live = courses[(courses["in_scope"] == True) & courses["is_canonical"]]  # noqa: E712
    f.check(len(active) == len(live),
            "active arms == canonical in-scope courses",
            f"{len(active)} vs {len(live)}")
    f.check(courses["course_id"].is_unique, "course_id is unique")
    per_cluster = courses.groupby("dedup_cluster_id")["is_canonical"].sum()
    f.check(set(per_cluster.unique()) == {1}, "exactly one canonical row per cluster")

    # referential integrity
    f.check(set(cs["skill_id"]) <= set(skills["skill_id"]),
            "every course_skills.skill_id resolves in skills")
    mapped = crosswalk[crosswalk["action"] == "map"]
    f.check(set(mapped["skill_id"].dropna()) <= set(skills["skill_id"]),
            "every mapped crosswalk entry resolves in skills")
    f.check(set(cs["course_id"]) <= set(live["course_id"]),
            "skills attach only to in-scope canonical courses")

    # DAG
    acyclic, cycle = _is_acyclic(prereqs)
    f.check(acyclic, "skill_prereqs is acyclic", " -> ".join(cycle[:5]))
    f.check(set(prereqs["prereq_skill_id"]) <= set(skills["skill_id"])
            and set(prereqs["skill_id"]) <= set(skills["skill_id"]),
            "every DAG endpoint resolves in skills")

    # features
    X = feats.drop(columns=["arm_index", "course_id"])
    f.check(len(feats) == len(active), "feature rows == active arms")
    f.check(not X.isna().any().any(), "no NaN in the feature matrix")
    f.check(spec["n_features"] == X.shape[1], "spec dimension matches matrix")
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(
        feats["arm_index"].to_numpy(np.int64)).tobytes()).hexdigest()
    f.check(h == spec["row_order_sha256"], "row-order hash matches the spec")
    f.check(feats["arm_index"].is_unique, "feature matrix carries unique arm_index")

    # imputation flags
    lvl_missing = courses["level_canonical"].isna() & (courses["in_scope"] == True)  # noqa: E712
    f.check(bool(courses.loc[lvl_missing, "level_is_imputed"].fillna(True).all()),
            "every missing level carries level_is_imputed")

    # firewall
    paths.reset_read_log()
    from . import stage10_features
    try:
        stage10_features.run()
        paths.assert_no_world_reads("stage10_features")
        firewall_ok = True
        detail = ""
    except AssertionError as exc:
        firewall_ok, detail = False, str(exc)
    f.check(firewall_ok, "agent/ build reads nothing from world/", detail)

    # licence gate
    nonredist = set(sources[~sources["redistributable"]]["source_id"])
    exported = set(feats.columns)
    f.check(not any(c in exported for c in ("description", "syllabus", "title_raw")),
            "no raw source text exported into agent/ artefacts",
            f"non-redistributable sources: {sorted(nonredist)}")

    # --- smoke test -------------------------------------------------------
    print("\n=== downstream smoke test (nearest neighbour over frozen embeddings) ===")
    recs = smoke_test()
    overlaps = []
    tops = {p: set(g["arm_index"]) for p, g in recs.groupby("profile")}
    names = list(tops)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = tops[names[i]], tops[names[j]]
            overlaps.append(len(a & b) / len(a | b))
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    f.check(mean_overlap < 0.25,
            "profiles return distinguishable recommendations",
            f"mean Jaccard overlap {mean_overlap:.3f}")
    recs.to_csv(paths.REPORTS / "smoke_test_recommendations.csv", index=False)

    for profile, grp in recs.groupby("profile", sort=False):
        print(f"\n  [{profile}]")
        for r in grp.head(5).itertuples():
            print(f"     {r.score:.3f}  {str(r.title)[:60]:<60} ({r.family})")
    print(f"\n  mean pairwise top-10 overlap: {mean_overlap:.3f} "
          f"(lower is better; >0.25 suggests the text pipeline is degenerate)")

    # --- datasheet --------------------------------------------------------
    ds = [
        "# Datasheet — course catalog", "",
        f"Dataset version: `{paths.dataset_version()}`  ",
        f"Feature spec: `{spec['spec_version']}`  Projection: `{spec['projection_version']}`", "",
        "## Composition", "",
        f"- source rows: {int(sources['row_count'].sum()):,}",
        f"- canonical courses: {len(courses):,}",
        f"- in scope: {len(live):,}",
        f"- arms (frozen index): {len(active):,}",
        f"- skills: {len(skills):,} across {skills['family'].nunique()} families",
        f"- skill assignments: {len(cs):,}",
        f"- prerequisite edges: {len(prereqs):,}", "",
        "## Sources", "", "| source | licence | redistributable | rows |", "|---|---|---|---|",
    ]
    for r in sources.itertuples():
        ds.append(f"| {r.source_id} | {r.license} | {r.redistributable} | {r.row_count:,} |")
    ds += [
        "", "## Known limitations", "",
    ]
    # Derived at write time. These were hardcoded prose and went stale as the
    # pipeline changed -- the shipped datasheet claimed 40.8% damaged text and
    # 2,188 arms long after both numbers had moved, and a hand-edit to the
    # leakage item would have been silently reverted by the next run.
    damaged = float(courses["description"].fillna("").str.contains("\ufffd").mean())
    with_skill = cs["course_id"].nunique()
    n_arms = len(active)
    no_skill = max(n_arms - int(live["course_id"].isin(cs["course_id"]).sum()), 0)
    # Plain definition: the share of (arm, skill) cells that are zero.
    sparsity = 1.0 - len(cs) / max(n_arms * len(skills), 1)
    effort = float(live["effort_hours_total"].notna().mean())
    per_family = skills.groupby("family").size().sort_values()
    thinnest = ", ".join(f"`{k}` ({v} leaves)" for k, v in per_family.head(2).items())
    platforms = sorted(set(live["platform"].dropna())) or ["unknown"]
    ds += [
        "1. Both the skill taxonomy and the prerequisite DAG are "
        "**model-generated**, not validated against a curriculum or expert.",
        f"2. **{damaged:.1%} of descriptions contain U+FFFD** replacement "
        "characters, baked in upstream at scrape time and unrecoverable.",
        "3. **Provider leakage resolved by construction.** The arm pool is a "
        f"single platform ({', '.join(platforms)}); edX was dropped because its "
        "disjoint field coverage made the provider perfectly predictable from "
        "the arm features (AUC 1.000), which would have made any reward term "
        "over duration or rating partly a provider term.",
        f"4. **{no_skill:,} of {n_arms:,} arms carry no skill** through the "
        f"frozen taxonomy; the arm-by-skill matrix is {sparsity:.1%} sparse. "
        "(Stage 7 extracted at least one raw skill phrase for 97.9% of arms; "
        "the gap is what the taxonomy crosswalk kept.)",
        "5. The catalog predates the GenAI wave, so the thinnest families are "
        f"{thinnest}.",
        f"6. Duration coverage: **{effort:.1%}** of in-scope arms have an "
        "effort-hours total, after donor merges.",
        "7. Scope precision/recall and taxonomy precision are **not measured**; "
        "audit samples with empty `human_label` columns are emitted instead.",
        "", "## Intended use", "",
        "Item space for a contextual-bandit vs content-based-baseline "
        "comparison. The taxonomy and DAG are scaffolding, not contributions.",
        "", "## Firewall", "",
        "`data/world/` (prerequisite DAG, study_plans) must never be read when "
        "building `data/agent/` features. Asserted in the build.",
    ]
    (paths.REPORTS / "DATASHEET.md").write_text("\n".join(ds) + "\n", encoding="utf-8")

    hashing.record_stage("validate", failures=len(f.items), passed=len(f.items) == 0)
    print(f"\n=== {'ALL CHECKS PASSED' if not f.items else str(len(f.items)) + ' FAILURE(S)'} ===")
    for item in f.items:
        print("  -", item)
    return not f.items


def main() -> None:
    argparse.ArgumentParser(description="Stage 11: validate").parse_args()
    raise SystemExit(0 if run() else 1)


if __name__ == "__main__":
    main()
