"""Stage 10 -- freeze arm identity and materialise item features.

Three artefacts, all under data/agent/:

  arm_index          arm_index -> dedup_cluster_id. FROZEN and append-only.
  arm_feature_spec   the versioned feature contract (names, transforms,
                     imputation rules) -- auditable, not implicit in code
  arm_features       the matrix, carrying arm_index as a column plus a
                     row-order hash the loader asserts against

FIREWALL: this module reads ONLY data/catalog/. It must never touch
data/world/ -- not skill_prereqs, not study plans. Prerequisite structure has
to stay a latent property the bandit discovers from features; if it were
handed over as a feature while the reward credited prerequisite
satisfaction, the bandit would win by construction. The rule is asserted at
the bottom of run() rather than left as a comment.

Why arm_index is keyed on dedup_cluster_id and not course_id: re-running
dedup can change which source row is canonical. Keyed on the row, a
re-ingest would silently re-point trained per-arm parameters at a different
course, and nothing would error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from . import hashing, llm, paths

SPEC_VERSION = "arm-spec-v1"
PROJECTION_VERSION = "svd64-v1"
SVD_DIMS = 64
RATING_PRIOR_WEIGHT = 25.0     # empirical-Bayes smoothing constant

# Inputs this stage is permitted to read. Anything under world/ is a bug.
ALLOWED_INPUTS = ("courses.parquet", "skills.parquet", "course_skills_final.parquet")


def _log1p(series: pd.Series) -> np.ndarray:
    return np.log1p(pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(float))


def build_arm_index(courses: pd.DataFrame) -> pd.DataFrame:
    """Freeze arm identity, append-only.

    An existing index is never renumbered. New clusters get new indices;
    clusters that vanish are marked retired rather than deleted, so a
    retired index is never recycled for a different course.
    """
    live = courses[courses["in_scope"] == True]  # noqa: E712
    live = live[live["is_canonical"]].copy()
    clusters = sorted(live["dedup_cluster_id"].unique())

    path = paths.AGENT / "arm_index.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        known = dict(zip(existing["dedup_cluster_id"], existing["arm_index"]))
        next_idx = int(existing["arm_index"].max()) + 1
    else:
        existing, known, next_idx = None, {}, 0

    rows = []
    for cluster in clusters:
        if cluster in known:
            rows.append({"arm_index": known[cluster], "dedup_cluster_id": cluster,
                         "arm_status": "active",
                         "first_seen_dataset_version": None})
        else:
            rows.append({"arm_index": next_idx, "dedup_cluster_id": cluster,
                         "arm_status": "active",
                         "first_seen_dataset_version": paths.dataset_version()})
            next_idx += 1

    index = pd.DataFrame(rows)
    if existing is not None:
        gone = existing[~existing["dedup_cluster_id"].isin(clusters)].copy()
        if len(gone):
            gone["arm_status"] = "retired"
            index = pd.concat([index, gone], ignore_index=True)
        # Preserve original first-seen stamps.
        first_seen = dict(zip(existing["dedup_cluster_id"],
                              existing["first_seen_dataset_version"]))
        index["first_seen_dataset_version"] = index.apply(
            lambda r: first_seen.get(r["dedup_cluster_id"],
                                     r["first_seen_dataset_version"]), axis=1)
    return index.sort_values("arm_index").reset_index(drop=True)


def _one_hot(series: pd.Series, values: List[str], prefix: str
             ) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    cols, spec = [], []
    filled = series.fillna("__missing__").astype(str)
    for v in values:
        cols.append((filled == v).to_numpy(float))
        spec.append({"name": f"{prefix}__{v}", "transform": "one_hot",
                     "source_field": prefix, "imputation_rule": "absent->0"})
    return np.column_stack(cols), spec


def run() -> pd.DataFrame:
    paths.reset_read_log()
    courses = paths.load_parquet(paths.CATALOG / "courses.parquet")
    skills = paths.load_parquet(paths.CATALOG / "skills.parquet")
    course_skills = paths.load_parquet(paths.CATALOG / "course_skills_final.parquet")

    index = build_arm_index(courses)
    active = index[index["arm_status"] == "active"]

    live = courses[(courses["in_scope"] == True) & courses["is_canonical"]]  # noqa: E712
    df = active.merge(live, on="dedup_cluster_id", how="inner").sort_values("arm_index")
    df = df.reset_index(drop=True)

    blocks: List[np.ndarray] = []
    spec: List[Dict[str, Any]] = []

    # --- difficulty ------------------------------------------------------
    lvl = pd.to_numeric(df["level_canonical"], errors="coerce")
    for v in (0, 1, 2):
        blocks.append((lvl == v).fillna(False).to_numpy(float)[:, None])
        spec.append({"name": f"level__{v}", "transform": "one_hot",
                     "source_field": "level_canonical",
                     "imputation_rule": "null->all zeros"})
    # Mandatory: without this the model cannot tell a declared Beginner from
    # an unknown level imputed as Beginner, and missingness is provider
    # correlated, so the reward model becomes partly a provider classifier.
    blocks.append(df["level_is_imputed"].fillna(True).astype(float).to_numpy()[:, None])
    spec.append({"name": "level_is_imputed", "transform": "indicator",
                 "source_field": "level_is_imputed", "imputation_rule": "null->1"})

    # --- duration --------------------------------------------------------
    hours = pd.to_numeric(df["effort_hours_total"], errors="coerce")
    blocks.append(np.log1p(hours.fillna(0.0).to_numpy(float))[:, None])
    spec.append({"name": "log1p_effort_hours", "transform": "log1p",
                 "source_field": "effort_hours_total",
                 "imputation_rule": "null->0, paired with has_effort_hours"})
    blocks.append(hours.notna().to_numpy(float)[:, None])
    spec.append({"name": "has_effort_hours", "transform": "indicator",
                 "source_field": "effort_hours_total", "imputation_rule": "-"})
    weeks = pd.to_numeric(df["calendar_weeks_lo"], errors="coerce")
    blocks.append(np.log1p(weeks.fillna(0.0).to_numpy(float))[:, None])
    spec.append({"name": "log1p_calendar_weeks", "transform": "log1p",
                 "source_field": "calendar_weeks_lo",
                 "imputation_rule": "null->0; kept separate from effort hours"})
    blocks.append(weeks.notna().to_numpy(float)[:, None])
    spec.append({"name": "has_calendar_weeks", "transform": "indicator",
                 "source_field": "calendar_weeks_lo", "imputation_rule": "-"})

    # --- popularity ------------------------------------------------------
    rating = pd.to_numeric(df["rating_mean"], errors="coerce")
    count = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0.0)
    # Empirical-Bayes shrinkage toward the PROVIDER mean, not the global one:
    # provider means differ materially and edX has no ratings at all.
    prov_mean = rating.groupby(df["platform"]).transform("mean")
    prior = prov_mean.fillna(rating.mean())
    shrunk = ((count * rating.fillna(prior)) + RATING_PRIOR_WEIGHT * prior) / (
        count + RATING_PRIOR_WEIGHT)
    blocks.append(shrunk.fillna(prior.mean()).to_numpy(float)[:, None])
    spec.append({"name": "rating_shrunk", "transform": "empirical_bayes",
                 "source_field": "rating_mean",
                 "imputation_rule": f"shrink to provider mean, weight {RATING_PRIOR_WEIGHT}"})
    blocks.append(df["rating_is_missing"].fillna(True).astype(float).to_numpy()[:, None])
    spec.append({"name": "rating_is_missing", "transform": "indicator",
                 "source_field": "rating_is_missing", "imputation_rule": "null->1"})
    blocks.append(_log1p(df["rating_count"])[:, None])
    spec.append({"name": "log1p_rating_count", "transform": "log1p",
                 "source_field": "rating_count", "imputation_rule": "null->0"})
    blocks.append(_log1p(df["enrollment_count"])[:, None])
    spec.append({"name": "log1p_enrollment", "transform": "log1p",
                 "source_field": "enrollment_count", "imputation_rule": "null->0"})
    blocks.append(df["enrollment_count"].notna().to_numpy(float)[:, None])
    spec.append({"name": "has_enrollment", "transform": "indicator",
                 "source_field": "enrollment_count", "imputation_rule": "-"})

    # --- categoricals ----------------------------------------------------
    for field, values in [
        ("access_model", ["free", "audit_free_cert_paid", "paid_one_time", "subscription"]),
        ("pace", ["self_paced", "scheduled"]),
        ("platform", sorted(df["platform"].dropna().unique().tolist())),
    ]:
        mat, sp = _one_hot(df[field], values, field)
        blocks.append(mat)
        spec.extend(sp)

    fam_values = sorted(skills["family"].unique().tolist())
    mat, sp = _one_hot(df["domain_family"], fam_values, "family")
    blocks.append(mat)
    spec.extend(sp)

    # --- skills multi-hot -------------------------------------------------
    skill_ids = skills["skill_id"].tolist()
    sidx = {s: i for i, s in enumerate(skill_ids)}
    aidx = {c: i for i, c in enumerate(df["course_id"])}
    skill_mat = np.zeros((len(df), len(skill_ids)), dtype=np.float32)
    for r in course_skills.itertuples():
        i, j = aidx.get(r.course_id), sidx.get(r.skill_id)
        if i is not None and j is not None:
            skill_mat[i, j] = 1.0
    blocks.append(skill_mat)
    for s in skill_ids:
        spec.append({"name": f"skill__{s}", "transform": "multi_hot",
                     "source_field": "course_skills_final",
                     "imputation_rule": "absent->0"})
    blocks.append((skill_mat.sum(axis=1) > 0).astype(float)[:, None])
    spec.append({"name": "has_any_skill", "transform": "indicator",
                 "source_field": "course_skills_final", "imputation_rule": "-"})

    # --- text embedding, projected and FROZEN ----------------------------
    text = (df["title_normalized"].fillna("") + ". " +
            df["description"].fillna("").str.slice(0, 600))
    raw = llm.embed(text.tolist(), llm.config()["models"]["embed"])
    # SVD fit once here and persisted. Refitting on a later ingest would
    # rotate the basis and silently drift every arm feature -- a
    # non-stationarity nobody intended.
    proj_path = paths.AGENT / f"svd_projection_{PROJECTION_VERSION}.npy"
    if proj_path.exists():
        components = np.load(proj_path)
    else:
        centred = raw - raw.mean(axis=0, keepdims=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[:SVD_DIMS]
        proj_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proj_path, components)
    with warnings.catch_warnings():
        # Spurious numpy/Accelerate BLAS warnings on macOS; verified
        # numerically clean elsewhere in this codebase.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        emb = raw @ components.T
    blocks.append(emb.astype(np.float32))
    for d in range(SVD_DIMS):
        spec.append({"name": f"text_svd_{d:02d}", "transform": f"svd:{PROJECTION_VERSION}",
                     "source_field": "title+description",
                     "imputation_rule": "empty text -> embedding of ' '"})

    matrix = np.hstack([b.astype(np.float32) for b in blocks])
    assert not np.isnan(matrix).any(), "NaN in the arm feature matrix"
    assert matrix.shape[1] == len(spec), (matrix.shape, len(spec))

    features = pd.DataFrame(matrix, columns=[s["name"] for s in spec])
    features.insert(0, "arm_index", df["arm_index"].to_numpy())
    features.insert(1, "course_id", df["course_id"].to_numpy())

    row_hash = hashlib.sha256(
        np.ascontiguousarray(df["arm_index"].to_numpy(np.int64)).tobytes()).hexdigest()

    spec_doc = {
        "spec_version": SPEC_VERSION,
        "projection_version": PROJECTION_VERSION,
        "dataset_version": paths.dataset_version(),
        "n_arms": int(len(df)),
        "n_features": int(matrix.shape[1]),
        "row_order_sha256": row_hash,
        "features": [dict(s, index=i) for i, s in enumerate(spec)],
    }
    (paths.AGENT / "arm_feature_spec.yaml").write_text(
        yaml.safe_dump(spec_doc, sort_keys=False), encoding="utf-8")

    hashing.write_parquet(index, paths.AGENT / "arm_index.parquet", stage="features")
    hashing.write_parquet(features, paths.AGENT / "arm_features.parquet",
                          stage="features", row_order_sha256=row_hash,
                          spec_version=SPEC_VERSION)
    emb_df = pd.DataFrame(emb.astype(np.float32),
                          columns=[f"e{d:02d}" for d in range(SVD_DIMS)])
    emb_df.insert(0, "arm_index", df["arm_index"].to_numpy())
    hashing.write_parquet(emb_df, paths.AGENT / "embeddings.parquet",
                          stage="features", projection_version=PROJECTION_VERSION)

    # --- firewall assertion ----------------------------------------------
    # Asserts on the paths actually read, not on source text.
    paths.assert_no_world_reads("stage10_features")

    lines = [
        "# Arm features (agent/)", "",
        f"- arms: **{len(df):,}** active (index frozen, append-only)",
        f"- features: **{matrix.shape[1]}** dimensions",
        f"- spec version: `{SPEC_VERSION}` | projection: `{PROJECTION_VERSION}`",
        f"- row-order sha256: `{row_hash[:16]}...`", "",
        "## Feature blocks", "", "| block | dims |", "|---|---|",
        f"| difficulty (one-hot + imputed flag) | 4 |",
        f"| duration (effort, calendar, indicators) | 4 |",
        f"| popularity (shrunk rating, counts, indicators) | 5 |",
        f"| access / pace / platform | "
        f"{4 + 2 + df['platform'].nunique()} |",
        f"| family one-hot | {len(fam_values)} |",
        f"| skills multi-hot | {len(skill_ids)} |",
        f"| text embedding (frozen SVD) | {SVD_DIMS} |", "",
        "## Firewall", "",
        "This stage reads only `data/catalog/`. It never touches",
        "`data/world/` -- no prerequisite edges, no study plans. Asserted in",
        "code at the end of `run()`, not left as a comment. Prerequisite",
        "satisfaction must stay a latent property the bandit discovers from",
        "features; handing it over directly while the reward credits it",
        "would let the bandit win by construction.", "",
        "## Dimensionality note", "",
        f"d = {matrix.shape[1]} over {len(df):,} arms, dominated by the",
        f"{len(skill_ids)}-dim skill multi-hot. Linear-bandit regret scales",
        "with d, so for the actual policy you will likely want interaction",
        "features (user-skill cosine, level gap, prereq coverage) computed",
        "from these columns rather than feeding all of them as context. The",
        "raw multi-hot is kept because those interaction features need it.",
    ]
    (paths.REPORTS / "arm_features.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")

    hashing.record_stage("features", arms=len(df), dims=int(matrix.shape[1]),
                         spec_version=SPEC_VERSION, row_order_sha256=row_hash)
    print(f"[features] arms={len(df):,} dims={matrix.shape[1]} "
          f"(skills {len(skill_ids)}, text {SVD_DIMS})")
    print(f"[features] spec={SPEC_VERSION} projection={PROJECTION_VERSION}")
    print(f"[features] row-order hash {row_hash[:16]}...")
    print("[features] firewall assertion passed: no world/ inputs")
    return features


def main() -> None:
    argparse.ArgumentParser(description="Stage 10: arm index and features").parse_args()
    run()


if __name__ == "__main__":
    main()
