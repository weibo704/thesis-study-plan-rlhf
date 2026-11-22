"""Stage 5 -- deduplicate the spine.

Duplicate arms split reward mass between identical actions, so this is a
correctness stage rather than tidiness. The cascade is deliberately
deterministic wherever possible:

  1. decisive key   -- identical url_slug on the same platform => same course
  2. blocked scoring -- TF-IDF char n-gram neighbours, then lexical scores
  3. three bands    -- auto-merge / review / auto-reject
  4. union-find     -- transitive closure, order-independent by construction

Only the review band needs human judgement, and it is written out as a CSV
rather than resolved silently.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import hashing, paths
from .textnorm import UnionFind, merge_key


def _cfg() -> Dict[str, Any]:
    with open(paths.CONFIG / "thresholds.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["dedup"]


def _match_text(row: pd.Series) -> str:
    return " ".join(
        str(x) for x in (row.get("title_normalized"), row.get("institution")) if x
    )


def candidate_pairs(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[Tuple[int, int, float]]:
    """Blocked candidate generation: exact comparison of all pairs is O(n^2),
    so restrict to TF-IDF nearest neighbours before scoring."""
    texts = df.apply(_match_text, axis=1).tolist()
    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(
        analyzer=cfg["tfidf_analyzer"],
        ngram_range=(cfg["tfidf_ngram_min"], cfg["tfidf_ngram_max"]),
        min_df=1,
    )
    matrix = vec.fit_transform(texts)
    k = min(cfg["neighbours_per_row"] + 1, len(texts))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(matrix)
    distances, indices = nn.kneighbors(matrix)

    out: List[Tuple[int, int, float]] = []
    seen = set()
    for i, (drow, irow) in enumerate(zip(distances, indices)):
        for dist, j in zip(drow, irow):
            if i == j:
                continue
            cosine = 1.0 - float(dist)
            if cosine < cfg["min_cosine_candidate"]:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b, cosine))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out


def classify(cosine: float, tsr: float, cfg: Dict[str, Any]) -> str:
    am, rv = cfg["auto_merge"], cfg["review"]
    if cosine >= am["min_cosine"] and tsr >= am["min_token_set_ratio"]:
        return "auto_merge"
    if cosine >= rv["min_cosine"] and tsr >= rv["min_token_set_ratio"]:
        return "review"
    return "auto_reject"


def run(report_bands: bool = False) -> pd.DataFrame:
    cfg = _cfg()
    df = pd.read_parquet(paths.STAGING / "merged.parquet").reset_index(drop=True)
    uf = UnionFind(df["course_id"].tolist())

    decisions: List[Dict[str, Any]] = []

    # --- 1. decisive keys -------------------------------------------------
    decisive_merges = 0
    for key in cfg["decisive_keys"]:
        if key not in df.columns:
            continue
        sub = df[df[key].notna()]
        for (_, _), grp in sub.groupby([sub["platform"], sub[key]]):
            ids = grp["course_id"].tolist()
            for other in ids[1:]:
                uf.union(ids[0], other)
                decisive_merges += 1
                decisions.append({
                    "course_id_a": ids[0], "course_id_b": other,
                    "cosine": 1.0, "token_set_ratio": 100.0,
                    "band": "decisive", "key": key,
                })

    # --- 2/3. blocked scoring and banding --------------------------------
    pairs = candidate_pairs(df, cfg)
    titles = df["title_normalized"].fillna("").tolist()
    ids = df["course_id"].tolist()
    band_counts = {"auto_merge": 0, "review": 0, "auto_reject": 0}

    for i, j, cosine in pairs:
        tsr = float(fuzz.token_set_ratio(titles[i], titles[j]))
        band = classify(cosine, tsr, cfg)
        band_counts[band] += 1
        if band == "auto_merge":
            uf.union(ids[i], ids[j])
        if band != "auto_reject":
            decisions.append({
                "course_id_a": ids[i], "course_id_b": ids[j],
                "title_a": titles[i], "title_b": titles[j],
                "cosine": round(cosine, 4), "token_set_ratio": tsr,
                "band": band, "key": "",
            })

    # --- 4. assign clusters and canonical rows ---------------------------
    df["dedup_cluster_id"] = [uf.find(cid) for cid in df["course_id"]]
    # Canonical = most complete row, ties broken by course_id for determinism.
    df["_completeness"] = df.notna().sum(axis=1)
    # Select by positional index, not by course_id: ids are unique after
    # stage 3, but keying on position keeps this correct even if that
    # invariant is ever weakened upstream.
    order = df.sort_values(["dedup_cluster_id", "_completeness", "course_id"],
                           ascending=[True, False, True])
    canonical_pos = set(order.groupby("dedup_cluster_id").head(1).index)
    df["is_canonical"] = df.index.isin(canonical_pos)
    df = df.drop(columns=["_completeness"])

    n_clusters = df["dedup_cluster_id"].nunique()
    collapsed = len(df) - n_clusters

    dec = pd.DataFrame(decisions)
    review = dec[dec["band"] == "review"] if len(dec) else dec
    if len(review):
        review.sort_values("cosine", ascending=False).to_csv(
            paths.REPORTS / "dedup_review.csv", index=False
        )

    hashing.write_parquet(df, paths.STAGING / "deduped.parquet", stage="dedup")
    if len(dec):
        hashing.write_parquet(dec, paths.REPORTS / "dedup_decisions.parquet", stage="dedup")

    total_pairs = sum(band_counts.values())
    review_share = (band_counts["review"] / total_pairs) if total_pairs else 0.0
    lines = [
        "# Dedup report", "",
        f"- rows in: **{len(df)}**",
        f"- clusters out: **{n_clusters}**  (collapsed {collapsed} duplicate rows)",
        f"- decisive merges (identical url_slug): **{decisive_merges}**", "",
        "## Candidate pair bands", "",
        "| band | pairs | share |", "|---|---|---|",
    ]
    for band in ("auto_merge", "review", "auto_reject"):
        share = (band_counts[band] / total_pairs) if total_pairs else 0.0
        lines.append(f"| {band} | {band_counts[band]} | {share:.1%} |")
    lines += [
        "", f"Review band is **{review_share:.1%}** of scored pairs.",
        "Plan section 3d targets 10-15%: outside that range, recalibrate",
        "`config/thresholds.yaml` against hand-labelled pairs before trusting",
        "the auto bands.", "",
        "Review-band pairs are written to `reports/dedup_review.csv` for",
        "manual adjudication; they are NOT merged automatically.",
    ]
    (paths.REPORTS / "dedup_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("dedup", rows=len(df), clusters=n_clusters,
                         collapsed=collapsed, bands=band_counts)
    print(f"[dedup] rows={len(df)} clusters={n_clusters} collapsed={collapsed} "
          f"decisive={decisive_merges}")
    print(f"[dedup] bands: {band_counts}  review_share={review_share:.1%}")
    if report_bands and len(dec):
        print(dec.groupby("band")[["cosine", "token_set_ratio"]].describe().to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5: dedup cascade")
    ap.add_argument("--report-bands", action="store_true",
                    help="print score distributions to help calibrate thresholds")
    args = ap.parse_args()
    run(report_bands=args.report_bands)


if __name__ == "__main__":
    main()
