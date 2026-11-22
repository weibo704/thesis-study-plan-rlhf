"""Stage 16 -- multi-seed experiment and paired analysis.

Runs the three-policy comparison across seeds and reports PAIRED
per-learner differences. Pairing matters: every learner is run under every
policy, so the within-learner difference removes the (large) variance from
learners differing in goal depth, readiness and available hours. Comparing
unpaired means across seeds wastes most of the power -- at 150 learners it
left a +0.042 effect inside a +-0.078 interval.

A single seed is not evidence. An earlier run reported "the planner wins"
off seed 7 (+0.108), while seed 23 gave -0.032. The sign flipped.
"""
from __future__ import annotations

import argparse
import glob
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import hashing, paths
from .stage15_planner import run as run_policies

POLICIES = ("baseline", "linucb", "planner")
CONTRASTS = (
    ("linucb", "baseline", "learning"),
    ("planner", "linucb", "planning"),
    ("planner", "baseline", "learning + planning"),
)


def _paired(per: pd.DataFrame, a: str, b: str, metric: str) -> Dict[str, Any]:
    pa = per[per["policy"] == a].set_index(["seed", "user_id"])[metric]
    pb = per[per["policy"] == b].set_index(["seed", "user_id"])[metric]
    d = (pa - pb).dropna()
    n = len(d)
    mean = float(d.mean())
    half = float(1.96 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return {"mean": mean, "ci95": half, "n": n,
            "win_rate": float((d > 0).mean()),
            "significant": abs(mean) > half}


def run(n_users: int = 500, horizon: int = 8, seeds: List[int] = None,
        prereg: str = "poc-v2", cand: int = 60) -> pd.DataFrame:
    seeds = seeds or [7, 11, 23, 37, 53]
    frames = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n=== seed {seed} ({i}/{len(seeds)}) ===", flush=True)
        log = run_policies(n_users=n_users, horizon=horizon, cand_size=cand,
                           seed=seed, prereg=prereg)
        log["seed"] = seed
        frames.append(log)

    log = pd.concat(frames, ignore_index=True)
    hashing.write_parquet(log, paths.AGENT / f"experiment_{prereg}.parquet",
                          stage="experiment")

    per = log.groupby(["seed", "policy", "user_id"]).agg(
        reward=("reward", "sum"), progress=("progress", "last")).reset_index()

    means = per.groupby("policy")[["reward", "progress"]].mean().round(4).loc[list(POLICIES)]
    by_seed = per.groupby(["seed", "policy"])["reward"].mean().unstack().round(4)

    lines = [
        f"# Experiment — `{prereg}`", "",
        f"- learners: **{n_users}** | horizon **{horizon}** | "
        f"seeds **{seeds}** | learner-runs **{len(per)//len(POLICIES):,}** per policy", "",
        "## Means", "", "| policy | mean total reward | mean final goal progress |",
        "|---|---|---|",
    ]
    for p in POLICIES:
        lines.append(f"| `{p}` | {means.loc[p,'reward']:.4f} | {means.loc[p,'progress']:.4f} |")

    lines += ["", "## Paired per-learner contrasts", "",
              "Every learner runs under every policy, so the within-learner",
              "difference removes variance from learners differing in goal",
              "depth, readiness and hours. A 95% interval that spans zero",
              "means the effect is not established, regardless of the sign.", ""]
    results = {}
    for metric in ("reward", "progress"):
        lines += [f"### {metric}", "",
                  "| contrast | difference | 95% CI | win rate | verdict |",
                  "|---|---|---|---|---|"]
        for a, b, label in CONTRASTS:
            r = _paired(per, a, b, metric)
            results[(metric, label)] = r
            verdict = "**established**" if r["significant"] else "not established"
            lines.append(f"| {label} | {r['mean']:+.4f} | ±{r['ci95']:.4f} | "
                         f"{r['win_rate']:.0%} | {verdict} |")
        lines.append("")

    lines += ["## Reward mean by seed", "",
              "| seed | " + " | ".join(f"`{p}`" for p in POLICIES) + " |",
              "|---|" + "---|" * len(POLICIES)]
    for seed in by_seed.index:
        lines.append(f"| {seed} | " + " | ".join(
            f"{by_seed.loc[seed, p]:.4f}" for p in POLICIES) + " |")
    lines += ["", "Per-seed means are shown because a single seed is not",
              "evidence: an earlier 150-learner run reported the planner",
              "ahead on seed 7 (+0.108) while seed 23 gave −0.032."]

    (paths.REPORTS / f"experiment_{prereg}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== MEANS ===")
    print(means.to_string())
    print("\n=== PAIRED CONTRASTS ===")
    for (metric, label), r in results.items():
        flag = "ESTABLISHED" if r["significant"] else "not established"
        print(f"  {metric:<9} {label:<21} {r['mean']:+.4f} ± {r['ci95']:.4f}  "
              f"wins {r['win_rate']:.0%}  {flag}")

    hashing.record_stage("experiment", prereg=prereg, seeds=seeds,
                         users=n_users, horizon=horizon)
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 16: multi-seed experiment")
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 11, 23, 37, 53])
    ap.add_argument("--prereg", default="poc-v2")
    a = ap.parse_args()
    run(a.users, a.horizon, a.seeds, a.prereg)


if __name__ == "__main__":
    main()
