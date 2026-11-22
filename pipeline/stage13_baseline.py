"""Stage 13 -- content-based baseline policy.

A fixed, non-learning scorer: the thing the bandit has to beat. It sees only
what a real recommender could see.

OBSERVABLE (the policy's context, by definition — a user types their goal,
states their level, says how many hours they have, and can say what they
already know):
    goal labels, start_level, weekly_hours, self-reported mastery

HIDDEN (oracle):
    skill_prereqs, study plans, reward_spec, and the latent user fields
    (readiness, persistence, novelty_pref, price_sensitivity, prereq_gap)

Notably the policy does NOT get prerequisite coverage. An earlier note of
mine suggested it as an interaction feature; that was wrong. The reward
credits prerequisite satisfaction, so a policy that could see coverage
would win by construction and the comparison would measure nothing.
Prerequisite satisfaction has to stay latent — something the policy
discovers through skill and level features, or fails to.

The weights below are the BASELINE'S OWN design, deliberately naive: match
the topic, prefer skills the learner still needs, mildly prefer courses
near their level. They are not the reward's weights, and the baseline is
never tuned against reward output.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import hashing, llm, paths

warnings.filterwarnings("ignore", category=RuntimeWarning)

POLICY_ID = "content_baseline_v1"

# A plausible naive content recommender, not the reward.
W_TOPIC = 1.00     # semantic match between the stated goal and the course
W_NEED = 0.60      # overlap with skills the learner still lacks
W_LEVEL = 0.25     # mild preference for courses near the learner's level
W_TIME = 0.15      # mild preference for courses that fit the weekly budget


@dataclass
class UserContext:
    user_id: str
    goal_text: str
    start_level: int
    weekly_hours: float
    mastered: set            # skill_ids the learner self-reports knowing
    needed: set              # goal skill_ids not yet mastered


def load_contexts(limit: Optional[int] = None) -> List[UserContext]:
    """Build contexts from the observable projection of world/users."""
    users = paths.load_parquet(paths.WORLD / "users.parquet")
    latent = paths.load_parquet(paths.WORLD / "user_latent_state.parquet")
    if limit:
        users = users.head(limit)
        latent = latent[latent["user_id"].isin(set(users["user_id"]))]

    mastered: Dict[str, set] = {}
    for r in latent.itertuples():
        if r.mastery >= 0.5:
            mastered.setdefault(r.user_id, set()).add(r.skill_id)

    out = []
    for r in users.itertuples():
        known = mastered.get(r.user_id, set())
        # Parquet round-trips list columns as numpy arrays, whose truthiness
        # is ambiguous -- coerce explicitly rather than relying on `or []`.
        goal = set(r.goal_skill_ids) if r.goal_skill_ids is not None else set()
        labels = list(r.goal_labels) if r.goal_labels is not None else []
        text = f"{' '.join(str(x) for x in labels)}. {r.background or ''}".strip()
        out.append(UserContext(
            user_id=r.user_id, goal_text=text or "learn programming",
            start_level=int(r.start_level), weekly_hours=float(r.weekly_hours),
            mastered=known, needed=goal - known))
    return out


class Baseline:
    def __init__(self) -> None:
        self.features = paths.load_parquet(paths.AGENT / "arm_features.parquet")
        self.emb = paths.load_parquet(paths.AGENT / "embeddings.parquet")
        courses = paths.load_parquet(paths.CATALOG / "courses.parquet")
        skills = paths.load_parquet(paths.CATALOG / "skills.parquet")

        self.arms = self.features["arm_index"].to_numpy()
        self.pos = {int(a): i for i, a in enumerate(self.arms)}
        vecs = self.emb.drop(columns=["arm_index"]).to_numpy(np.float32)
        self.vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        self.components = np.load(paths.AGENT / f"svd_projection_svd64-v1.npy")

        self.skill_cols = [c for c in self.features.columns if c.startswith("skill__")]
        self.skill_ids = [c[len("skill__"):] for c in self.skill_cols]
        self.skill_mat = self.features[self.skill_cols].to_numpy(np.float32)

        meta = self.features[["arm_index", "course_id"]].merge(
            courses[["course_id", "title_normalized", "domain_family",
                     "level_canonical", "effort_hours_total"]], on="course_id")
        self.meta = meta.set_index("arm_index")
        self.level = pd.to_numeric(meta["level_canonical"], errors="coerce").to_numpy(float)
        self.hours = pd.to_numeric(meta["effort_hours_total"], errors="coerce").to_numpy(float)

    def _goal_vec(self, text: str) -> np.ndarray:
        raw = llm.embed([text], llm.config()["models"]["embed"])
        v = raw @ self.components.T
        return (v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None))[0]

    def rank(self, ctx: UserContext, k: int = 10) -> pd.DataFrame:
        topic = self.vecs @ self._goal_vec(ctx.goal_text)

        need_vec = np.array([1.0 if s in ctx.needed else 0.0
                             for s in self.skill_ids], dtype=np.float32)
        denom = max(float(need_vec.sum()), 1.0)
        need = (self.skill_mat @ need_vec) / denom

        gap = np.nan_to_num(self.level - ctx.start_level)
        level_term = np.where(np.isnan(self.level), 0.0, -np.abs(gap))

        budget = max(ctx.weekly_hours * 4.0, 1e-6)
        over = np.clip((np.nan_to_num(self.hours) - budget) / budget, 0, 3)
        time_term = np.where(np.isnan(self.hours), 0.0, -over)

        total = (W_TOPIC * topic + W_NEED * need
                 + W_LEVEL * level_term + W_TIME * time_term)
        order = np.argsort(-total)[:k]
        return pd.DataFrame({
            "user_id": ctx.user_id,
            "rank": np.arange(1, len(order) + 1),
            "arm_index": self.arms[order],
            "score": np.round(total[order], 4),
            "topic": np.round(topic[order], 4),
            "need": np.round(need[order], 4),
            "level_term": np.round(level_term[order], 4),
            "time_term": np.round(time_term[order], 4),
            "policy_id": POLICY_ID,
        })


def run(limit: Optional[int] = None, k: int = 10) -> pd.DataFrame:
    paths.reset_read_log()
    contexts = load_contexts(limit)
    bl = Baseline()
    frames = [bl.rank(c, k) for c in contexts]
    recs = pd.concat(frames, ignore_index=True)

    # Policy-form firewall: user context allowed, oracle structure not.
    paths.assert_no_oracle_reads(POLICY_ID)

    hashing.write_parquet(recs, paths.AGENT / "baseline_recommendations.parquet",
                          stage="baseline")

    titles = bl.meta["title_normalized"]
    fam = bl.meta["domain_family"]
    top1 = recs[recs["rank"] == 1]
    lines = [
        "# Content-based baseline", "",
        f"- policy: `{POLICY_ID}` (fixed weights, no learning)",
        f"- users scored: **{recs['user_id'].nunique():,}**",
        f"- recommendations: **{len(recs):,}** (top-{k} each)",
        f"- distinct arms used at rank 1: **{top1['arm_index'].nunique():,}**", "",
        "## Weights (the baseline's own design, NOT the reward)", "",
        "| term | weight |", "|---|---|",
        f"| topic match | {W_TOPIC} |", f"| needed-skill overlap | {W_NEED} |",
        f"| level proximity | {W_LEVEL} |", f"| time fit | {W_TIME} |", "",
        "## What the policy can and cannot see", "",
        "Observable: goal labels, start level, weekly hours, self-reported",
        "mastery — all things a user states or a real system could ask.", "",
        "Hidden: `skill_prereqs`, study plans, `reward_spec`, and the latent",
        "user fields (readiness, persistence, novelty_pref, price_sensitivity,",
        "prereq_gap).", "",
        "**Prerequisite coverage is deliberately NOT a feature.** The reward",
        "credits prerequisite satisfaction, so a policy that could see",
        "coverage would win by construction. It has to stay latent.", "",
        "## Rank-1 family spread", "", "| family | times at rank 1 |", "|---|---|",
    ]
    for f, n in fam.reindex(top1["arm_index"]).value_counts().head(15).items():
        lines.append(f"| `{f}` | {n} |")
    (paths.REPORTS / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("baseline", users=int(recs["user_id"].nunique()),
                         recommendations=len(recs), policy_id=POLICY_ID)
    print(f"[baseline] users={recs['user_id'].nunique():,} "
          f"recs={len(recs):,} distinct rank-1 arms={top1['arm_index'].nunique():,}")
    print(f"[baseline] firewall: oracle structure not read")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 13: content baseline")
    ap.add_argument("--limit", type=int, help="score only the first N users")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    run(a.limit, a.k)


if __name__ == "__main__":
    main()
