"""Stage 14 -- contextual bandit (LinUCB) vs the content baseline.

Shared-weight linear UCB over INTERACTION features. Not disjoint per-arm
parameters: with 1,899 arms and a few thousand interactions each arm would
see one or two pulls, and nothing would be learned. Shared weights also
make an unseen arm scorable from its features, which is what cold start
needs.

The agent never sees the reward spec, the prerequisite DAG, or any latent
learner field. It sees a context-action feature vector and a scalar reward.
Prerequisite structure stays latent.

Episodes run for `horizon` steps per learner, and the environment transitions
state between steps, so the bandit is acting in an MDP while modelling it as
stateless. That gap is the point: the MDP planner in the next stage is the
same setup with lookahead added, which isolates the value of planning.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import hashing, llm, paths
from .environment import Environment, LearnerState

warnings.filterwarnings("ignore", category=RuntimeWarning)

POLICY_ID = "linucb_v1"
FEATURES = ["topic", "need", "too_hard", "too_easy", "time_over",
            "rating_c", "has_level", "has_hours", "bias"]


class Featuriser:
    """Context-action features. Observable information only."""

    def __init__(self, env: Environment) -> None:
        self.env = env
        emb = pd.read_parquet(paths.AGENT / "embeddings.parquet")
        v = emb.drop(columns=["arm_index"]).to_numpy(np.float32)
        self.vecs = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        self.components = np.load(paths.AGENT / "svd_projection_svd64-v1.npy")
        self._cache: Dict[str, np.ndarray] = {}

    def goal_vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            raw = llm.embed([text], llm.config()["models"]["embed"])
            g = raw @ self.components.T
            self._cache[text] = (g / np.clip(np.linalg.norm(g, axis=1, keepdims=True),
                                             1e-9, None))[0]
        return self._cache[text]

    def topic(self, goal_text: str) -> np.ndarray:
        return self.vecs @ self.goal_vec(goal_text)

    def matrix(self, state: LearnerState, cand: np.ndarray,
               topic: np.ndarray) -> np.ndarray:
        e = self.env
        idx = np.array([e.pos[int(a)] for a in cand])
        need_vec = np.array([1.0 if s in state.needed() else 0.0
                             for s in e.skill_ids], dtype=np.float32)
        need = (e.skill_mat[idx] @ need_vec) / max(float(need_vec.sum()), 1.0)

        lvl = e.level[idx]
        known = ~np.isnan(lvl)
        gap = np.nan_to_num(lvl) - state.level()
        too_hard = np.where(known, np.clip(gap, 0, None), 0.0)
        too_easy = np.where(known, np.clip(-gap, 0, None), 0.0)

        hrs = e.hours[idx]
        hk = ~np.isnan(hrs)
        budget = max(state.weekly_hours * 4.0, 1e-6)
        over = np.where(hk, np.clip((np.nan_to_num(hrs) - budget) / budget, 0, 2), 0.0)

        rat = e.rating[idx]
        rk = ~np.isnan(rat)
        rating_c = np.where(rk, np.nan_to_num(rat) - 4.0, 0.0)

        return np.column_stack([
            topic[idx], need, too_hard, too_easy, over, rating_c,
            known.astype(float), hk.astype(float), np.ones(len(idx))]).astype(float)


class LinUCB:
    def __init__(self, d: int, alpha: float = 0.6, lam: float = 1.0) -> None:
        self.A = lam * np.eye(d)
        self.b = np.zeros(d)
        self.alpha = alpha

    def scores(self, X: np.ndarray) -> np.ndarray:
        Ainv = np.linalg.inv(self.A)
        theta = Ainv @ self.b
        mean = X @ theta
        bonus = self.alpha * np.sqrt(np.einsum("ij,jk,ik->i", X, Ainv, X))
        return mean + bonus

    def update(self, x: np.ndarray, r: float) -> None:
        self.A += np.outer(x, x)
        self.b += r * x

    def theta(self) -> np.ndarray:
        return np.linalg.inv(self.A) @ self.b


def _baseline_pick(F: np.ndarray) -> int:
    """Fixed-weight content score over the same features."""
    w = np.array([1.00, 0.60, -0.25, -0.25, -0.15, 0.0, 0.0, 0.0, 0.0])
    return int(np.argmax(F @ w))


def run(n_users: Optional[int] = None, horizon: int = 5,
        cand_size: int = 60, alpha: float = 0.6, seed: int = 7) -> pd.DataFrame:
    env = Environment()
    fz = Featuriser(env)
    users = pd.read_parquet(paths.WORLD / "users.parquet")
    if n_users:
        users = users.head(n_users)
    rng = np.random.default_rng(seed)

    agent = LinUCB(len(FEATURES), alpha=alpha)
    rows: List[Dict[str, Any]] = []

    for policy in ("baseline", "linucb"):
        # Fresh agent per policy; identical user order and candidate sets.
        if policy == "linucb":
            agent = LinUCB(len(FEATURES), alpha=alpha)
        prng = np.random.default_rng(seed)
        for u in users.itertuples():
            labels = list(u.goal_labels) if u.goal_labels is not None else []
            text = f"{' '.join(map(str, labels))}. {u.background or ''}".strip()
            topic = fz.topic(text or "learn programming")
            state = env.reset(u.user_id)
            for t in range(horizon):
                cand = prng.choice(env.arms, size=cand_size, replace=False)
                F = fz.matrix(state, cand, topic)
                if policy == "baseline":
                    pick = _baseline_pick(F)
                else:
                    pick = int(np.argmax(agent.scores(F)))
                arm = int(cand[pick])
                r, state, info = env.step(state, arm, float(topic[env.pos[arm]]))
                if policy == "linucb":
                    agent.update(F[pick], r)
                rows.append({"policy": policy, "user_id": u.user_id, "t": t,
                             "arm_index": arm, "reward": round(r, 4),
                             "progress": info["progress_after"],
                             "absorption": info["absorption"]})

    log = pd.DataFrame(rows)
    hashing.write_parquet(log, paths.AGENT / "bandit_log.parquet", stage="bandit")

    summ = log.groupby("policy").agg(
        mean_reward=("reward", "mean"),
        total_reward=("reward", "sum"),
        final_progress=("progress", "last")).round(4)
    final = (log[log["t"] == horizon - 1].groupby("policy")["progress"].mean().round(4))
    theta = dict(zip(FEATURES, np.round(agent.theta(), 3)))

    lines = [
        "# Contextual bandit vs content baseline", "",
        f"- pre-registered bundle: `{env.bundle_hash[:16]}`",
        f"- learners: **{len(users):,}** | horizon: **{horizon}** | "
        f"candidate set: **{cand_size}** | alpha: **{alpha}**", "",
        "## Outcome", "", "| policy | mean reward | total reward | mean final goal progress |",
        "|---|---|---|---|",
    ]
    for p in ("baseline", "linucb"):
        lines.append(f"| `{p}` | {summ.loc[p,'mean_reward']:.4f} | "
                     f"{summ.loc[p,'total_reward']:.1f} | {final[p]:.4f} |")
    lift = (summ.loc["linucb", "mean_reward"] - summ.loc["baseline", "mean_reward"])
    lines += [
        "", f"Reward lift: **{lift:+.4f}** "
        f"({lift / abs(summ.loc['baseline','mean_reward']):+.1%})", "",
        "## Learned weights", "", "| feature | theta |", "|---|---|",
    ]
    for k, v in theta.items():
        lines.append(f"| `{k}` | {v:+.3f} |")
    lines += [
        "", "Sign check: `too_hard` should be more negative than `too_easy`",
        "if the agent recovered the reward's asymmetry from experience alone —",
        "it never saw the spec.", "",
        "## What this does and does not show", "",
        "The bandit models the problem as stateless, but the environment",
        "transitions: taking a course raises mastery, so step *t+1* faces a",
        "different learner. A bandit cannot represent 'take this now so that",
        "becomes viable later'. The MDP planner stage is the same setup with",
        "lookahead added, which isolates the value of planning from the value",
        "of learning.",
    ]
    (paths.REPORTS / "bandit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("bandit", policy_id=POLICY_ID, users=len(users),
                         horizon=horizon, lift=float(lift))
    print(f"[bandit] users={len(users)} horizon={horizon} cand={cand_size}")
    print(summ.to_string())
    print(f"[bandit] mean final goal progress: {dict(final)}")
    print(f"[bandit] reward lift {lift:+.4f}")
    print(f"[bandit] learned theta: {theta}")
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 14: contextual bandit")
    ap.add_argument("--users", type=int)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cand", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.6)
    a = ap.parse_args()
    run(a.users, a.horizon, a.cand, a.alpha)


if __name__ == "__main__":
    main()
