"""Stage 15 -- MDP planner vs bandit vs baseline.

Three policies on one environment, isolating two different things:

    baseline   fixed weights, no learning, no planning
    linucb     learns a reward model, acts myopically   -> value of LEARNING
    planner    same learned model, plus lookahead       -> value of PLANNING

The planner is model-based with a LEARNED reward: it reuses LinUCB's theta
as r-hat, simulates the state transition for each candidate, and adds the
discounted best next-step value.

    score(a) = r_hat(s, a) + gamma * max_a' r_hat(s', a')

The transition model is the AGENT'S OWN and uses only observable data: a
course teaches the skills listed in `course_skills`, so taking it raises
mastery of those skills. It never reads the prerequisite DAG. Whether
ordering matters is something the planner has to discover through reward,
not something it is told -- which is the whole point of the firewall.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import hashing, paths
from .environment import Environment
from .stage14_bandit import FEATURES, Featuriser, LinUCB, _baseline_pick

warnings.filterwarnings("ignore", category=RuntimeWarning)

GAMMA = 0.85
LOOKAHEAD_TOP = 12      # candidates worth simulating
NEXT_SAMPLE = 25        # successor actions evaluated per simulated state
ABSORPTION_GUESS = 0.45  # the agent's belief about how much a course teaches


class Planner:
    """Lookahead over the agent's own transition model."""

    def __init__(self, env: Environment, fz: Featuriser, theta: np.ndarray) -> None:
        self.env, self.fz, self.theta = env, fz, theta

    def _simulate(self, state, arm: int) -> Dict[str, float]:
        """Predicted mastery after taking `arm` — observable info only."""
        e = self.env
        i = e.pos[int(arm)]
        nxt = dict(state.mastery)
        for j in np.nonzero(e.skill_mat[i])[0]:
            sid = e.skill_ids[j]
            cur = nxt.get(sid, 0.0)
            nxt[sid] = min(1.0, cur + ABSORPTION_GUESS * (1.0 - cur))
        return nxt

    def pick(self, state, cand: np.ndarray, topic: np.ndarray,
             rng: np.random.Generator) -> int:
        F = self.fz.matrix(state, cand, topic)
        r_hat = F @ self.theta
        top = np.argsort(-r_hat)[:LOOKAHEAD_TOP]

        # ONE successor sample shared by every candidate. Drawing a fresh
        # sample per candidate made the lookahead vary by 0.8x the magnitude
        # of the immediate reward -- sampling noise competing with signal,
        # not planning.
        nxt_cand = rng.choice(self.env.arms, size=NEXT_SAMPLE, replace=False)

        best_i, best_v = int(top[0]), -np.inf
        saved_mastery, saved_taken = state.mastery, set(state.taken)
        for i in top:
            arm = int(cand[i])
            state.mastery = self._simulate(state, arm)
            state.taken = saved_taken | {arm}
            Fn = self.fz.matrix(state, nxt_cand, topic)
            v_next = float((Fn @ self.theta).max())
            state.mastery, state.taken = saved_mastery, saved_taken

            total = float(r_hat[i]) + GAMMA * v_next
            if total > best_v:
                best_v, best_i = total, int(i)
        return best_i


def run(n_users: Optional[int] = None, horizon: int = 5, cand_size: int = 60,
        alpha: float = 0.6, seed: int = 7, prereg: str = "poc-v2") -> pd.DataFrame:
    env = Environment(prereg=prereg)
    fz = Featuriser(env)
    users = pd.read_parquet(paths.WORLD / "users.parquet")
    if n_users:
        users = users.head(n_users)

    texts, topics = {}, {}
    for u in users.itertuples():
        labels = list(u.goal_labels) if u.goal_labels is not None else []
        t = f"{' '.join(map(str, labels))}. {u.background or ''}".strip() or "learn programming"
        texts[u.user_id] = t
        topics[u.user_id] = fz.topic(t)

    rows: List[Dict[str, Any]] = []
    thetas: Dict[str, np.ndarray] = {}

    for policy in ("baseline", "linucb", "planner"):
        agent = LinUCB(len(FEATURES), alpha=alpha)
        if policy == "planner":
            # Plan with the model LinUCB already learned, so the only
            # difference from the bandit is lookahead.
            agent.A = thetas["_A"].copy()
            agent.b = thetas["_b"].copy()
            planner = Planner(env, fz, agent.theta())
        prng = np.random.default_rng(seed)
        for u in users.itertuples():
            topic = topics[u.user_id]
            state = env.reset(u.user_id)
            for t in range(horizon):
                cand = prng.choice(env.arms, size=cand_size, replace=False)
                F = fz.matrix(state, cand, topic)
                if policy == "baseline":
                    pick = _baseline_pick(F)
                elif policy == "linucb":
                    pick = int(np.argmax(agent.scores(F)))
                else:
                    pick = planner.pick(state, cand, topic, prng)
                arm = int(cand[pick])
                r, state, info = env.step(state, arm, float(topic[env.pos[arm]]))
                if t == horizon - 1:
                    # Sparse episode-end payout (zero unless the bundle
                    # sets w_goal_terminal).
                    r += env.terminal_reward(state)
                if policy == "linucb":
                    agent.update(F[pick], r)
                rows.append({"policy": policy, "user_id": u.user_id, "t": t,
                             "arm_index": arm, "reward": round(r, 4),
                             "progress": info["progress_after"]})
        if policy == "linucb":
            thetas["_A"], thetas["_b"] = agent.A.copy(), agent.b.copy()
            thetas["linucb"] = agent.theta()

    log = pd.DataFrame(rows)
    log["prereg"] = prereg
    hashing.write_parquet(log, paths.AGENT / f"policy_log_{prereg}.parquet",
                          stage="planner")
    hashing.write_parquet(log, paths.AGENT / f"policy_log_{prereg}_seed{seed}.parquet",
                          stage="planner")

    per_user = log.groupby(["policy", "user_id"]).agg(
        total_reward=("reward", "sum"),
        final_progress=("progress", "last")).reset_index()
    summ = per_user.groupby("policy").agg(
        mean_total_reward=("total_reward", "mean"),
        mean_final_progress=("final_progress", "mean")).round(4)
    order = ["baseline", "linucb", "planner"]
    summ = summ.loc[order]

    lines = [
        f"# Policy comparison — `{prereg}`", "",
        f"- bundle: `{env.bundle_hash[:16]}`",
        f"- learners: **{len(users):,}** | horizon **{horizon}** | "
        f"candidates **{cand_size}** | gamma **{GAMMA}**", "",
        "| policy | learns | plans | mean total reward | mean final goal progress |",
        "|---|---|---|---|---|",
    ]
    flags = {"baseline": ("no", "no"), "linucb": ("yes", "no"), "planner": ("yes", "yes")}
    for p in order:
        lf, pf = flags[p]
        lines.append(f"| `{p}` | {lf} | {pf} | "
                     f"{summ.loc[p,'mean_total_reward']:.3f} | "
                     f"{summ.loc[p,'mean_final_progress']:.4f} |")
    lines += [
        "", "## Reading the contrasts", "",
        "- `linucb` − `baseline` isolates the value of **learning** the reward.",
        "- `planner` − `linucb` isolates the value of **planning**: identical "
        "learned model, lookahead added.", "",
        "The planner's transition model is its own and uses only observable "
        "data (which skills a course teaches). It never reads the prerequisite "
        "DAG — whether ordering pays off is something it must discover.",
    ]
    (paths.REPORTS / f"policies_{prereg}.md").write_text("\n".join(lines) + "\n",
                                                         encoding="utf-8")
    hashing.record_stage(f"planner_{prereg}", users=len(users), horizon=horizon)
    print(f"[planner] prereg={prereg} users={len(users)} horizon={horizon}")
    print(summ.to_string())
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 15: MDP planner comparison")
    ap.add_argument("--users", type=int)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cand", type=int, default=60)
    ap.add_argument("--prereg", default="poc-v2")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.users, a.horizon, a.cand, seed=a.seed, prereg=a.prereg)


if __name__ == "__main__":
    main()
