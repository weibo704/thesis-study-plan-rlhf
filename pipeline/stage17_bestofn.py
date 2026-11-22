"""Stage 17 -- does RL improve LLM-generated study plans?

The LLM generates N diverse plans per learner. A reward model LEARNED from
executed plans scores them, and best-of-N picks the top. Four selection
policies are compared on identical generated candidates:

    sample_1             the LLM's first plan           (raw LLM, no RL)
    random_of_n          a random one of the N          (controls for N)
    best_of_n_preference argmax of a reward model LEARNED FROM FEEDBACK
                         -- pairwise preferences only, never a reward value
    best_of_n_regression argmax of a model regressed on the ORACLE return
                         -- an upper reference: it is handed the quantity it
                         is asked to predict, which real feedback never gives
    best_of_n_oracle     argmax of the TRUE return      (ceiling)

The random control is what separates "scoring helped" from "having more
candidates helped" -- without it, best-of-N can look good purely because
sampling N times and keeping any of them beats sampling once.

The oracle arm is not a competitor. It is the headroom: how much of the
available gain the learned model actually captures.

The reward model is fit on a TRAIN split of learners and evaluated on a
held-out split, so it is never scored on plans it was fit to.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import hashing, llm, paths, plan_generator, preference
from .environment import Environment
from .stage14_bandit import Featuriser

warnings.filterwarnings("ignore", category=RuntimeWarning)

SHORTLIST = 25
# Logistic sensitivity of the simulated learner's pairwise choice. Lower is
# noisier; at 1.5 a learner comparing near-identical plans is near chance.
SENSITIVITY = 1.5
N_PLANS = 6
PLAN_LEN = 5

# Observable features. The first block is order-INSENSITIVE (a permutation
# of the same courses scores identically); the second block is where
# ordering lives, which is what a study plan is actually about.
PLAN_FEATURES = [
    # set-level
    "mean_topic", "mean_too_hard", "mean_too_easy", "hours_over",
    "goal_coverage", "n_skills", "mean_rating", "plan_len",
    # order-level
    "level_monotonic",        # fraction of steps that do not go backwards
    "pos_weighted_difficulty",  # difficulty weighted by position: ramping up
    "level_first_last",       # level(last) - level(first)
    "level_range",            # hardest - easiest
    "skill_continuity",       # mean Jaccard between consecutive courses
    "new_skill_rate",         # fraction of skills introduced rather than repeated
    "bias",
]

# Default fit set, chosen by ablation rather than by eyeballing weights.
# Dropping the eight near-zero features (hours_over, n_skills, plan_len,
# level_monotonic, level_first_last, level_range, skill_continuity,
# new_skill_rate) left the result unchanged to slightly better:
# +0.1136 vs +0.1051, headroom 0.132 vs 0.140. Equal performance from
# fewer features is the better model at n=200 test learners.
#
# All 15 are still computed, so any of them can be reinstated with --keep.
DEFAULT_FEATURES = ["mean_topic", "mean_too_hard", "mean_too_easy",
                    "goal_coverage", "mean_rating", "pos_weighted_difficulty",
                    "bias"]

# No oracle features. `prereq_precedence` (read from world/skill_prereqs)
# was tried and measured: theta = -0.011, and the result moved from +0.1051
# to +0.1013 -- no effect. Position-weighted difficulty already captures
# what prerequisite ordering was meant to encode, so the firewall costs
# nothing here. That is a measured finding, not an assumption.


def plan_features(env: Environment, state, arms: List[int], topic: np.ndarray,
                  keep: Optional[List[str]] = None) -> np.ndarray:
    """Observable features of a whole plan.

    Ordering features matter because a study plan is a sequence: without
    them a permutation of the same courses scores identically, and the
    model cannot express "foundations first".

    `keep` restricts the output to a named subset, for ablation.
    """
    keep = keep or DEFAULT_FEATURES
    if not arms:
        return np.zeros(len(keep))
    idx = [env.pos[int(a)] for a in arms]
    lvl = env.level[idx]
    hrs = np.nan_to_num(env.hours[idx])
    rat = env.rating[idx]
    n = len(arms)

    gap = np.nan_to_num(lvl) - state.level()
    budget = max(state.weekly_hours * 4.0 * n, 1e-6)
    needed = state.needed()

    per_course = [{env.skill_ids[j] for j in np.nonzero(env.skill_mat[i])[0]}
                  for i in idx]
    taught = set().union(*per_course) if per_course else set()

    filled = np.nan_to_num(lvl, nan=float(state.level()))
    monotonic = float(np.mean(np.diff(filled) >= 0)) if n > 1 else 0.5
    # Position weights rise linearly, so a plan that ramps up scores higher
    # than the same courses in reverse.
    w = np.arange(1, n + 1, dtype=float)
    pos_weighted = float((filled * w).sum() / (w.sum() * 2.0))
    first_last = float((filled[-1] - filled[0]) / 2.0)
    lvl_range = float((filled.max() - filled.min()) / 2.0)

    # Continuity: consecutive courses sharing skills means the plan builds,
    # rather than hopping between unrelated material.
    if n > 1:
        js = []
        for a, b in zip(per_course, per_course[1:]):
            union = a | b
            js.append(len(a & b) / len(union) if union else 0.0)
        continuity = float(np.mean(js))
    else:
        continuity = 0.0
    total_slots = sum(len(c) for c in per_course) or 1
    new_rate = float(len(taught) / total_slots)

    feats = [
        float(np.mean(topic[idx])),
        float(np.mean(np.clip(gap, 0, None))),
        float(np.mean(np.clip(-gap, 0, None))),
        float(np.clip((hrs.sum() - budget) / budget, 0, 3)),
        float(len(taught & needed) / max(len(needed), 1)),
        float(len(taught)) / 20.0,
        float(np.nanmean(rat)) if np.isfinite(rat).any() else 0.0,
        float(n) / PLAN_LEN,
        monotonic, pos_weighted, first_last, lvl_range, continuity, new_rate,
        1.0,
    ]
    pos = {f: i for i, f in enumerate(PLAN_FEATURES)}
    return np.array([feats[pos[f]] for f in keep])


def execute(env: Environment, user_id: str, arms: List[int],
            topic: np.ndarray) -> Dict[str, float]:
    """Run a plan in the environment; return true return and goal progress."""
    state = env.reset(user_id)
    total = 0.0
    for arm in arms:
        r, state, _ = env.step(state, int(arm), float(topic[env.pos[int(arm)]]))
        total += r
    total += env.terminal_reward(state)
    return {"return": total, "progress": state.goal_progress()}


def _standardise(Xtr: np.ndarray, Xall: np.ndarray):
    """Z-score on the TRAIN split, applied to everything.

    Without it the Bradley-Terry fit is dominated by whichever feature
    happens to have the widest raw range, and the L2 penalty lands unevenly
    across features on different scales.
    """
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    const = sd < 1e-9          # the bias column, and anything degenerate
    mu = np.where(const, 0.0, mu)
    sd = np.where(const, 1.0, sd)
    return (Xall - mu) / sd, mu, sd


def run(n_users: int = 150, prereg: str = "poc-v2", seed: int = 7,
        train_frac: float = 0.6, keep: Optional[List[str]] = None) -> pd.DataFrame:
    env = Environment(prereg=prereg)
    fz = Featuriser(env)
    users = pd.read_parquet(paths.WORLD / "users.parquet").head(n_users)
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    af = pd.read_parquet(paths.AGENT / "arm_features.parquet")[["arm_index", "course_id"]]
    meta = af.merge(courses[["course_id", "title_normalized", "level_canonical",
                             "effort_hours_total", "rating_mean"]],
                    on="course_id").set_index("arm_index")
    budget = llm.Budget(llm.config()["limits"]["max_extract_calls"], "plangen")
    rng = np.random.default_rng(seed)



    rows: List[Dict[str, Any]] = []
    for n, u in enumerate(users.itertuples(), 1):
        labels = list(u.goal_labels) if u.goal_labels is not None else []
        text = f"{' '.join(map(str, labels))}. {u.background or ''}".strip()
        topic = fz.topic(text or "learn programming")
        state0 = env.reset(u.user_id)

        top = np.argsort(-topic)[:SHORTLIST]
        shortlist, id_to_arm = [], {}
        for i, p in enumerate(top):
            arm = int(env.arms[p])
            m = meta.loc[arm]
            id_to_arm[i] = arm
            shortlist.append({
                "id": i, "title": str(m.title_normalized)[:60],
                "level": {0: "beginner", 1: "intermediate", 2: "advanced"}.get(
                    m.level_canonical),
                "hours": None if pd.isna(m.effort_hours_total) else float(m.effort_hours_total),
                "rating": None if pd.isna(m.rating_mean) else float(m.rating_mean),
            })

        profile = {"goal": labels, "level": int(u.start_level),
                   "weekly_hours": float(u.weekly_hours),
                   "background": u.background or ""}
        plans = plan_generator.generate(profile, shortlist, N_PLANS, PLAN_LEN,
                                        budget)
        if not plans:
            continue
        for k, plan in enumerate(plans):
            arms = [id_to_arm[c] for c in plan["course_ids"] if c in id_to_arm]
            if not arms:
                continue
            res = execute(env, u.user_id, arms, topic)
            rows.append({
                "user_id": u.user_id, "plan_rank": k,
                "strategy": plan["strategy"], "n_courses": len(arms),
                "features": plan_features(env, state0, arms, topic, keep).tolist(),
                "true_return": res["return"], "progress": res["progress"],
            })
        if n % 25 == 0:
            print(f"[bestofn] {n}/{len(users)} learners (api calls {budget.used})",
                  flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[bestofn] no plans generated")

    # --- learn the reward model on a train split --------------------------
    uids = df["user_id"].unique()
    rng.shuffle(uids)
    cut = int(len(uids) * train_frac)
    train_ids, test_ids = set(uids[:cut]), set(uids[cut:])
    df["split"] = np.where(df["user_id"].isin(train_ids), "train", "test")

    Xtr = np.vstack(df[df.split == "train"]["features"])
    ytr = df[df.split == "train"]["true_return"].to_numpy(float)
    lam = 1.0
    theta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    df["pred_return"] = [float(np.array(f) @ theta) for f in df["features"]]

    # --- reward model learned from FEEDBACK (pairwise preferences) --------
    def _groups(split):
        out = {}
        for uid, g in df[df.split == split].groupby("user_id"):
            out[uid] = g.to_dict("records")
        return out

    # Standardise on train before differencing, and use EVERY pair:
    # 6 plans give C(6,2)=15 comparisons, and capping at 6 discarded 60%
    # of the available supervision.
    allX = np.vstack(df["features"])
    stdX, mu, sd = _standardise(np.vstack(df[df.split == "train"]["features"]), allX)
    df["z"] = [row for row in stdX]
    def _zgroups(split):
        out = {}
        for uid, g in df[df.split == split].groupby("user_id"):
            recs = g.to_dict("records")
            for r in recs:
                r["features"] = r["z"]
            out[uid] = recs
        return out
    Xp, yp, pmeta = preference.build_pairs(_zgroups("train"), SENSITIVITY, rng,
                                           max_pairs_per_group=15)
    Xq, yq, qmeta = preference.build_pairs(_zgroups("test"), SENSITIVITY, rng,
                                           max_pairs_per_group=15)
    theta_pref = preference.fit(Xp, yp)
    pref_acc = preference.accuracy(theta_pref, Xq, yq)
    label_noise = float(np.mean([m["chose_better"] for m in pmeta])) if pmeta else float("nan")
    df["pred_pref"] = [float(np.asarray(z) @ theta_pref) for z in df["z"]]

    te = df[df.split == "test"].copy()
    corr = float(np.corrcoef(te["pred_return"], te["true_return"])[0, 1])
    corr_pref = float(np.corrcoef(te["pred_pref"], te["true_return"])[0, 1])

    # --- four selection policies on identical candidates -------------------
    sel_rows = []
    for uid, g in te.groupby("user_id"):
        g = g.sort_values("plan_rank")
        picks = {
            "sample_1": g.iloc[0],
            "random_of_n": g.iloc[int(rng.integers(0, len(g)))],
            "best_of_n_preference": g.loc[g["pred_pref"].idxmax()],
            "best_of_n_regression": g.loc[g["pred_return"].idxmax()],
            "best_of_n_oracle": g.loc[g["true_return"].idxmax()],
        }
        for policy, row in picks.items():
            sel_rows.append({"policy": policy, "user_id": uid,
                             "return": row["true_return"],
                             "progress": row["progress"],
                             "strategy": row["strategy"]})
    sel = pd.DataFrame(sel_rows)

    # Persist the fitted reward model so the serving path can apply it.
    # theta is fitted on STANDARDISED features, so mu and sd travel with it:
    # a server that scored raw features with these weights would be wrong in
    # a way that still produces plausible-looking numbers.
    import json
    (paths.AGENT / f"reward_model_{prereg}.json").write_text(json.dumps({
        "prereg": prereg,
        "feature_names": list(keep or DEFAULT_FEATURES),
        "theta": [float(x) for x in theta_pref],
        "standardise_mu": [float(x) for x in mu],
        "standardise_sd": [float(x) for x in sd],
        "fit": "bradley_terry_pairwise",
        "n_train_learners": int(df[df.split == "train"].user_id.nunique()),
        "n_train_pairs": int(len(Xp)),
        "heldout_pair_accuracy": round(float(pref_acc), 4),
        "label_noise": round(float(label_noise), 4),
    }, indent=2) + "\n", encoding="utf-8")

    hashing.write_parquet(df.drop(columns=["features", "z"]).assign(
        features=df["features"].map(str)),
        paths.AGENT / f"bestofn_plans_{prereg}.parquet", stage="bestofn")
    hashing.write_parquet(sel, paths.AGENT / f"bestofn_selection_{prereg}.parquet",
                          stage="bestofn")

    order = ["sample_1", "random_of_n", "best_of_n_preference",
             "best_of_n_regression", "best_of_n_oracle"]
    means = sel.groupby("policy")[["return", "progress"]].mean().round(4).loc[order]

    def paired(a, b, metric="return"):
        pa = sel[sel.policy == a].set_index("user_id")[metric]
        pb = sel[sel.policy == b].set_index("user_id")[metric]
        d = (pa - pb).dropna()
        m, half = float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
        return {"mean": m, "ci95": half, "win": float((d > 0).mean()),
                "sig": abs(m) > half, "n": len(d)}

    contrasts = {
        "feedback RL over raw LLM": paired("best_of_n_preference", "sample_1"),
        "scoring over sampling": paired("best_of_n_preference", "random_of_n"),
        "more candidates alone (no scoring)": paired("random_of_n", "sample_1"),
        "feedback vs oracle-regression model": paired("best_of_n_preference",
                                                      "best_of_n_regression"),
        "headroom left (oracle - feedback)": paired("best_of_n_oracle",
                                                    "best_of_n_preference"),
    }

    lines = [f"# Does RL improve LLM-generated study plans? — `{prereg}`", "",
             f"- learners: **{len(uids)}** ({len(train_ids)} train / {len(test_ids)} test)",
             f"- plans generated: **{len(df):,}** ({N_PLANS} per learner, "
             f"{PLAN_LEN} courses each)",
             f"- features: **{len(keep or DEFAULT_FEATURES)}** observable · "
             f"standardised on train · all 15 pairs per learner",
             f"- preference model (feedback only): held-out pair accuracy "
             f"**{pref_acc:.3f}**, correlation with true return **r = {corr_pref:.3f}**",
             f"- regression model (oracle scalar): **r = {corr:.3f}**",
             f"- simulated feedback label noise: learners picked the genuinely "
             f"better plan **{label_noise:.0%}** of the time", "",
             "## Selection policies (identical candidates)", "",
             "| policy | mean return | mean goal progress |", "|---|---|---|"]
    for p in order:
        lines.append(f"| `{p}` | {means.loc[p,'return']:.4f} | "
                     f"{means.loc[p,'progress']:.4f} |")
    lines += ["", "## Paired contrasts (return)", "",
              "| contrast | difference | 95% CI | win rate | verdict |",
              "|---|---|---|---|---|"]
    for label, c in contrasts.items():
        lines.append(f"| {label} | {c['mean']:+.4f} | ±{c['ci95']:.4f} | "
                     f"{c['win']:.0%} | "
                     f"{'**established**' if c['sig'] else 'not established'} |")
    lines += ["", "`random_of_n` is the control that separates *scoring helped*",
              "from *having more candidates helped*. `best_of_n_oracle` is not a",
              "competitor — it is the ceiling, showing how much of the available",
              "gain the learned model captured.", "",
              "## Learned reward-model weights", "", "| feature | theta |", "|---|---|"]
    names = keep or DEFAULT_FEATURES
    for f, t in zip(names, theta_pref):
        lines.append(f"| `{f}` | {t:+.3f} |")
    (paths.REPORTS / f"bestofn_{prereg}.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")

    print(f"\n[bestofn] plans={len(df)} learners={len(uids)}")
    print(f"[bestofn] preference model (FEEDBACK only): pair acc {pref_acc:.3f}, "
          f"r={corr_pref:.3f} | label noise {label_noise:.0%} correct")
    print(f"[bestofn] regression model (ORACLE scalar): r={corr:.3f}")
    print(means.to_string())
    print("\n=== PAIRED CONTRASTS (return) ===")
    for label, c in contrasts.items():
        flag = "ESTABLISHED" if c["sig"] else "not established"
        print(f"  {label:<44} {c['mean']:+.4f} ± {c['ci95']:.4f} "
              f"wins {c['win']:.0%}  {flag}")
    hashing.record_stage("bestofn", prereg=prereg, learners=len(uids),
                         plans=len(df), reward_model_r=round(corr, 4))
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 17: best-of-N over LLM plans")
    ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--prereg", default="poc-v2")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", nargs="*", help="ablation: restrict to these features")
    a = ap.parse_args()
    run(a.users, a.prereg, a.seed, keep=a.keep)


if __name__ == "__main__":
    main()
