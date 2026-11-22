"""Randomized-graph ablation — the go/no-go gate the plan asked for.

The plan called for this BEFORE building further on the prerequisite DAG,
and it never ran. It asks one question: does the agent-derived DAG carry
signal, or would any graph with the same shape do as well?

Three populations, identical in every respect except the graph their goals
and starting mastery derive from:

  real    the 254 agent-and-co-occurrence edges actually built
  random  degree-preserving double-edge swaps: every skill keeps its exact
          in- and out-degree, while WHICH skill precedes which is destroyed.
          Degree preservation does NOT preserve path depth -- random rewiring
          makes chains longer (mean plan depth 8.7 -> 12.8) -- so each learner
          is additionally matched to the random plan closest in depth to the
          real plan they would have received. Without that, the arm would be
          testing "deeper plans" as much as "different ordering".
  none    no edges at all; a goal with no prerequisite chain

Each learner keeps their user_id, start level, weekly hours, persistence
and budget across all three. Only the goal they are given and the mastery
they start with are re-derived. Each condition gets its own frozen bundle
with identical reward weights, so the only thing varying is the graph.

What a pass means: the DAG beats noise. It does NOT mean the comparison is
uncontaminated -- only the world/ -> agent/ firewall establishes that, and
that is an implementation invariant rather than a measurement.

Runs entirely in a scratch data root; the live data/ is never written.
No API calls: personas are reused, not regenerated.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

CONDITIONS = ("real", "random", "none")


# ----------------------------------------------------------------- shuffling
def degree_preserving_shuffle(dag: pd.DataFrame, rng: np.random.Generator,
                              passes: int = 40) -> pd.DataFrame:
    """Double-edge swaps that preserve every node's in- and out-degree.

    (a->b, c->d) becomes (a->d, c->b). Swaps that duplicate an existing
    edge, create a self-loop, or close a cycle are rejected and retried,
    so the result is still a DAG with the same degree sequence.
    """
    edges: List[Tuple[str, str]] = list(zip(dag["prereq_skill_id"], dag["skill_id"]))
    present: Set[Tuple[str, str]] = set(edges)
    succ: Dict[str, Set[str]] = {}
    for u, v in edges:
        succ.setdefault(u, set()).add(v)

    def reaches(src: str, dst: str) -> bool:
        seen, stack = {src}, [src]
        while stack:
            n = stack.pop()
            if n == dst:
                return True
            for m in succ.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return False

    attempts = passes * len(edges)
    swaps = 0
    for _ in range(attempts):
        i, j = rng.integers(0, len(edges), 2)
        if i == j:
            continue
        (a, b), (c, d) = edges[i], edges[j]
        if len({a, b, c, d}) < 4:
            continue
        if (a, d) in present or (c, b) in present:
            continue
        # tentatively apply, then reject if it closes a cycle
        for u, v in ((a, b), (c, d)):
            succ[u].discard(v); present.discard((u, v))
        succ.setdefault(a, set()).add(d); present.add((a, d))
        succ.setdefault(c, set()).add(b); present.add((c, b))
        if reaches(d, a) or reaches(b, c):
            for u, v in ((a, d), (c, b)):
                succ[u].discard(v); present.discard((u, v))
            for u, v in ((a, b), (c, d)):
                succ.setdefault(u, set()).add(v); present.add((u, v))
            continue
        edges[i], edges[j] = (a, d), (c, b)
        swaps += 1

    out = dag.copy()
    out["prereq_skill_id"] = [e[0] for e in edges]
    out["skill_id"] = [e[1] for e in edges]
    out["derivation"] = "degree_preserving_random"
    rewired = sum(1 for e, o in zip(edges, zip(dag["prereq_skill_id"], dag["skill_id"]))
                  if e != o)
    print(f"[ablation]   {swaps} accepted swaps, {rewired}/{len(edges)} edges rewired")
    return out


# ------------------------------------------------------------- rebuild world
def rebuild(root: Path, condition: str, seed: int = 7) -> Dict[str, float]:
    """Re-derive plans and learner goals in `root` from a modified graph."""
    env = dict(os.environ, THESIS_DATA_ROOT=str(root))
    code = f'''
import sys; sys.path.insert(0, {str(Path.cwd())!r})
import numpy as np, pandas as pd
from pipeline import paths, stage09_prereqs as s9, stage12_simulator as s12
from pipeline.stage16_ablation import degree_preserving_shuffle

rng = np.random.default_rng({seed})
skills = pd.read_parquet(paths.CATALOG / "skills.parquet")
skills = skills[skills["tier"] == "skill"] if "tier" in skills else skills
dag = pd.read_parquet(paths.WORLD / "skill_prereqs.parquet")
real_dag = dag.copy()

cond = {condition!r}
if cond == "random":
    dag = degree_preserving_shuffle(dag, rng)
elif cond == "none":
    dag = dag.iloc[0:0]
dag.to_parquet(paths.WORLD / "skill_prereqs.parquet", index=False)

# With no edges every plan is one skill deep, so the depth>=3 filter would
# empty the world. Relax it for that condition only: a goal with no chain
# is exactly what the control is meant to represent.
if cond == "none":
    s9.MIN_PLAN_DEPTH = 1
plans, items = s9._study_plans(skills, dag)
plans.to_parquet(paths.WORLD / "study_plans.parquet", index=False)
items.to_parquet(paths.WORLD / "study_plan_items.parquet", index=False)

label_of = dict(zip(skills["skill_id"], skills["label"]))
fam_of = dict(zip(skills["skill_id"], skills["family"]))
by_plan = {{p: g.sort_values("position") for p, g in items.groupby("study_plan_id")}}
usable = sorted(by_plan)
plan_family = dict(zip(plans["study_plan_id"], plans["domain_family"]))
depth_of = {{p: int(g["position"].max()) for p, g in by_plan.items()}}

# Depth matching. Degree-preserving swaps keep each skill's in/out degree
# but lengthen paths, so the random arm would otherwise differ from real in
# plan DEPTH as well as in ordering, and a difference could not be
# attributed to either. Give every learner the plan closest in depth to the
# one the real graph would have handed them.
real_plans, real_items = s9._study_plans(skills, real_dag)
real_usable = sorted(real_items["study_plan_id"].unique())
real_depth = {{p: int(g["position"].max())
               for p, g in real_items.groupby("study_plan_id")}}
used = {{p: 0 for p in usable}}

def pick(i):
    if cond == "real":
        return usable[i % len(usable)]
    want = real_depth[real_usable[i % len(real_usable)]]
    # nearest depth, then least-used, so the assignment still spreads
    return min(usable, key=lambda p: (abs(depth_of[p] - want), used[p], p))

users = pd.read_parquet(paths.WORLD / "users.parquet")
rows, latent = [], []
for i, u in enumerate(users.itertuples()):
    pid = pick(i)
    used[pid] += 1
    plan = by_plan[pid]
    target = int(plan["position"].max())
    goal_ids = plan[plan["is_goal"]]["skill_id"].tolist() or \\
        plan[plan["position"] == target]["skill_id"].tolist()
    m = s12._mastery_for(int(u.start_level), plan, target, float(u.readiness), rng)
    for sid, v in m.items():
        if v > 0.01:
            latent.append({{"user_id": u.user_id, "skill_id": sid,
                           "family": fam_of.get(sid), "mastery": v}})
    d = u._asdict(); d.pop("Index", None)
    d.update({{"goal_plan_id": pid, "goal_family": plan_family.get(pid),
              "goal_position": target, "goal_skill_ids": goal_ids,
              "goal_labels": [label_of.get(s) for s in goal_ids]}})
    rows.append(d)

pd.DataFrame(rows).to_parquet(paths.WORLD / "users.parquet", index=False)
pd.DataFrame(latent).to_parquet(paths.WORLD / "user_latent_state.parquet", index=False)
assigned = [depth_of[r["goal_plan_id"]] for r in rows]
print(f"[ablation]   plans {{len(plans)}}  edges {{len(dag)}}  "
      f"pool mean depth {{plans['n_milestones'].mean() if len(plans) else 0:.1f}}  "
      f"| ASSIGNED mean depth {{np.mean(assigned):.1f}}  "
      f"mastery rows {{len(latent)}}")
'''
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"[ablation] rebuild failed for {condition}")
    return {}


def run_experiment(root: Path, condition: str, prereg: str, n_users: int,
                   seeds: List[int]) -> pd.DataFrame:
    env = dict(os.environ, THESIS_DATA_ROOT=str(root))
    code = f'''
import sys; sys.path.insert(0, {str(Path.cwd())!r})
from pipeline import freeze_bundle, stage16_experiment
freeze_bundle.run({prereg!r}, force=True)
stage16_experiment.run(n_users={n_users}, seeds={seeds!r}, prereg={prereg!r})
'''
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
        raise SystemExit(f"[ablation] experiment failed for {condition}")
    return pd.read_parquet(root / "data" / "agent" / f"experiment_{prereg}.parquet")


def run(n_users: int = 500, seeds: List[int] = None,
        prereg: str = "poc-v6-stratified", seed: int = 7) -> str:
    seeds = seeds or [7, 11, 23, 37, 53]
    from pipeline import paths as live_paths

    results = {}
    scratch = Path(tempfile.mkdtemp(prefix="ablation-"))
    try:
        for cond in CONDITIONS:
            root = scratch / cond
            (root / "data").mkdir(parents=True)
            (root / "reports").mkdir(parents=True, exist_ok=True)
            for sub in ("catalog", "world", "agent"):
                shutil.copytree(live_paths.DATA / sub, root / "data" / sub)
            print(f"\n[ablation] === {cond} ===")
            rebuild(root, cond, seed)
            results[cond] = run_experiment(root, cond, prereg, n_users, seeds)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    def totals(df: pd.DataFrame) -> pd.DataFrame:
        """Total reward per (policy, learner, seed), then averaged over seeds."""
        g = df.groupby(["policy", "user_id", "seed"])["reward"].sum().reset_index()
        return g.groupby(["policy", "user_id"])["reward"].mean().unstack(0)

    tot = {c: totals(results[c]) for c in CONDITIONS}

    out = [
        "# Randomized-graph ablation", "",
        "The go/no-go gate from the ingestion plan, run at last. Three "
        "populations identical in learner id, start level, weekly hours, "
        "persistence and budget; only the graph their goals and starting "
        "mastery derive from differs.", "",
        f"`python -m pipeline.stage16_ablation` · bundle `{prereg}` weights · "
        f"{n_users} learners × {len(seeds)} seeds · no API calls", "",
        "Degree-preserving swaps keep every skill's in- and out-degree but "
        "lengthen paths, so each learner is also matched to the random plan "
        "closest in depth to the real plan they would have received. Without "
        "that the random arm would differ in plan depth as well as ordering "
        "and neither could be blamed for a difference.", "",
        "## Mean total reward", "",
        "| graph | baseline | linucb | planner |",
        "|---|---|---|---|",
    ]
    for cond in CONDITIONS:
        m = tot[cond].mean()
        out.append(f"| `{cond}` | {m.get('baseline', float('nan')):+.4f} | "
                   f"{m.get('linucb', float('nan')):+.4f} | "
                   f"{m.get('planner', float('nan')):+.4f} |")

    out += [
        "", "## Paired contrasts — real graph vs each control", "",
        "Every learner appears under all three graphs, so the within-learner "
        "difference removes variance from learners differing in level, hours "
        "and budget. A 95% interval spanning zero means the graph made no "
        "established difference to that policy.", "",
        "| policy | contrast | difference | 95% CI | win rate | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for pol in ("baseline", "linucb", "planner"):
        for other in ("random", "none"):
            a, b = tot["real"].get(pol), tot[other].get(pol)
            if a is None or b is None:
                continue
            d = (a - b).dropna()
            if d.empty:
                continue
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            est = "**established**" if abs(d.mean()) > ci else "not established"
            out.append(f"| `{pol}` | real − {other} | {d.mean():+.4f} | "
                       f"±{ci:.4f} | {(d > 0).mean():.0%} | {est} |")

    out += [
        "",
        "## Reading", "",
        "A real-vs-random difference that is not established means the "
        "prerequisite structure adds no signal to the simulation beyond its "
        "shape — which is a result worth reporting, not a failure to hide. "
        "The `none` column is the weaker control: it removes the chain "
        "entirely, so learners start with almost no seeded mastery and the "
        "comparison confounds ordering with preparation.", "",
        "A gap does **not** establish that the main comparison is "
        "uncontaminated. Only the `world/` → `agent/` firewall does that, and "
        "it is an implementation invariant rather than a measurement.", "",
    ]
    text = "\n".join(out)
    (live_paths.REPORTS / "ablation_randomized_graph.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Randomized-graph ablation")
    ap.add_argument("--users", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 11, 23, 37, 53])
    ap.add_argument("--prereg", default="poc-v6-stratified")
    a = ap.parse_args()
    print(run(a.users, a.seeds, a.prereg))


if __name__ == "__main__":
    main()
