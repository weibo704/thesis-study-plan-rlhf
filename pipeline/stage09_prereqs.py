"""Stage 9 -- prerequisite DAG and study_plans.

Two independent edge sources, per docs/ingestion-plan.md section 2:

  agent_proposed        an LLM reads the frozen leaf labels and says which
                        skills presuppose which. Asked per family (a
                        tractable partial order over 7-14 labels) plus one
                        cross-family pass -- never as 187x186 pairs.
  co_occurrence_inferred a skill that consistently appears on courses at a
                        higher difficulty than skills it co-occurs with is a
                        candidate downstream skill.

Agreement between the two is reported: where they disagree is where the DAG
is weakest, and that belongs in the write-up.

FIREWALL: everything here is written to data/world/ and must never be read
by anything that builds agent/ features. Co-occurrence edges are derived
from level_canonical and skill overlap -- the same signals the recommender
scores on -- so if these edges reached arm features while the reward
credited prerequisite satisfaction, the bandit would win by construction.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from . import hashing, llm, paths

PROMPT_VERSION = "prereq-v1"

WITHIN_SYSTEM = (
    "You order skills within one topic area by what a learner must know first.\n\n"
    "Given a list of skill labels from a single family, return the prerequisite "
    "edges among them.\n\n"
    "Rules:\n"
    "1. An edge prereq -> skill means a learner should normally learn 'prereq' "
    "before 'skill'.\n"
    "2. Only use labels from the supplied list, copied exactly.\n"
    "3. Emit an edge only where the dependency is real and widely agreed. "
    "Prefer few confident edges over many speculative ones.\n"
    "4. Never emit both directions, and never emit a self-edge.\n"
    "5. Do not chain everything into one long line; parallel skills are normal.\n"
    'Return JSON: {"edges": [{"prereq": "...", "skill": "...", '
    '"confidence": 0.0-1.0, "reason": "max 12 words"}]}'
)

CROSS_SYSTEM = (
    "You identify prerequisite relationships BETWEEN topic areas in a "
    "computer-science and data-science curriculum.\n\n"
    "You receive skill labels grouped by family. Return only edges where the "
    "prerequisite is in a DIFFERENT family from the dependent skill.\n\n"
    "Rules:\n"
    "1. Only use labels exactly as supplied.\n"
    "2. prereq and skill must come from different families.\n"
    "3. Emit only widely agreed dependencies (e.g. foundational mathematics "
    "before applied modelling). Prefer few confident edges.\n"
    "4. No self-edges, no both-directions pairs.\n"
    'Return JSON: {"edges": [{"prereq": "...", "skill": "...", '
    '"confidence": 0.0-1.0, "reason": "max 12 words"}]}'
)


def _cfg() -> Dict[str, Any]:
    with open(paths.CONFIG / "thresholds.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("prereqs", {})


# ------------------------------------------------------------ agent edges
def _agent_edges(skills: pd.DataFrame, budget: llm.Budget) -> List[Dict[str, Any]]:
    model = llm.config()["models"]["adjudicate"]
    label_to_id = dict(zip(skills["label"], skills["skill_id"]))
    edges: List[Dict[str, Any]] = []

    for family, grp in skills.groupby("family"):
        labels = sorted(grp["label"].tolist())
        if len(labels) < 2:
            continue
        out = llm.chat_json(
            [{"role": "system", "content": WITHIN_SYSTEM},
             {"role": "user", "content": json.dumps(
                 {"family": family, "skills": labels}, ensure_ascii=False)}],
            model=model, budget=budget)
        for e in (out or {}).get("edges", []) or []:
            a, b = str(e.get("prereq", "")).strip(), str(e.get("skill", "")).strip()
            if a in label_to_id and b in label_to_id and a != b:
                edges.append({
                    "prereq_skill_id": label_to_id[a], "skill_id": label_to_id[b],
                    "prereq_label": a, "skill_label": b,
                    "confidence": float(e.get("confidence", 0.5) or 0.5),
                    "derivation": "agent_proposed", "scope": "within_family",
                    "support_evidence": str(e.get("reason", ""))[:120],
                })
        print(f"[prereqs] {family:<26} {len(labels):>3} skills -> "
              f"{sum(1 for x in edges if x['scope']=='within_family' and x['prereq_label'] in labels)} edges")

    grouped = {fam: sorted(g["label"].tolist())
               for fam, g in skills.groupby("family")}
    out = llm.chat_json(
        [{"role": "system", "content": CROSS_SYSTEM},
         {"role": "user", "content": json.dumps({"families": grouped},
                                                ensure_ascii=False)}],
        model=model, budget=budget)
    fam_of = dict(zip(skills["label"], skills["family"]))
    n_cross = 0
    for e in (out or {}).get("edges", []) or []:
        a, b = str(e.get("prereq", "")).strip(), str(e.get("skill", "")).strip()
        if a in label_to_id and b in label_to_id and a != b and fam_of[a] != fam_of[b]:
            edges.append({
                "prereq_skill_id": label_to_id[a], "skill_id": label_to_id[b],
                "prereq_label": a, "skill_label": b,
                "confidence": float(e.get("confidence", 0.5) or 0.5),
                "derivation": "agent_proposed", "scope": "cross_family",
                "support_evidence": str(e.get("reason", ""))[:120],
            })
            n_cross += 1
    print(f"[prereqs] cross-family -> {n_cross} edges")
    return edges


# ----------------------------------------------------- co-occurrence edges
def _cooccurrence_edges(skills: pd.DataFrame, course_skills: pd.DataFrame,
                        courses: pd.DataFrame, cfg) -> List[Dict[str, Any]]:
    """A skill sitting on systematically harder courses is downstream."""
    levels = courses.set_index("course_id")["level_canonical"]
    cs = course_skills.merge(levels, left_on="course_id", right_index=True, how="left")
    cs = cs[cs["level_canonical"].notna()]

    mean_level = cs.groupby("skill_id")["level_canonical"].mean()
    by_course = cs.groupby("course_id")["skill_id"].apply(list)

    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for ids in by_course:
        uniq = sorted(set(ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_counts[(uniq[i], uniq[j])] += 1

    label_of = dict(zip(skills["skill_id"], skills["label"]))
    edges = []
    for (a, b), n in pair_counts.items():
        if n < cfg["min_cooccurrence"]:
            continue
        la, lb = mean_level.get(a), mean_level.get(b)
        if la is None or lb is None or np.isnan(la) or np.isnan(lb):
            continue
        delta = float(lb - la)
        if abs(delta) < cfg["min_level_delta"]:
            continue
        lo, hi = (a, b) if delta > 0 else (b, a)
        edges.append({
            "prereq_skill_id": lo, "skill_id": hi,
            "prereq_label": label_of.get(lo), "skill_label": label_of.get(hi),
            # Confidence scales with the difficulty gap and co-occurrence count.
            "confidence": round(min(0.6, 0.2 + abs(delta) * 0.5 + min(n, 20) * 0.01), 3),
            "derivation": "co_occurrence_inferred", "scope": "statistical",
            "support_evidence": f"co-occur {n}, level delta {abs(delta):.2f}",
        })
    return edges


# ------------------------------------------------------------- acyclicity
def _make_acyclic(edges: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Greedy highest-confidence-first insertion.

    Adding edges in descending confidence and skipping any that would close
    a cycle keeps the most trusted edges and is deterministic, which a
    'detect cycles then delete' pass is not.
    """
    ordered = edges.sort_values(
        ["confidence", "prereq_skill_id", "skill_id"], ascending=[False, True, True])
    adj: Dict[str, Set[str]] = defaultdict(set)

    def reaches(src: str, dst: str) -> bool:
        stack, seen = [src], set()
        while stack:
            node = stack.pop()
            if node == dst:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj[node])
        return False

    keep_idx, dropped = [], 0
    for idx, row in ordered.iterrows():
        a, b = row["prereq_skill_id"], row["skill_id"]
        if a == b or reaches(b, a):      # would close a cycle
            dropped += 1
            continue
        adj[a].add(b)
        keep_idx.append(idx)
    return ordered.loc[keep_idx], dropped


# --------------------------------------------------------------- study_plans
MIN_PLAN_DEPTH = 3


def _study_plans(skills: pd.DataFrame, dag: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One plan per goal skill: the topological layering of its ANCESTORS.

    Previously one plan per family, using only within-family edges. That
    discarded 35% of the DAG -- every cross-family edge -- and with it the
    sequencing that matters most: Python before machine learning, linear
    algebra before neural networks. Per-family depth came out 3-7 while the
    same DAG supports mean depth 6.1 and up to 18 globally.

    A learner's goal sits in one family, but the chain to reach it does not.
    """
    label_of = dict(zip(skills["skill_id"], skills["label"]))
    fam_of = dict(zip(skills["skill_id"], skills["family"]))

    preds: Dict[str, Set[str]] = {s: set() for s in skills["skill_id"]}
    for r in dag.itertuples():
        preds.setdefault(r.skill_id, set()).add(r.prereq_skill_id)

    def ancestors(goal: str) -> Set[str]:
        seen, stack = set(), [goal]
        while stack:
            n = stack.pop()
            for pr in preds.get(n, ()):
                if pr not in seen:
                    seen.add(pr)
                    stack.append(pr)
        return seen

    plan_rows, item_rows = [], []
    for goal in skills["skill_id"]:
        nodes = ancestors(goal) | {goal}
        depth: Dict[str, int] = {}
        remaining, level = set(nodes), 0
        while remaining:
            ready = {s for s in remaining if not (preds.get(s, set()) & remaining)}
            if not ready:
                ready = set(remaining)
            for s in ready:
                depth[s] = level
            remaining -= ready
            level += 1
        if level < MIN_PLAN_DEPTH:
            continue   # no sequencing to plan over

        pid = "PLAN_" + label_of[goal].upper().replace(" ", "_")[:40]
        pid = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in pid)
        plan_rows.append({
            "study_plan_id": pid, "name": f"reach {label_of[goal]}",
            "goal_skill_id": goal, "goal_label": label_of[goal],
            "domain_family": fam_of.get(goal), "n_milestones": level,
            "n_skills": len(nodes),
            "n_families": len({fam_of.get(s) for s in nodes}),
        })
        for s in sorted(nodes, key=lambda x: (depth[x], label_of[x])):
            tied = sum(1 for o in nodes if depth[o] == depth[s])
            item_rows.append({
                "study_plan_id": pid, "position": depth[s], "skill_id": s,
                "label": label_of[s], "family": fam_of.get(s),
                "is_goal": s == goal, "is_optional": False,
                # Parallel skills at one depth are genuinely unordered;
                # breaking the tie would look like a real ordering claim.
                "tied_at_position": tied > 1, "n_tied": tied,
            })
    return pd.DataFrame(plan_rows), pd.DataFrame(item_rows)


def run() -> pd.DataFrame:
    cfg = _cfg()
    budget = llm.Budget(llm.config()["limits"]["max_adjudicate_calls"], "prereqs")

    skills = pd.read_parquet(paths.CATALOG / "skills.parquet")
    course_skills = pd.read_parquet(paths.CATALOG / "course_skills_final.parquet")
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")

    agent = _agent_edges(skills, budget)
    cooc = _cooccurrence_edges(skills, course_skills, courses, cfg)
    print(f"[prereqs] agent={len(agent)} co-occurrence={len(cooc)} raw edges")

    raw = pd.DataFrame(agent + cooc)
    raw["model_version"] = llm.config()["models"]["adjudicate"]
    raw["prompt_version"] = PROMPT_VERSION

    # Agreement between the two independent sources.
    a_set = {(e["prereq_skill_id"], e["skill_id"]) for e in agent}
    c_set = {(e["prereq_skill_id"], e["skill_id"]) for e in cooc}
    c_rev = {(b, a) for a, b in c_set}
    agree = len(a_set & c_set)
    contradict = len(a_set & c_rev)

    # Collapse duplicates, keeping the most confident and noting corroboration.
    raw["pair"] = list(zip(raw["prereq_skill_id"], raw["skill_id"]))
    raw["corroborated"] = raw["pair"].isin(a_set & c_set)
    dedup = (raw.sort_values("confidence", ascending=False)
                .drop_duplicates(subset=["pair"], keep="first"))
    dedup.loc[dedup["corroborated"], "confidence"] = (
        dedup.loc[dedup["corroborated"], "confidence"].clip(lower=0.8))

    dag, dropped = _make_acyclic(dedup.drop(columns=["pair"]))
    study_plans, items = _study_plans(skills, dag)

    hashing.write_parquet(dag.reset_index(drop=True),
                          paths.WORLD / "skill_prereqs.parquet", stage="prereqs")
    hashing.write_parquet(study_plans, paths.WORLD / "study_plans.parquet", stage="prereqs")
    hashing.write_parquet(items, paths.WORLD / "study_plan_items.parquet", stage="prereqs")

    by_deriv = dict(dag["derivation"].value_counts())
    lines = [
        "# Prerequisite DAG and study_plans", "",
        "> **Firewall.** Everything here lives in `data/world/` and must never",
        "> be read by anything building `agent/` features. Co-occurrence edges",
        "> are derived from `level_canonical` and skill overlap -- the same",
        "> signals the recommender scores on -- so if they reached arm features",
        "> while the reward credited prerequisite satisfaction, the bandit",
        "> would win by construction.", "",
        f"- skills: **{len(skills)}**",
        f"- raw edges proposed: **{len(raw)}** "
        f"(agent {len(agent)}, co-occurrence {len(cooc)})",
        f"- after dedup: **{len(dedup)}**",
        f"- dropped to enforce acyclicity: **{dropped}**",
        f"- final DAG edges: **{len(dag)}**", "",
        "## Edge provenance", "", "| derivation | edges |", "|---|---|",
    ]
    for k, v in by_deriv.items():
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "", "## Agreement between the two sources", "",
        f"- both propose the same direction: **{agree}** edges",
        f"- they propose OPPOSITE directions: **{contradict}** edges", "",
        "Where the two disagree is where the DAG is weakest. Corroborated",
        "edges are floored at confidence 0.8; contradictions are resolved by",
        "confidence during the acyclic pass and are worth a sentence in the",
        "write-up rather than silent resolution.", "",
        "## Study Plans", "", "| family | milestones | skills |", "|---|---|---|",
    ]
    for r in study_plans.sort_values("domain_family").itertuples():
        lines.append(f"| `{r.domain_family}` | {r.n_milestones} | {r.n_skills} |")
    tied = int(items["tied_at_position"].sum())
    lines += [
        "", f"{tied} of {len(items)} study plan items sit at a position shared",
        "with at least one other skill. Those ties are recorded rather than",
        "broken: parallel skills at one depth are genuinely unordered, and an",
        "arbitrary order would read downstream as a real sequencing claim.",
        "", "## Validation status", "",
        "Held-out labelled edges and inter-annotator agreement are **not**",
        "done -- the plan drops them, since the DAG is scaffolding rather",
        "than a contribution.",
        "",
        "The **randomized-graph ablation** -- the plan's go/no-go gate -- HAS",
        "now run: `python -m pipeline.stage16_ablation`, written up in",
        "`reports/ablation_randomized_graph.md`. Against a degree-preserving",
        "randomized graph, depth-matched per learner, the real edges are worth",
        "+0.055 to both linucb and the planner (95% CI +/-0.045, established)",
        "and +0.035 to the fixed-weight baseline (not established). The DAG",
        "beats noise for the policies that learn.",
        "",
        "That is all it means. A pass does NOT make the main comparison",
        "uncontaminated -- only the world/ -> agent/ firewall does that, and",
        "it is an implementation invariant rather than a measurement.",
    ]
    (paths.REPORTS / "prereqs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("prereqs", edges=len(dag), dropped_for_cycles=dropped,
                         agree=agree, contradict=contradict)
    print(f"[prereqs] DAG: {len(dag)} edges ({by_deriv}); dropped {dropped} for cycles")
    print(f"[prereqs] agreement: {agree} same-direction, {contradict} opposite")
    print(f"[prereqs] study_plans: {len(study_plans)} families, {len(items)} milestone items")
    return dag


def main() -> None:
    argparse.ArgumentParser(description="Stage 9: prerequisite DAG").parse_args()
    run()


if __name__ == "__main__":
    main()
