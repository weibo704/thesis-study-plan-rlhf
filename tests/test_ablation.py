"""Randomized-graph ablation: the shuffle must be a fair control.

The whole gate rests on the randomized graph being the real graph's shape
with its ordering destroyed. If the shuffle silently changed the degree
sequence, or emitted a cycle, or barely rewired anything, the ablation
would be comparing the wrong things and would still look like it ran.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import paths
from pipeline.stage16_ablation import degree_preserving_shuffle


def _degrees(dag: pd.DataFrame):
    out = dag.groupby("prereq_skill_id").size().to_dict()
    inn = dag.groupby("skill_id").size().to_dict()
    return out, inn


def _is_acyclic(dag: pd.DataFrame) -> bool:
    succ = {}
    for r in dag.itertuples():
        succ.setdefault(r.prereq_skill_id, set()).add(r.skill_id)
    colour = {}

    def visit(n):
        state = colour.get(n, 0)
        if state == 1:
            return False
        if state == 2:
            return True
        colour[n] = 1
        for m in succ.get(n, ()):
            if not visit(m):
                return False
        colour[n] = 2
        return True

    return all(visit(n) for n in list(succ))


def _synthetic(n_skills: int = 40, n_edges: int = 90,
               seed: int = 3) -> pd.DataFrame:
    """A layered DAG, so edges always point from lower to higher index."""
    rng = np.random.default_rng(seed)
    ids = [f"S{i:03d}" for i in range(n_skills)]
    seen, rows = set(), []
    while len(rows) < n_edges:
        a, b = sorted(rng.choice(n_skills, 2, replace=False))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append({"prereq_skill_id": ids[a], "skill_id": ids[b],
                     "confidence": 0.7, "derivation": "agent_proposed"})
    return pd.DataFrame(rows)


class TestShuffle:
    @pytest.fixture(scope="class")
    def pair(self):
        dag = _synthetic()
        return dag, degree_preserving_shuffle(dag, np.random.default_rng(7))

    def test_edge_count_preserved(self, pair):
        dag, shuf = pair
        assert len(shuf) == len(dag)

    def test_degree_sequence_preserved(self, pair):
        dag, shuf = pair
        assert _degrees(dag) == _degrees(shuf), \
            "a degree-preserving swap must leave every in/out degree untouched"

    def test_result_is_acyclic(self, pair):
        _, shuf = pair
        assert _is_acyclic(shuf), "the control graph must still be a DAG"

    def test_no_self_loops_or_duplicates(self, pair):
        _, shuf = pair
        assert not (shuf["prereq_skill_id"] == shuf["skill_id"]).any()
        edges = list(zip(shuf["prereq_skill_id"], shuf["skill_id"]))
        assert len(set(edges)) == len(edges)

    def test_actually_rewires(self, pair):
        dag, shuf = pair
        same = sum(a == b for a, b in
                   zip(zip(dag["prereq_skill_id"], dag["skill_id"]),
                       zip(shuf["prereq_skill_id"], shuf["skill_id"])))
        assert same / len(dag) < 0.25, \
            "a shuffle that leaves most edges in place is not a control"

    def test_deterministic_under_seed(self):
        dag = _synthetic()
        a = degree_preserving_shuffle(dag, np.random.default_rng(11))
        b = degree_preserving_shuffle(dag, np.random.default_rng(11))
        assert a.equals(b)

    def test_marks_its_provenance(self, pair):
        _, shuf = pair
        assert (shuf["derivation"] == "degree_preserving_random").all(), \
            "randomized edges must never be mistaken for derived ones"


@pytest.mark.skipif(not (paths.WORLD / "skill_prereqs.parquet").exists(),
                    reason="requires a built DAG")
class TestOnRealGraph:
    def test_real_dag_shuffles_cleanly(self):
        dag = pd.read_parquet(paths.WORLD / "skill_prereqs.parquet")
        shuf = degree_preserving_shuffle(dag, np.random.default_rng(7))
        assert _degrees(dag) == _degrees(shuf)
        assert _is_acyclic(shuf)
        assert len(shuf) == len(dag)
