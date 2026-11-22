"""Stage 12 tests: the synthetic learner population."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import paths


@pytest.mark.skipif(not (paths.WORLD / "users.parquet").exists(),
                    reason="requires a built population")
class TestPopulation:
    @pytest.fixture(scope="class")
    def users(self):
        return pd.read_parquet(paths.WORLD / "users.parquet")

    @pytest.fixture(scope="class")
    def latent(self):
        return pd.read_parquet(paths.WORLD / "user_latent_state.parquet")

    def test_levels_are_stratified_not_catalog_matched(self, users):
        """The catalog is 49% beginner. A matching population would leave
        difficulty inert for most users, since a beginner's top candidates
        are already beginner-level and the penalty is exactly zero."""
        share = users["start_level"].value_counts(normalize=True)
        assert share.min() > 0.15, "a level stratum is too thin to measure"
        assert share.max() < 0.55, "population is skewed toward one level"

    def test_every_user_has_a_reachable_goal(self, users):
        assert users["goal_skill_ids"].map(len).gt(0).all()
        assert users["goal_position"].ge(1).all()

    def test_goals_span_all_families(self, users):
        """Regression: plans were sampled uniformly, so a thin family could
        vanish. nlp-and-llm holds 4 of 122 plans and one run drew none —
        the thinnest and most AI-relevant family, silently absent. Family
        is now round-robined alongside level."""
        skills = pd.read_parquet(paths.CATALOG / "skills.parquet")
        plans = pd.read_parquet(paths.WORLD / "study_plans.parquet")
        reachable = set(plans["domain_family"].dropna())
        assert set(users["goal_family"].dropna()) >= reachable

    def test_mastery_is_bounded(self, latent):
        assert latent["mastery"].between(0, 1).all()

    def test_mastery_is_noisy_not_a_clean_frontier(self, latent):
        """A learner who has mastered every prerequisite and nothing else is
        a fiction; a policy tuned against it is tuned against a fiction."""
        assert latent["mastery"].std() > 0.15

    def test_most_users_have_a_real_prerequisite_gap(self, users):
        gap = users["prereq_gap"].dropna()
        assert (gap > 0).mean() > 0.65, "sequencing is not a live problem"

    def test_readiness_controls_the_gap(self, users):
        """Regression: mastery was hard-coded generous, leaving 39% of users
        with every prerequisite already satisfied. Readiness makes the gap a
        designed property, so it must actually drive it."""
        d = users.dropna(subset=["prereq_gap"])
        lo = d[d["readiness"] <= d["readiness"].median()]
        hi = d[d["readiness"] > d["readiness"].median()]
        assert (lo["prereq_gap"] > 0).mean() > (hi["prereq_gap"] > 0).mean()

    def test_both_prepared_and_unprepared_learners_exist(self, users):
        """A population that is all one or the other cannot show whether
        sequencing helps."""
        gap = users["prereq_gap"].dropna()
        assert (gap > 0).mean() < 0.95
        assert (gap == 0).mean() > 0.02

    def test_weekly_hours_vary(self, users):
        assert users["weekly_hours"].nunique() >= 4

    def test_population_is_reproducible(self, tmp_path):
        """Runs in a scratch data root.

        An earlier version called run() directly, which WRITES to
        world/users.parquet -- so simply running the test suite replaced the
        real 500-learner population with its own 40-user fixture. A test
        must never have a side effect on live data.
        """
        import os
        import subprocess
        import sys
        repo = paths.ROOT
        env = dict(os.environ, THESIS_DATA_ROOT=str(tmp_path), PYTHONPATH=str(repo))
        # Seed the scratch root with the artefacts the simulator reads.
        for rel in ("data/catalog/skills.parquet", "data/world/study_plan_items.parquet",
                    "data/world/study_plans.parquet", "data/world/skill_prereqs.parquet"):
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((repo / rel).read_bytes())

        def _run():
            subprocess.run([sys.executable, "-m", "pipeline.stage12_simulator",
                            "--n", "30", "--seed", "99", "--mode", "statistical"],
                           cwd=repo, env=env, capture_output=True, check=True)
            return pd.read_parquet(tmp_path / "data/world/users.parquet")

        a, b = _run(), _run()
        assert a["goal_plan_id"].tolist() == b["goal_plan_id"].tolist()
        assert a["start_level"].tolist() == b["start_level"].tolist()
        # And the live population is untouched.
        live = pd.read_parquet(paths.WORLD / "users.parquet")
        assert len(live) > 100, "a test clobbered the live population"


class TestFirewall:
    def test_population_lives_in_world_not_agent(self):
        """It is the oracle the reward is computed against, so agent/ must
        never be able to read it."""
        assert (paths.WORLD / "users.parquet").exists()
        assert not (paths.AGENT / "users.parquet").exists()
        assert not (paths.AGENT / "user_latent_state.parquet").exists()


class TestPreregistration:
    """The bundle is what makes the bandit comparison interpretable."""

    @staticmethod
    def _bundles():
        import yaml
        d = paths.WORLD / "preregistrations"
        return [yaml.safe_load(p.read_text()) for p in sorted(d.glob("*.yaml"))]

    def test_bundles_exist_and_are_hashed(self):
        bundles = self._bundles()
        assert bundles, "no pre-registration: the comparison is uninterpretable"
        for b in bundles:
            assert len(b["bundle_sha256"]) == 64
            assert b["declared_before_any_bandit_run"] is True

    def test_each_bundle_is_a_distinct_configuration(self):
        """Separate hypotheses, all reported — not tweaks of one another."""
        hashes = [b["bundle_sha256"] for b in self._bundles()]
        assert len(hashes) == len(set(hashes))

    def test_reward_asymmetry_is_frozen_in(self):
        for b in self._bundles():
            spec = b["reward_spec"]
            assert spec["w_too_hard"] > spec["w_too_easy"], (
                f"{b['name']}: a symmetric difficulty penalty makes the "
                "optimal policy level-neutral and erases the sequencing "
                "behaviour under study")

    def test_taxonomy_and_dag_are_hashed_with_the_reward(self):
        """The reward is computed OVER these, so freezing it alone is not
        enough: changing them changes what the reward means while leaving
        the reward's own hash untouched."""
        for b in self._bundles():
            for needed in ("skills", "skill_prereqs", "users", "arm_features"):
                assert needed in b["dependencies"], b["name"]
                assert len(b["dependencies"][needed]["sha256"]) == 64

    def test_latest_bundle_matches_what_is_on_disk(self):
        """If an artefact changed after freezing, that bundle no longer
        describes the experiment and results under it must be re-run."""
        from pipeline import hashing
        bundles = sorted(self._bundles(), key=lambda b: b["frozen_at"])
        latest = bundles[-1]
        drifted = [label for label, dep in latest["dependencies"].items()
                   if (paths.ROOT / dep["path"]).exists()
                   and hashing.sha256_file(paths.ROOT / dep["path"]) != dep["sha256"]]
        assert not drifted, (f"{latest['name']}: artefacts changed since "
                             f"pre-registration: {drifted}")

    def test_every_weight_has_a_recorded_rationale(self):
        for b in self._bundles():
            for w in ("w_topic", "w_too_hard", "w_too_easy", "w_time",
                      "w_quality", "w_unknown_penalty"):
                assert b["rationale"].get(w), f"{b['name']}: {w} has no reason"
