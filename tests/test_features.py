"""Stage 10 tests: arm identity stability and the namespace firewall."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import paths


class TestFirewall:
    def test_catches_a_real_world_read(self):
        """The check must FAIL when world/ is actually read.

        An earlier version grepped this module's own source for
        'paths.WORLD' and tripped over the assertion line implementing it —
        a grep cannot distinguish a reference from a read.
        """
        paths.reset_read_log()
        paths.load_parquet(paths.WORLD / "skill_prereqs.parquet")
        with pytest.raises(AssertionError, match="FIREWALL VIOLATION"):
            paths.assert_no_world_reads("deliberate_violation")

    def test_passes_on_catalog_only_reads(self):
        paths.reset_read_log()
        paths.load_parquet(paths.CATALOG / "courses.parquet")
        paths.assert_no_world_reads("catalog_only")  # must not raise

    def test_mentioning_world_in_source_is_not_a_violation(self):
        paths.reset_read_log()
        _ = str(paths.WORLD)          # referencing the path is fine
        paths.assert_no_world_reads("reference_only")


@pytest.mark.skipif(not (paths.AGENT / "arm_features.parquet").exists(),
                    reason="requires built features")
class TestArmIndex:
    @pytest.fixture(scope="class")
    def index(self):
        return pd.read_parquet(paths.AGENT / "arm_index.parquet")

    @pytest.fixture(scope="class")
    def features(self):
        return pd.read_parquet(paths.AGENT / "arm_features.parquet")

    def test_index_is_a_bijection(self, index):
        assert index["arm_index"].is_unique
        assert index["dedup_cluster_id"].is_unique

    def test_index_is_keyed_on_cluster_not_course(self, index):
        # Keyed on a source row, re-running dedup would silently re-point
        # trained per-arm parameters at a different course.
        assert "dedup_cluster_id" in index.columns
        assert "course_id" not in index.columns

    def test_index_is_contiguous_from_zero(self, index):
        active = index[index["arm_status"] == "active"]
        assert active["arm_index"].min() == 0
        assert len(set(active["arm_index"])) == len(active)

    def test_arm_status_is_tracked(self, index):
        assert set(index["arm_status"]) <= {"active", "retired", "out_of_scope"}

    def test_rebuild_is_stable(self):
        """Re-running must not renumber existing arms."""
        before = pd.read_parquet(paths.AGENT / "arm_index.parquet")
        from pipeline.stage10_features import build_arm_index
        courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
        after = build_arm_index(courses)
        merged = before.merge(after, on="dedup_cluster_id", suffixes=("_a", "_b"))
        assert (merged["arm_index_a"] == merged["arm_index_b"]).all()


@pytest.mark.skipif(not (paths.AGENT / "arm_features.parquet").exists(),
                    reason="requires built features")
class TestFeatureMatrix:
    @pytest.fixture(scope="class")
    def features(self):
        return pd.read_parquet(paths.AGENT / "arm_features.parquet")

    def test_no_nan_anywhere(self, features):
        assert not features.isna().any().any()

    def test_carries_arm_index_as_a_column(self, features):
        # Row order must never be the only alignment mechanism.
        assert "arm_index" in features.columns
        assert features["arm_index"].is_unique

    def test_row_order_hash_matches_spec(self, features):
        import hashlib, yaml
        spec = yaml.safe_load((paths.AGENT / "arm_feature_spec.yaml").read_text())
        h = hashlib.sha256(np.ascontiguousarray(
            features["arm_index"].to_numpy(np.int64)).tobytes()).hexdigest()
        assert h == spec["row_order_sha256"]

    def test_spec_covers_every_feature_column(self, features):
        import yaml
        spec = yaml.safe_load((paths.AGENT / "arm_feature_spec.yaml").read_text())
        names = [f["name"] for f in spec["features"]]
        cols = [c for c in features.columns if c not in ("arm_index", "course_id")]
        assert names == cols
        assert spec["n_features"] == len(cols)

    def test_every_imputable_field_has_an_indicator(self, features):
        # Missingness is provider-correlated, so without these the reward
        # model becomes partly a provider classifier.
        for ind in ("level_is_imputed", "has_effort_hours", "has_calendar_weeks",
                    "rating_is_missing", "has_enrollment"):
            assert ind in features.columns

    def test_no_prerequisite_feature_leaked_in(self, features):
        banned = ("prereq", "study_plan", "milestone", "depth")
        assert not [c for c in features.columns
                    if any(b in c.lower() for b in banned)]
