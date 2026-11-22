"""Tests for interaction logging and feedback capture."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import feedback


class TestReasonVocabulary:
    def test_categories_cover_the_reward_terms(self):
        """Each category must map to something the reward spec can weight."""
        for needed in ("too_advanced", "too_basic", "too_long", "wrong_topic"):
            assert needed in feedback.REASON_CATEGORIES

    def test_other_is_available_as_a_fallback(self):
        assert "other" in feedback.REASON_CATEGORIES


class TestLogging:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(feedback, "LOG_DIR", tmp_path)
        monkeypatch.setattr(feedback, "IMPRESSIONS", tmp_path / "impressions.jsonl")
        monkeypatch.setattr(feedback, "FEEDBACK", tmp_path / "feedback.jsonl")

    def _impression(self, **over):
        kw = dict(session_id="s1", turn_index=0, profile_text="learn python",
                  refinements=[], candidate_arms=[1, 2, 3],
                  action_probs=[0.5, 0.3, 0.2],
                  shown=[{"arm_index": 1, "slot_position": 0, "propensity": 0.5}],
                  policy_id="softmax_text_v1", temperature=0.3, seed=7,
                  dataset_version="v1", spec_version="arm-spec-v1")
        kw.update(over)
        return feedback.log_impression(**kw)

    def test_impression_records_full_probability_vector(self):
        """Not just the chosen propensity: the full vector is what keeps
        IPS, SNIPS and doubly-robust estimators available later."""
        self._impression()
        rec = json.loads(feedback.IMPRESSIONS.read_text().splitlines()[0])
        assert len(rec["action_prob_vector"]) == len(rec["candidate_set"])
        assert abs(sum(rec["action_prob_vector"]) - 1.0) < 1e-6

    def test_impression_records_policy_and_seed(self):
        self._impression()
        rec = json.loads(feedback.IMPRESSIONS.read_text().splitlines()[0])
        for field in ("policy_id", "temperature", "seed", "dataset_version",
                      "feature_spec_version"):
            assert rec[field] is not None

    def test_propensities_are_never_degenerate(self):
        """A deterministic policy would log 1.0 and make IPS undefined."""
        self._impression()
        rec = json.loads(feedback.IMPRESSIONS.read_text().splitlines()[0])
        assert all(0.0 < p <= 1.0 for p in rec["action_prob_vector"])
        for shown in rec["shown"]:
            assert 0.0 < shown["propensity"] <= 1.0

    def test_session_and_turn_are_recorded(self):
        """Refinement makes turns conditional on earlier ones, so they are
        not independent draws and must be linkable."""
        self._impression(turn_index=0)
        self._impression(turn_index=1, refinements=["too advanced"])
        recs = [json.loads(l) for l in feedback.IMPRESSIONS.read_text().splitlines()]
        assert [r["turn_index"] for r in recs] == [0, 1]
        assert recs[1]["refinements"] == ["too advanced"]
        assert recs[0]["session_id"] == recs[1]["session_id"]

    def test_feedback_links_back_to_its_impression(self):
        eid = self._impression()
        feedback.log_feedback(impression_event_id=eid, session_id="s1", arm_index=1,
                              slot_position=0, propensity=0.5, verdict="down",
                              reason_text="too hard", reason_category="too_advanced",
                              category_confidence=0.9)
        rec = json.loads(feedback.FEEDBACK.read_text().splitlines()[0])
        assert rec["impression_event_id"] == eid
        assert rec["verdict"] == "down"
        assert rec["reason_category"] == "too_advanced"

    def test_logs_are_append_only(self):
        self._impression()
        self._impression()
        assert len(feedback.IMPRESSIONS.read_text().strip().splitlines()) == 2

    def test_summary_counts_reasons(self):
        eid = self._impression()
        for cat in ("too_advanced", "too_advanced", "too_long"):
            feedback.log_feedback(impression_event_id=eid, session_id="s1", arm_index=1,
                                  slot_position=0, propensity=0.5, verdict="down",
                                  reason_text="x", reason_category=cat,
                                  category_confidence=0.9)
        s = feedback.summary()
        assert s["n_feedback"] == 3
        assert s["by_reason"]["too_advanced"] == 2


class TestSoftmaxPolicy:
    def test_temperature_keeps_good_items_likely(self):
        """Sanity on the logging policy: sampling must not be uniform, or
        the recommendations stop being recommendations."""
        scores = np.array([0.9, 0.8, 0.5, 0.2])
        probs = np.exp((scores - scores.max()) / 0.30)
        probs /= probs.sum()
        assert probs[0] > probs[-1] * 5
        assert abs(probs.sum() - 1.0) < 1e-9
        assert all(p > 0 for p in probs)   # every arm retains support
