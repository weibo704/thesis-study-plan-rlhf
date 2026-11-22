"""The student study: storage, spend caps, and the feature contract.

The contract test is the important one. The served plan score applies
weights fitted offline, so if the serving feature order ever drifts from
the order stage 17 fitted on, every plan is still scored, every number
still looks reasonable, and the wrong plan is ranked first. Nothing else
in the system would notice.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import budget, plans as planlib, store
from pipeline import stage17_bestofn as s17


class TestFeatureContract:
    def test_serving_features_match_the_fitted_order(self):
        assert planlib.FEATURES == s17.DEFAULT_FEATURES, (
            "app/plans.py serves features in a different order than stage 17 "
            "fitted them in; the learned weights would be misapplied")

    def test_plan_length_and_shortlist_match_the_experiment(self):
        assert planlib.SHORTLIST == s17.SHORTLIST
        assert planlib.PLAN_LEN == s17.PLAN_LEN
        assert planlib.N_PLANS == s17.N_PLANS

    def test_saved_model_is_rejected_if_its_feature_order_differs(self, tmp_path, monkeypatch):
        from pipeline import paths
        bad = {"feature_names": list(reversed(planlib.FEATURES)),
               "theta": [0.0] * 7, "standardise_mu": [0.0] * 7,
               "standardise_sd": [1.0] * 7}
        monkeypatch.setattr(paths, "AGENT", tmp_path)
        (tmp_path / "reward_model_x.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="feature order"):
            planlib.load_reward_model("x")

    def test_missing_model_is_not_an_error(self, tmp_path, monkeypatch):
        from pipeline import paths
        monkeypatch.setattr(paths, "AGENT", tmp_path)
        assert planlib.load_reward_model("nope") is None


class TestDomainGate:
    """A query the catalog cannot serve must be refused, not answered.

    The pool holds finance courses the scope classifier let through, so
    "how to trade stocks" retrieves real, confidently-matched, useless
    courses. Course-level similarity cannot catch that -- it scores that
    query ABOVE "become a backend web developer". The skill taxonomy can.
    """

    def test_threshold_sits_between_the_measured_bands(self):
        # Measured over 14 in-domain and 8 out-of-domain queries: in-domain
        # floor 0.506 ("become a data engineer"), out-of-domain ceiling
        # 0.477 ("how to cook Italian food"). A narrow gap, so the margin is
        # spent on the in-domain side -- turning a real participant away is
        # worse than one stretch of a plan they can reject.
        assert 0.477 < planlib.DOMAIN_FIT_MIN <= 0.506

    def test_thin_coverage_band_sits_above_the_floor(self):
        assert planlib.DOMAIN_FIT_THIN > planlib.DOMAIN_FIT_MIN

    def test_mean_not_max_so_one_spurious_hit_cannot_admit_a_query(self):
        import inspect
        src = inspect.getsource(planlib.PlanService.domain_fit)
        assert "np.mean" in src, (
            "domain_fit must average the top matches: 'I want to learn "
            "Spanish' hits the leaf `Language Learning` at 0.638, which "
            "passes a top-1 test on its own")

    def test_out_of_scope_short_circuits_before_generation(self, monkeypatch):
        """The refusal must happen before the paid call, not after."""
        import numpy as np
        svc = planlib.PlanService.__new__(planlib.PlanService)
        svc.theta = None
        svc.domain_fit = lambda qv, k=6: (0.10, ["x"])
        svc.cat = type("C", (), {
            "embed_query": staticmethod(lambda q: np.zeros(64))})()
        charged = []
        out = planlib.PlanService.build(
            svc, "how to trade stocks", 1, 5, "s", 0,
            on_generate=lambda: charged.append(1))
        assert out["out_of_scope"] is True
        assert not charged, "an out-of-scope query must not be billed"


class TestPairSchedule:
    def test_covers_every_unordered_pair(self):
        got = planlib.pair_schedule(6, 7)
        assert len(got) == 15
        assert len(set(got)) == 15
        assert all(a < b for a, b in got)

    def test_seed_decides_the_order(self):
        assert planlib.pair_schedule(6, 1) == planlib.pair_schedule(6, 1)
        assert planlib.pair_schedule(6, 1) != planlib.pair_schedule(6, 2)


class TestDisplayTitle:
    def test_repairs_the_recoverable_damage(self):
        f = planlib.display_title
        assert f("The Data Scientist�s Toolbox") == "The Data Scientist's Toolbox"
        assert f("Smart Cities � Management") == "Smart Cities — Management"
        assert f("Intel� Network Academy") == "Intel® Network Academy"
        assert f("What is �the mind� now") == "What is ‘the mind’ now"

    def test_leaves_unrecoverable_damage_alone(self):
        # Two lost accents inside words must not be read as a quoted span.
        out = planlib.display_title("Aspectos b�sicos de la gesti�n")
        assert out == "Aspectos b�sicos de la gesti�n"

    def test_is_a_no_op_on_clean_text(self):
        for t in ("Python for Everybody", "A/B Testing", "", "C++ 101"):
            assert planlib.display_title(t) == t


class TestStore:
    @pytest.fixture(autouse=True)
    def scratch(self, tmp_path):
        store.reset_for_test(store.JsonlStore(tmp_path))
        yield
        store.reset_for_test(None)

    def test_round_trip(self):
        store.log_plans_shown(
            participant_id="p1", session_id="s1", round_index=0, query="q",
            level=0, weekly_hours=5, plans=[{"plan_id": "a"}], shortlist_arms=[1, 2],
            goal_skills=["x"], policy_id="v1", dataset_version="d", spec_version="s")
        rows = store.store().read(store.PLANS_SHOWN)
        assert len(rows) == 1 and rows[0]["participant_id"] == "p1"

    def test_numpy_scalars_survive_serialisation(self):
        store.log_plans_shown(
            participant_id="p1", session_id="s1", round_index=np.int64(0), query="q",
            level=np.int64(2), weekly_hours=np.float64(5.0),
            plans=[{"f": [np.float32(0.5)]}], shortlist_arms=[np.int64(7)],
            goal_skills=["x"], policy_id="v1", dataset_version="d", spec_version="s")
        rows = store.store().read(store.PLANS_SHOWN)
        assert rows[0]["shortlist_arms"] == [7]
        assert rows[0]["plans"][0]["f"] == [0.5]

    def test_appends_never_overwrite(self):
        for i in range(3):
            store.log_choice(
                plans_event_id="e", participant_id="p", session_id="s",
                round_index=i, left_plan_id="a", right_plan_id="b",
                chosen_plan_id="a", shown_order=["a", "b"], seconds_on_screen=1.0)
        assert len(store.store().read(store.CHOICE)) == 3

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            store.store().put("not_a_kind", {"event_id": "x", "created_at": "t"})

    def test_a_torn_line_does_not_kill_the_read(self):
        s = store.store()
        store.log_choice(plans_event_id="e", participant_id="p", session_id="s",
                         round_index=0, left_plan_id="a", right_plan_id="b",
                         chosen_plan_id="a", shown_order=["a", "b"],
                         seconds_on_screen=None)
        with open(s.path(store.CHOICE), "a") as fh:
            fh.write('{"event_id": "half')
        assert len(s.read(store.CHOICE)) == 1


class TestBudget:
    @pytest.fixture(autouse=True)
    def clean(self, monkeypatch):
        budget.reset_for_test()
        monkeypatch.delenv("THESIS_STORE", raising=False)
        yield
        budget.reset_for_test()

    def test_participant_cap_bites(self, monkeypatch):
        monkeypatch.setattr(budget, "PER_PARTICIPANT_PER_DAY", 3)
        monkeypatch.setattr(budget, "GLOBAL_PER_DAY", 100)
        for _ in range(3):
            budget.charge("p1")
        with pytest.raises(budget.BudgetExceeded):
            budget.charge("p1")

    def test_one_participant_cannot_exhaust_another(self, monkeypatch):
        monkeypatch.setattr(budget, "PER_PARTICIPANT_PER_DAY", 2)
        monkeypatch.setattr(budget, "GLOBAL_PER_DAY", 100)
        budget.charge("p1"); budget.charge("p1")
        budget.charge("p2")          # must still be allowed

    def test_global_cap_bounds_the_worst_case(self, monkeypatch):
        monkeypatch.setattr(budget, "PER_PARTICIPANT_PER_DAY", 100)
        monkeypatch.setattr(budget, "GLOBAL_PER_DAY", 4)
        for i in range(4):
            budget.charge(f"p{i}")
        with pytest.raises(budget.BudgetExceeded):
            budget.charge("p99")


class TestStudyRoutes:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from app import study
        from flask import Flask
        store.reset_for_test(store.JsonlStore(tmp_path))
        budget.reset_for_test()
        app = Flask(__name__, template_folder="../app/templates")
        app.register_blueprint(study.bp)
        yield app.test_client()
        store.reset_for_test(None)

    def test_short_query_is_refused(self, client):
        r = client.post("/api/study/start", json={"query": "hi", "level": 0,
                                                  "weekly_hours": 5})
        assert r.status_code == 400

    def test_out_of_range_inputs_are_refused(self, client):
        for bad in ({"level": 9, "weekly_hours": 5}, {"level": 0, "weekly_hours": 999}):
            r = client.post("/api/study/start",
                            json={"query": "a real goal here", **bad})
            assert r.status_code == 400

    def test_a_choice_outside_the_pair_is_refused(self, client):
        r = client.post("/api/study/choose", json={
            "plans_event_id": "e", "left_plan_id": "a", "right_plan_id": "b",
            "chosen_plan_id": "c"})
        assert r.status_code == 400
        assert store.store().count(store.CHOICE) == 0

    def test_a_valid_choice_is_recorded(self, client):
        r = client.post("/api/study/choose", json={
            "plans_event_id": "e", "left_plan_id": "a", "right_plan_id": "b",
            "chosen_plan_id": "b", "participant_id": "p-abcdefgh",
            "session_id": "s", "round_index": 1, "seconds_on_screen": 4.2})
        assert r.status_code == 200
        row = store.store().read(store.CHOICE)[0]
        assert row["chosen_plan_id"] == "b" and row["shown_order"] == ["a", "b"]

    def test_unknown_reason_categories_are_dropped(self, client):
        client.post("/api/study/reason", json={
            "choice_event_id": "c", "participant_id": "p-abcdefgh",
            "reason_categories": ["too_basic", "made_up"], "reason_text": "x"})
        row = store.store().read(store.REASON)[0]
        assert row["reason_categories"] == ["too_basic"]

    def test_empty_reason_is_not_stored(self, client):
        r = client.post("/api/study/reason", json={"choice_event_id": "c"})
        assert r.get_json()["skipped"] is True
        assert store.store().count(store.REASON) == 0

    def test_out_of_scope_is_logged_not_discarded(self, client):
        store.log_out_of_scope(
            participant_id="p-abcdefgh", session_id="s",
            query="how to trade stocks", fit=0.31, nearest_skills=["Risk Management"])
        rows = store.store().read(store.OUT_OF_SCOPE)
        assert len(rows) == 1 and rows[0]["domain_fit"] == 0.31

    def test_a_forged_participant_id_is_replaced(self, client):
        client.post("/api/study/choose", json={
            "plans_event_id": "e", "left_plan_id": "a", "right_plan_id": "b",
            "chosen_plan_id": "a", "participant_id": "../../etc/passwd"})
        assert store.store().read(store.CHOICE)[0]["participant_id"].startswith("anon-")
