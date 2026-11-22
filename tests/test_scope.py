"""Unit tests for stage 8 scope classification."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import paths
from pipeline.stage08_scope import _cfg, _compile, _family_profiles, _score_keywords


class TestKeywordMatching:
    def test_word_boundaries_prevent_substring_matches(self):
        """'ai' must not fire on 'said', 'detail', 'training'."""
        pats = _compile(["ai"])
        _, pat = pats[0]
        assert pat.search("ai for everyone")
        assert not pat.search("he said detail")
        assert not pat.search("retail training")

    def test_multiword_terms_tolerate_extra_whitespace(self):
        _, pat = _compile(["machine learning"])[0]
        assert pat.search("applied machine  learning basics")

    def test_matching_is_case_insensitive(self):
        _, pat = _compile(["pytorch"])[0]
        assert pat.search("deep learning with PyTorch".casefold())


class TestFieldWeighting:
    def test_title_outweighs_description(self):
        cfg = _cfg()
        _, patterns, _ = _family_profiles()
        in_title = pd.Series({
            "title_normalized": "Introduction to Machine Learning",
            "raw_skills_text": None, "description": "A general course.",
        })
        in_desc = pd.Series({
            "title_normalized": "A General Course",
            "raw_skills_text": None,
            "description": "This mentions machine learning once in passing.",
        })
        t_scores, _ = _score_keywords(in_title, patterns, cfg)
        d_scores, _ = _score_keywords(in_desc, patterns, cfg)
        assert max(t_scores.values()) > max(d_scores.values())

    def test_no_signal_scores_zero(self):
        cfg = _cfg()
        _, patterns, _ = _family_profiles()
        row = pd.Series({
            "title_normalized": "Singing Popular Music",
            "raw_skills_text": "vocals breathing",
            "description": "Learn to sing.",
        })
        scores, _ = _score_keywords(row, patterns, cfg)
        assert not scores or max(scores.values()) == 0


class TestConfigIntegrity:
    def test_thresholds_are_ordered_sensibly(self):
        cfg = _cfg()
        assert cfg["min_keyword_score"] > cfg["promote_keyword_score"]
        assert cfg["promote_keyword_score"] >= cfg["ambiguous_keyword_score"]
        assert cfg["min_similarity_alone"] > cfg["min_similarity_supporting"]
        assert cfg["min_similarity_supporting"] > cfg["ambiguous_similarity"]

    def test_families_have_required_fields(self):
        # Count is not asserted: the scope revision of 2026-09-27 took it
        # from 12 to 15 when security, mobile and systems were added.
        fams = paths.load_families()["families"]
        assert len(fams) >= 12
        for f in fams:
            assert f["id"] and f["label"] and f["scope"]
            assert f.get("keywords_include"), f["id"]

    def test_family_ids_are_unique(self):
        ids = [f["id"] for f in paths.load_families()["families"]]
        assert len(ids) == len(set(ids))


@pytest.mark.skipif(not (paths.CATALOG / "courses.parquet").exists(),
                    reason="requires a built catalog")
class TestBuiltCatalog:
    @pytest.fixture(scope="class")
    def courses(self):
        return pd.read_parquet(paths.CATALOG / "courses.parquet")

    def test_nothing_is_deleted(self, courses):
        """Out-of-scope rows stay, flagged -- the funnel must stay
        reconstructable and the decision revisitable."""
        assert (courses["arm_status"] == "out_of_scope").any()
        assert (courses["arm_status"] == "active").any()
        assert set(courses["arm_status"]) <= {"active", "out_of_scope", "retired"}

    def test_out_of_scope_rows_carry_no_family(self, courses):
        """Regression: argmax-similarity was assigning a family to every row,
        producing 'Singing Popular Music' -> mlops-deployment."""
        out = courses[courses["in_scope"] == False]  # noqa: E712
        assert out["domain_family"].isna().all()

    def test_in_scope_rows_all_have_a_family(self, courses):
        live = courses[courses["in_scope"] == True]  # noqa: E712
        assert live["domain_family"].notna().all()

    def test_arm_status_matches_in_scope(self, courses):
        active = courses["arm_status"] == "active"
        assert (active == (courses["in_scope"] == True)).all()  # noqa: E712

    def test_non_english_excluded(self, courses):
        live = courses[courses["in_scope"] == True]  # noqa: E712
        assert (live["language"].fillna("English") == "English").all()
