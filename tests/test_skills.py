"""Unit tests for stage 7 deterministic skill extraction."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import paths
from pipeline.stage07_skills import _segment, build_vocabulary


class TestSegmentation:
    @pytest.fixture
    def lookup(self):
        return {"machine learning": "Machine Learning", "data": "Data",
                "data analysis": "Data Analysis", "python": "Python"}

    def test_prefers_longest_match(self, lookup):
        """'data' is a substring of 'data analysis'; short-first matching
        would shred the real tag."""
        matched, _ = _segment("data analysis python", lookup)
        assert "Data Analysis" in matched
        assert "Data" not in matched

    def test_unmatched_words_become_residue(self, lookup):
        matched, residue = _segment("quantum flapdoodle python", lookup)
        assert matched == ["Python"]
        assert "quantum" in residue and "flapdoodle" in residue

    def test_matching_is_case_insensitive(self, lookup):
        matched, _ = _segment("Machine Learning", lookup)
        assert matched == ["Machine Learning"]

    def test_empty_input_is_safe(self, lookup):
        assert _segment("", lookup) == ([], "")


class TestVocabulary:
    def test_built_only_from_delimited_source(self):
        """The 2021 blob is undelimited; splitting it on whitespace would
        fabricate tokens, so it must never seed the vocabulary."""
        df = pd.DataFrame([
            {"source_id": "coursera2025",
             "raw_skills_text": "Machine Learning, Data Analysis, Python"},
            {"source_id": "coursera2021",
             "raw_skills_text": "html php Python Programming credential c"},
        ])
        vocab = build_vocabulary(df)
        phrases = set(vocab["phrase"])
        assert {"Machine Learning", "Data Analysis", "Python"} <= phrases
        assert "html php Python Programming credential c" not in phrases
        assert "credential" not in phrases

    def test_word_order_variants_collapse(self):
        df = pd.DataFrame([{"source_id": "coursera2025",
                            "raw_skills_text": "Data Analysis, Analysis Data"}])
        assert len(build_vocabulary(df)) == 1

    def test_stop_phrases_dropped(self):
        df = pd.DataFrame([{"source_id": "coursera2025",
                            "raw_skills_text": "Fundamentals, Overview, Python"}])
        assert set(build_vocabulary(df)["phrase"]) == {"Python"}


@pytest.mark.skipif(not (paths.CATALOG / "course_skills.parquet").exists(),
                    reason="requires a built catalog")
class TestBuiltSkills:
    @pytest.fixture(scope="class")
    def skills(self):
        return pd.read_parquet(paths.CATALOG / "course_skills.parquet")

    def test_no_duplicate_skill_per_course(self, skills):
        assert not skills.duplicated(subset=["course_id", "skill_key"]).any()

    def test_provenance_recorded_on_every_assignment(self, skills):
        assert set(skills["tag_source"]) <= {"source_field", "dictionary_matched"}
        assert skills["tag_confidence"].notna().all()

    def test_donor_skills_rank_above_segmented(self, skills):
        """Clean delimited tags must carry more confidence than a
        dictionary guess, so downstream weighting can tell them apart."""
        by_source = skills.groupby("tag_source")["tag_confidence"].max()
        if {"source_field", "dictionary_matched"} <= set(by_source.index):
            assert by_source["source_field"] > by_source["dictionary_matched"]

    def test_all_assignments_are_teaches(self, skills):
        # 'requires' edges come from syllabus extraction, not this stage.
        assert set(skills["relation"]) == {"teaches"}

    def test_skills_only_attach_to_in_scope_courses(self, skills):
        courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
        live = set(courses[courses["in_scope"] == True]["course_id"])  # noqa: E712
        assert set(skills["course_id"]) <= live
