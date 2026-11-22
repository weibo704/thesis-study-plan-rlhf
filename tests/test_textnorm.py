"""Unit tests for text repair, merge keys and union-find."""
from __future__ import annotations

import random

import pytest

from pipeline.textnorm import UnionFind, merge_key, normalize_title, repair_text, url_slug


class TestRepairText:
    def test_repairs_mojibake(self):
        assert repair_text("Ã‰cole Polytechnique") == "École Polytechnique"
        assert repair_text("Universitat PolitÃ¨cnica") == "Universitat Politècnica"

    def test_repairs_smart_quotes(self):
        assert repair_text("youâ€™ll learn") == "you'll learn"

    @pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
    def test_nullish_becomes_none(self, value):
        # Regression: float('nan') used to str() into the literal "nan" and
        # be stored as a course description.
        assert repair_text(value) is None

    @pytest.mark.parametrize("value", ["nan", "NaN", "<NA>", "None"])
    def test_null_lookalike_strings_become_none(self, value):
        assert repair_text(value) is None

    def test_collapses_whitespace(self):
        assert repair_text("  a\t\tb\n c  ") == "a b c"

    def test_is_idempotent(self):
        once = repair_text("Ã‰cole   Polytechnique")
        assert repair_text(once) == once


class TestMergeKey:
    def test_is_word_order_insensitive(self):
        assert merge_key("Automated Testing") == merge_key("testing automated")

    def test_ignores_punctuation_and_case(self):
        assert merge_key("Deep-Learning, with PyTorch!") == merge_key("pytorch deep learning")

    def test_strips_accents(self):
        assert merge_key("Introducción") == merge_key("Introduccion")

    def test_drops_stopwords(self):
        assert merge_key("Introduction to Python") == merge_key("Python")

    def test_all_stopword_input_still_keys(self):
        # Must not collapse to the empty string, or every such row collides.
        assert merge_key("Introduction") != ""

    def test_distinct_concepts_stay_distinct(self):
        assert merge_key("computer vision") != merge_key("natural language processing")


class TestUrlSlug:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.coursera.org/learn/machine-learning", "machine-learning"),
        ("https://www.coursera.org/specializations/data-science", "data-science"),
        ("https://www.edx.org/course/cs50s-intro", "cs50s-intro"),
        ("https://www.coursera.org/learn/ML-Basics", "ml-basics"),
    ])
    def test_extracts_slug(self, url, expected):
        assert url_slug(url) == expected

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_nullish(self, value):
        assert url_slug(value) is None


class TestUnionFind:
    def test_transitive_closure(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_grouping_is_order_independent(self):
        """The whole point: pairwise decisions in any order give one grouping."""
        pairs = [("a", "b"), ("c", "d"), ("b", "c"), ("e", "f")]
        baseline = None
        for seed in range(10):
            shuffled = pairs[:]
            random.Random(seed).shuffle(shuffled)
            uf = UnionFind()
            for x, y in shuffled:
                uf.union(x, y)
            groups = sorted(sorted(g) for g in uf.groups().values())
            if baseline is None:
                baseline = groups
            assert groups == baseline

    def test_root_is_deterministic_lexicographic(self):
        uf1, uf2 = UnionFind(), UnionFind()
        uf1.union("z", "a")
        uf2.union("a", "z")
        assert uf1.find("z") == uf2.find("z") == "a"

    def test_singletons_are_their_own_group(self):
        uf = UnionFind(["solo"])
        assert uf.find("solo") == "solo"
