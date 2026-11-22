"""Unit tests for the field parsers.

These encode the plan's section-4 decisions. A failure here means a
reward-modelling hazard has been reintroduced, not just a formatting slip.
"""
from __future__ import annotations

import pytest

from pipeline import parsers


class TestLevel:
    @pytest.mark.parametrize("raw,expected", [
        ("Beginner", 0), ("Introductory", 0), ("Intermediate", 1), ("Advanced", 2),
    ])
    def test_declared_levels(self, raw, expected):
        level, source, container = parsers.parse_level(raw)
        assert (level, source, container) == (expected, "declared", False)

    @pytest.mark.parametrize("raw", ["Not Calibrated", "NA", "", None])
    def test_sentinels_are_null_not_imputed(self, raw):
        # Imputing the mode here manufactures a correlation between level and
        # popularity that flows into any level-match reward term.
        level, source, container = parsers.parse_level(raw)
        assert level is None and source == "unknown" and container is False

    def test_mixed_is_a_container_signal_not_a_difficulty(self):
        level, _, container = parsers.parse_level("Mixed")
        assert level is None and container is True

    def test_text_inference_is_labelled_as_inferred(self):
        level, source = parsers.infer_level_from_text(
            "Python for Everybody", "A no prior experience introduction."
        )
        assert level == 0 and source == "inferred_text"

    def test_text_inference_declines_when_unsure(self):
        assert parsers.infer_level_from_text("Course", "Some words.") == (None, "unknown")


class TestDuration:
    def test_hour_bucket_is_effort(self):
        d = parsers.parse_duration_bucket("Less Than 2 Hours")
        assert (d.effort_hours_lo, d.effort_hours_hi) == (0.0, 2.0)
        assert d.calendar_weeks_lo is None

    def test_month_bucket_is_calendar_not_effort(self):
        """Converting months to effort hours needs an hours-per-week
        assumption, which is a user parameter, not a course property."""
        d = parsers.parse_duration_bucket("1 - 3 Months")
        assert d.effort_hours_lo is None and d.effort_hours_hi is None
        assert d.calendar_weeks_lo == pytest.approx(4.345)
        assert d.calendar_weeks_hi == pytest.approx(13.035)

    def test_week_bucket_is_calendar(self):
        d = parsers.parse_duration_bucket("1 - 4 Weeks")
        assert (d.calendar_weeks_lo, d.calendar_weeks_hi) == (1.0, 4.0)
        assert d.effort_hours_lo is None

    def test_interval_is_preserved_not_collapsed(self):
        d = parsers.parse_duration_bucket("1 - 4 Weeks")
        assert d.calendar_weeks_lo != d.calendar_weeks_hi

    @pytest.mark.parametrize("value", [None, "", "whenever"])
    def test_unparseable_is_unknown(self, value):
        assert parsers.parse_duration_bucket(value).source == "unknown"

    def test_effort_requires_both_operands(self):
        assert parsers.effort_from_effort_and_length(5.0, None).effort_hours_lo is None
        assert parsers.effort_from_effort_and_length(None, 6.0).effort_hours_lo is None

    def test_effort_derived_when_both_known(self):
        d = parsers.effort_from_effort_and_length(5.0, 6.0)
        assert d.effort_hours_lo == 30.0 and d.source == "derived_effort_x_weeks"
        assert d.calendar_weeks_lo == 6.0  # calendar stays separate

    def test_hours_per_week_range_averages(self):
        assert parsers.parse_hours_per_week("6-18 hours per week") == 12.0
        assert parsers.parse_hours_per_week("4 hours per week") == 4.0


class TestRatingAndCounts:
    def test_sentinel_rating_is_missing_not_zero(self):
        assert parsers.parse_rating("Not Calibrated") == (None, True)

    def test_valid_rating(self):
        assert parsers.parse_rating("4.8") == (4.8, False)

    def test_out_of_range_rejected(self):
        assert parsers.parse_rating("9.9") == (None, True)

    @pytest.mark.parametrize("raw,expected", [
        ("2,442,271", 2442271), ("1.2k", 1200), ("930", 930), (None, None),
    ])
    def test_counts(self, raw, expected):
        assert parsers.parse_count(raw) == expected


class TestGranularityAndPace:
    @pytest.mark.parametrize("raw,expected", [
        ("Course", "course"), ("Guided Project", "project"),
        ("Specialization", "program"), ("Professional Certificate", "program"),
    ])
    def test_granularity(self, raw, expected):
        assert parsers.parse_granularity(raw) == expected

    def test_pace(self):
        assert parsers.parse_pace("Self-paced on your time") == "self_paced"
        assert parsers.parse_pace("Instructor-led on a course schedule") == "scheduled"

    def test_price(self):
        assert parsers.parse_price("FREE")[1] == "free"
        assert parsers.parse_price("$99") == (99.0, "paid_one_time")


class TestUndocumentedVocabulary:
    """Real Coursera 2021 data contains difficulty values the dataset page
    does not document. They must not fall through to text inference."""

    def test_conversant_is_declared_not_inferred(self):
        level, source, container = parsers.parse_level("Conversant")
        assert level == 0
        assert source == "declared_vocab_mapped"
        assert container is False

    def test_vocab_mapped_is_distinguishable_from_plain_declared(self):
        assert parsers.parse_level("Beginner")[1] == "declared"
        assert parsers.parse_level("Conversant")[1] != "declared"
