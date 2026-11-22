"""Field-level parsers: difficulty, duration, rating, price, granularity.

These carry the decisions from docs/ingestion-plan.md section 4 that most
easily corrupt reward modelling if done naively:

  * Sentinels ("Not Calibrated", "Mixed") become nulls plus an explicit
    flag, never an imputed mode. Imputing manufactures a correlation between
    difficulty and popularity that flows straight into any level-match term.
  * Durations keep their interval. Collapsing a bucket to a midpoint and
    log-transforming produces spikes at a handful of values, so any reward
    term monotone in duration degenerates into a step function over buckets.
  * Effort hours and calendar weeks stay separate, and an hours-per-week
    assumption is NEVER baked into a course row -- that is a user parameter,
    and folding it in double-counts the learner's time budget downstream.
"""
from __future__ import annotations

import re
from typing import Dict, NamedTuple, Optional

# Difficulty sentinels that mean "unknown", not "average".
LEVEL_MISSING = {"not calibrated", "na", "n/a", "none", "unknown", ""}
# "Mixed" is a multi-level container, i.e. a signal about granularity.
LEVEL_CONTAINER = {"mixed", "all levels"}

LEVEL_MAP: Dict[str, int] = {
    "beginner": 0, "introductory": 0, "intro": 0, "basic": 0, "foundational": 0,
    "intermediate": 1,
    "advanced": 2, "expert": 2,
}

# Declared values outside the documented Beginner/Intermediate/Advanced
# vocabulary. "Conversant" is real and undocumented on the dataset page: it
# appears on 186 Coursera 2021 rows. It is a *declared* level, not a missing
# one, so mapping it is vocabulary alignment rather than imputation -- but it
# is a judgement call, so it gets its own level_source label and can be
# audited or reverted without touching the declared rows.
LEVEL_VOCAB_MAP: Dict[str, int] = {
    "conversant": 0,
}

WEEKS_PER_MONTH = 4.345  # mean weeks per calendar month


class Duration(NamedTuple):
    effort_hours_lo: Optional[float]
    effort_hours_hi: Optional[float]
    calendar_weeks_lo: Optional[float]
    calendar_weeks_hi: Optional[float]
    unit_original: Optional[str]
    source: str  # declared | bucket | derived_effort_x_weeks | unknown


EMPTY_DURATION = Duration(None, None, None, None, None, "unknown")

_NUM = r"(\d+(?:\.\d+)?)"
_RANGE_RE = re.compile(rf"{_NUM}\s*[-–—to]+\s*{_NUM}\s*(hour|week|month|day)", re.I)
_SINGLE_RE = re.compile(rf"{_NUM}\s*(hour|week|month|day)", re.I)
_LESS_RE = re.compile(rf"less than\s*{_NUM}\s*(hour|week|month|day)", re.I)
_PER_WEEK_RE = re.compile(rf"{_NUM}(?:\s*[-–]\s*{_NUM})?\s*hours?\s*(?:a|per)\s*week", re.I)


def parse_level(value: Optional[str]):
    """Return (level_canonical, level_source, is_container).

    level_canonical is ordinal 0/1/2 and must be treated as ordinal, not
    interval: Beginner->Intermediate is not the same distance as
    Intermediate->Advanced.
    """
    if value is None:
        return None, "unknown", False
    text = str(value).strip().casefold()
    if text in LEVEL_CONTAINER:
        return None, "unknown", True
    if text in LEVEL_MISSING:
        return None, "unknown", False
    for key, level in LEVEL_MAP.items():
        if key in text:
            return level, "declared", False
    for key, level in LEVEL_VOCAB_MAP.items():
        if key in text:
            return level, "declared_vocab_mapped", False
    return None, "unknown", False


def infer_level_from_text(*chunks: Optional[str]):
    """Weak fallback: read difficulty off the title/description wording.

    Only ever produces level_source='inferred_text', and always leaves
    level_is_imputed set so a model can tell it apart from a declared level.
    """
    blob = " ".join(c for c in chunks if c).casefold()
    if not blob:
        return None, "unknown"
    if re.search(r"\b(no (prior|previous) (experience|knowledge)|from scratch|"
                 r"absolute beginner|for beginners|getting started|introduction to)\b", blob):
        return 0, "inferred_text"
    if re.search(r"\b(advanced|in depth|deep dive|mastery|expert|"
                 r"assumes (strong|solid|prior))\b", blob):
        return 2, "inferred_text"
    if re.search(r"\b(intermediate|some (prior )?experience|builds on)\b", blob):
        return 1, "inferred_text"
    return None, "unknown"


def _to_weeks(value: float, unit: str) -> float:
    unit = unit.casefold()
    if unit.startswith("week"):
        return value
    if unit.startswith("month"):
        return value * WEEKS_PER_MONTH
    if unit.startswith("day"):
        return value / 7.0
    raise ValueError(unit)


def parse_duration_bucket(value: Optional[str]) -> Duration:
    """Parse a Coursera-style bucket string into an interval.

    Hour buckets are learner effort. Week and month buckets are *calendar*
    spans: converting them to effort hours would require an assumed
    hours-per-week, which is a property of the user, not the course.
    """
    if value is None:
        return EMPTY_DURATION
    text = str(value).strip()
    if not text:
        return EMPTY_DURATION

    m = _LESS_RE.search(text)
    if m:
        hi, unit = float(m.group(1)), m.group(2)
        if unit.casefold().startswith("hour"):
            return Duration(0.0, hi, None, None, text, "bucket")
        w = _to_weeks(hi, unit)
        return Duration(None, None, 0.0, w, text, "bucket")

    m = _RANGE_RE.search(text)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        if unit.casefold().startswith("hour"):
            return Duration(lo, hi, None, None, text, "bucket")
        return Duration(None, None, _to_weeks(lo, unit), _to_weeks(hi, unit), text, "bucket")

    m = _SINGLE_RE.search(text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        if unit.casefold().startswith("hour"):
            return Duration(val, val, None, None, text, "declared")
        w = _to_weeks(val, unit)
        return Duration(None, None, w, w, text, "declared")

    return Duration(None, None, None, None, text, "unknown")


def parse_hours_per_week(value: Optional[str]) -> Optional[float]:
    """Extract hours-per-week from an edX `course_effort` string."""
    if value is None:
        return None
    m = _PER_WEEK_RE.search(str(value))
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return (lo + hi) / 2.0


def parse_weeks(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    m = _SINGLE_RE.search(str(value))
    if not m:
        return None
    try:
        return _to_weeks(float(m.group(1)), m.group(2))
    except ValueError:
        return None


def effort_from_effort_and_length(
    hours_per_week: Optional[float], weeks: Optional[float]
) -> Duration:
    """edX is the only source where a defensible effort total exists.

    Requires *two* of {weeks, hours/week, total} to be genuinely known; with
    only one the total stays null rather than being invented.
    """
    if hours_per_week is None or weeks is None:
        return Duration(None, None, weeks, weeks, None, "unknown")
    total = hours_per_week * weeks
    return Duration(total, total, weeks, weeks,
                    f"{hours_per_week:g} h/wk x {weeks:g} wk",
                    "derived_effort_x_weeks")


def parse_rating(value: Optional[str]):
    """Return (rating, is_missing). 'Not Calibrated' is a sentinel, not 0."""
    if value is None:
        return None, True
    text = str(value).strip()
    if text.casefold() in LEVEL_MISSING:
        return None, True
    try:
        rating = float(text.replace(",", ""))
    except ValueError:
        return None, True
    if not (0.0 <= rating <= 5.0):
        return None, True
    return rating, False


def parse_count(value: Optional[str]) -> Optional[int]:
    """Parse enrolment/review counts written as '124,980' or '1.2k'."""
    if value is None:
        return None
    text = str(value).strip().casefold().replace(",", "")
    if not text or text in LEVEL_MISSING:
        return None
    mult = 1
    if text.endswith("k"):
        mult, text = 1_000, text[:-1]
    elif text.endswith("m"):
        mult, text = 1_000_000, text[:-1]
    m = re.search(_NUM, text)
    if not m:
        return None
    return int(float(m.group(1)) * mult)


def parse_price(value: Optional[str]):
    """Return (price_usd, access_model) from a free-text price field."""
    if value is None:
        return None, None
    text = str(value).strip().casefold()
    if not text:
        return None, None
    if "free" in text and "certificate" not in text:
        return 0.0, "free"
    m = re.search(r"\$\s*" + _NUM, text)
    if m:
        price = float(m.group(1))
        return price, ("free" if price == 0 else "paid_one_time")
    return None, None


GRANULARITY_MAP = {
    "course": "course",
    "guided project": "project",
    "project": "project",
    "specialization": "program",
    "professional certificate": "program",
    "mastertrack": "program",
    "degree": "program",
}


def parse_granularity(value: Optional[str]) -> Optional[str]:
    """Containers (Specializations, Professional Certificates) are excluded
    from the arm pool: treating a 6-month programme and a 90-minute guided
    project as interchangeable actions makes duration meaningless."""
    if value is None:
        return None
    text = str(value).strip().casefold()
    for key, out in GRANULARITY_MAP.items():
        if key in text:
            return out
    return None


def parse_pace(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).casefold()
    if "self-paced" in text or "self paced" in text:
        return "self_paced"
    if "instructor-led" in text or "instructor led" in text:
        return "scheduled"
    return None
