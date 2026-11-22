"""Agent-generated learner personas.

Replaces hard-coded distributions with coherent people. A Beta draw gives a
number; an agent gives a learner whose stated background, available hours
and already-held skills are mutually consistent -- "bootcamp graduate, knows
HTML and CSS, never touched React, two evenings a week". That correlation
structure is exactly what independent sampling cannot produce.

Determinism is preserved the same way as the taxonomy stage: temperature 0,
every response cached by sha256(model, payload), no retry-at-higher-
temperature. A rebuild replays the cache and needs no API access.

The agent supplies the POPULATION, never the REWARD. The reward stays a
frozen specification that can be stated in a sentence; an agent judging its
own recommendations would make "bandit beat baseline" mean "bandit pleased
the judge".
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import llm

BATCH = 5

SYSTEM = (
    "You invent realistic self-paced learners for a course recommender "
    "study. Each learner is a coherent person: their background, available "
    "time and existing skills must fit together.\n\n"
    "You receive a goal (a topic family and the specific skills the learner "
    "wants to reach) plus the skills that normally come earlier in that "
    "area.\n\n"
    "Rules:\n"
    "1. Vary them. Some are well prepared, some are badly under-prepared, "
    "some are career-changers, some are students, some are working "
    "professionals with very little time.\n"
    "2. `known_skills` must be drawn ONLY from the supplied earlier_skills "
    "list, copied exactly. Pick what this particular person would plausibly "
    "already hold — an under-prepared learner picks few or none.\n"
    "3. `weekly_hours` must follow from the stated constraints. Someone with "
    "a full-time job and children does not have 20 hours.\n"
    "4. `start_level` is GIVEN to you as target_level. Every learner in this "
    "batch must be at that level, and their background must justify it — an "
    "advanced learner has real prior experience, a beginner does not. Do not "
    "drift toward beginners because the goal sounds introductory.\n"
    "5. `persistence` 0-1: how much friction they tolerate before giving up.\n"
    "6. Keep `background` to one sentence.\n"
    'Return JSON: {"learners": [{"background": "...", "start_level": 0, '
    '"weekly_hours": 5, "known_skills": ["..."], "persistence": 0.6, '
    '"novelty_pref": 0.5, "price_sensitivity": 0.4, "motivation": "..."}]}'
)


def generate(
    goal_family: str,
    goal_labels: List[str],
    earlier_labels: List[str],
    n: int,
    variant: int,
    target_level: int = 0,
    budget: Optional[llm.Budget] = None,
) -> List[Dict[str, Any]]:
    """Generate `n` learners for one goal at `target_level`.

    The level is steered rather than left to the model. Unsteered, personas
    skew hard to beginner -- a pilot produced 19/18/3 across the three
    levels -- because "I want to learn X" reads as a beginner framing. That
    would undo the stratification the population needs: for a beginner the
    difficulty term is exactly zero, so an all-beginner population makes
    difficulty matching inert.
    """
    model = llm.config()["models"]["extract"]
    user = json.dumps({
        "goal_family": goal_family,
        "goal_skills": goal_labels,
        "earlier_skills": earlier_labels,
        "how_many": n,
        "target_level": target_level,
        "variant": variant,
    }, ensure_ascii=False)
    out = llm.chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        model=model, budget=budget)
    if not out:
        return []

    allowed = set(earlier_labels)
    learners = []
    for item in (out.get("learners") or [])[:n]:
        try:
            hours = float(item.get("weekly_hours", 5))
        except (TypeError, ValueError):
            continue
        # Level is assigned, not accepted from the model: the batch was
        # requested at this level and the stratification depends on it.
        lvl = target_level
        # The model may echo a skill that was not offered; drop rather than
        # invent a mastery row for a skill outside this learner's plan.
        known = [k for k in (item.get("known_skills") or []) if k in allowed]
        learners.append({
            "background": str(item.get("background", ""))[:240],
            "motivation": str(item.get("motivation", ""))[:160],
            "start_level": max(0, min(2, lvl)),
            "weekly_hours": float(max(1.0, min(40.0, hours))),
            "known_labels": known,
            "persistence": _unit(item.get("persistence"), 0.6),
            "novelty_pref": _unit(item.get("novelty_pref"), 0.55),
            "price_sensitivity": _unit(item.get("price_sensitivity"), 0.4),
        })
    return learners


def _unit(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return round(min(1.0, max(0.0, v)), 3)
