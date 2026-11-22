"""Reward specification: interpretable terms, tunable weights, frozen specs.

Each weight corresponds 1:1 to a thumbs-down reason category, so the
observed feedback distribution grounds the weights rather than the weights
being asserted:

    wrong_topic   -> w_topic
    (no category) -> w_goal        progress toward what the learner came for
    too_advanced  -> w_too_hard     (asymmetric with w_too_easy)
    too_basic     -> w_too_easy
    too_long      -> w_time
    low_quality   -> w_quality

The difficulty term is DELIBERATELY ASYMMETRIC. A symmetric |level_gap|
penalty makes the optimal policy level-neutral around the learner and
erases exactly the sequencing behaviour the thesis studies -- being pushed
past your depth is worse than being bored.

Freezing is the commitment point. `freeze()` writes a named spec with a
content hash; exploring weights afterwards creates a NEW named spec rather
than mutating a frozen one, so "which spec produced this result" always has
an answer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from . import paths

SPEC_DIR = paths.WORLD / "reward_specs"

# Maps each tunable weight to the feedback category that justifies it.
WEIGHT_TO_REASON = {
    "w_topic": "wrong_topic",
    "w_too_hard": "too_advanced",
    "w_too_easy": "too_basic",
    "w_time": "too_long",
    "w_quality": "low_quality",
}


@dataclass
class RewardSpec:
    name: str = "draft"
    w_topic: float = 1.00        # semantic match to the stated goal
    w_too_hard: float = 0.45     # penalty per level ABOVE the learner
    w_too_easy: float = 0.15     # penalty per level BELOW the learner
    w_time: float = 0.15         # penalty for exceeding the weekly budget
    w_quality: float = 0.10      # shrunk rating, contribution capped
    # Overlap with skills the learner still LACKS from their stated goal.
    # Absent in poc-v1, which measured only "is this a good fit right now".
    # The bandit learned a weight of +0.006 on that feature -- correctly,
    # since nothing in the reward credited it -- and so ignored goal
    # progress entirely, ending at 0.25 progress against the baseline's
    # 0.45 while earning MORE reward. That is a specification gap, not a
    # policy failure: the reward omitted the stated objective.
    w_goal: float = 0.55

    # --- goal attainment -------------------------------------------------
    # The fit terms above are myopic: they score "is this a good course for
    # you right now" and say nothing about reaching the goal. A learner can
    # take eight well-fitted courses, learn nothing they came for, and
    # collect maximum reward. These two terms fix that in different ways,
    # and the difference is the bandit/MDP contrast:
    #
    #   w_progress_step   DENSE. Pays each step for the goal progress that
    #                     step produced. A myopic policy can learn it.
    #   w_goal_terminal   SPARSE. Pays once, at the end, in proportion to
    #                     final goal attainment. Requires delayed credit
    #                     assignment, which a bandit structurally cannot do
    #                     and a planner can.
    w_progress_step: float = 0.0
    w_goal_terminal: float = 0.0
    w_unknown_penalty: float = 0.05   # small cost for missing metadata
    notes: str = ""
    grounded_in: Dict[str, int] = field(default_factory=dict)

    def asymmetry_ok(self) -> bool:
        """Too-hard must cost more than too-easy, or sequencing collapses."""
        return self.w_too_hard > self.w_too_easy

    def content_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def default_spec() -> RewardSpec:
    return RewardSpec()


def score(
    spec: RewardSpec,
    topic_sim: np.ndarray,
    course_level: np.ndarray,
    level_known: np.ndarray,
    course_hours: np.ndarray,
    hours_known: np.ndarray,
    rating: np.ndarray,
    rating_known: np.ndarray,
    user_level: float,
    weekly_hours: float,
    weeks_available: float = 4.0,
    need_overlap: Optional[np.ndarray] = None,
    progress_delta: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Return the total reward and every component, for auditability."""
    n = len(topic_sim)
    topic = spec.w_topic * np.asarray(topic_sim, dtype=float)

    gap = np.asarray(course_level, dtype=float) - float(user_level)
    gap = np.nan_to_num(gap)
    too_hard = np.clip(gap, 0, None)
    too_easy = np.clip(-gap, 0, None)
    level_term = -(spec.w_too_hard * too_hard + spec.w_too_easy * too_easy)
    level_term = np.where(level_known, level_term, 0.0)

    budget = max(float(weekly_hours) * float(weeks_available), 1e-6)
    # Clipped at 2x over budget, not 3x: beyond "twice what I have" the
    # distinction stops being meaningful, and the wider clip made time the
    # single largest term in the reward, ahead of both topic and difficulty.
    over = np.clip((np.nan_to_num(course_hours) - budget) / budget, 0, 2.0)
    time_term = np.where(hours_known, -spec.w_time * over, 0.0)

    # Ratings are a selection-biased popularity signal, not per-user utility,
    # so the term is CAPPED. Uncapped, a handful of 1-star outliers spanned
    # 0.40 of reward -- as much as the entire topic range -- which would let
    # the bandit win on popularity rather than fit.
    raw_quality = spec.w_quality * (np.nan_to_num(rating) - 4.0)
    quality = np.where(rating_known, np.clip(raw_quality, -0.15, 0.10), 0.0)

    # Missing metadata is mildly penalised rather than treated as neutral,
    # so a course with no level and no duration cannot win on silence.
    unknown = -spec.w_unknown_penalty * (
        (~np.asarray(level_known)).astype(float) + (~np.asarray(hours_known)).astype(float))

    if need_overlap is None:
        goal = np.zeros(n)
    else:
        goal = spec.w_goal * np.asarray(need_overlap, dtype=float)

    if progress_delta is None:
        prog = np.zeros(n)
    else:
        prog = spec.w_progress_step * np.asarray(progress_delta, dtype=float)

    total = topic + level_term + time_term + quality + unknown + goal + prog
    return {"total": total, "topic": topic, "level": level_term,
            "time": time_term, "quality": quality, "unknown": unknown,
            "goal": goal, "progress": prog}


def terminal(spec: RewardSpec, final_progress: float) -> float:
    """Episode-end payout, proportional to goal attainment."""
    return float(spec.w_goal_terminal) * float(final_progress)


def freeze(spec: RewardSpec, grounded_in: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Write a named, hashed spec. Refuses to overwrite an existing name."""
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    if grounded_in:
        spec.grounded_in = grounded_in
    path = SPEC_DIR / f"{spec.name}.yaml"
    if path.exists():
        raise FileExistsError(
            f"reward spec '{spec.name}' is already frozen. Choose a new name; "
            f"a frozen spec is never mutated."
        )
    doc = asdict(spec)
    doc["content_sha256"] = spec.content_hash()
    doc["dataset_version"] = paths.dataset_version()
    doc["asymmetry_ok"] = spec.asymmetry_ok()
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return doc


def list_specs() -> List[Dict[str, Any]]:
    if not SPEC_DIR.exists():
        return []
    out = []
    for p in sorted(SPEC_DIR.glob("*.yaml")):
        out.append(yaml.safe_load(p.read_text(encoding="utf-8")))
    return out


def load(name: str) -> RewardSpec:
    doc = yaml.safe_load((SPEC_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    fields = {k: v for k, v in doc.items() if k in RewardSpec.__dataclass_fields__}
    return RewardSpec(**fields)


def suggest_from_feedback(counts: Dict[str, int]) -> Dict[str, float]:
    """Propose weights from observed thumbs-down reason counts.

    A suggestion, not an answer: it scales each weight by how often that
    complaint actually occurred, leaving the ordering to be sanity-checked
    by a human before freezing.
    """
    total = sum(counts.values()) or 1
    base = default_spec()
    out = {}
    for weight, reason in WEIGHT_TO_REASON.items():
        share = counts.get(reason, 0) / total
        current = getattr(base, weight)
        # Scale within [0.5x, 2x] of the default so a single complaint
        # cannot swamp the spec.
        out[weight] = round(current * (0.5 + 1.5 * min(share * len(WEIGHT_TO_REASON), 1.0)), 3)
    return out
