"""The learning environment (world/ oracle).

Holds everything a policy must NOT see: the reward spec, the prerequisite
DAG, and each learner's latent state. A policy interacts only through
`step()`, which returns a scalar reward and the observable part of the new
state.

This is also what makes the MDP framing possible: taking a course raises
mastery of the skills it teaches, so the state genuinely transitions and a
later recommendation faces a different learner. A bandit ignores that; a
planner can exploit it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from . import paths, reward


@dataclass
class LearnerState:
    user_id: str
    start_level: int
    weekly_hours: float
    goal_skills: Set[str]
    mastery: Dict[str, float]
    taken: Set[int] = field(default_factory=set)

    def level(self) -> float:
        """Observable proficiency, which GROWS as mastery accumulates.

        This previously returned start_level unchanged, which quietly
        removed the whole sequencing dynamic: if proficiency never rises,
        no course ever makes a harder course more appropriate, the value
        function is flat with respect to the action, and planning has
        nothing to exploit. Measured, one-step lookahead varied by exactly
        0.0000 across actions.

        A learner advances about one level for every ~6 skills genuinely
        mastered, capped at the top of the 0-2 scale.
        """
        gained = sum(1 for m in self.mastery.values() if m >= 0.5)
        return float(min(2.0, self.start_level + gained / 6.0))

    def needed(self) -> Set[str]:
        return {s for s in self.goal_skills if self.mastery.get(s, 0.0) < 0.5}

    def goal_progress(self) -> float:
        if not self.goal_skills:
            return 1.0
        return 1.0 - len(self.needed()) / len(self.goal_skills)


class Environment:
    """Oracle. Computes reward; policies never read the spec."""

    def __init__(self, prereg: str = "poc-v1") -> None:
        bundle_file = paths.WORLD / "preregistrations" / f"{prereg}.yaml"
        if not bundle_file.exists():      # legacy single-file layout
            bundle_file = paths.WORLD / "preregistration.yaml"
        doc = yaml.safe_load(bundle_file.read_text(encoding="utf-8"))
        self.prereg_name = doc["name"]
        fields = {k: v for k, v in doc["reward_spec"].items()
                  if k in reward.RewardSpec.__dataclass_fields__}
        self.spec = reward.RewardSpec(**fields)
        self.bundle_hash = doc["bundle_sha256"]

        feats = pd.read_parquet(paths.AGENT / "arm_features.parquet")
        courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
        meta = feats[["arm_index", "course_id"]].merge(
            courses[["course_id", "level_canonical", "effort_hours_total",
                     "rating_mean"]], on="course_id")
        self.arms = meta["arm_index"].to_numpy()
        self.pos = {int(a): i for i, a in enumerate(self.arms)}
        self.level = pd.to_numeric(meta["level_canonical"], errors="coerce").to_numpy(float)
        self.hours = pd.to_numeric(meta["effort_hours_total"], errors="coerce").to_numpy(float)
        self.rating = pd.to_numeric(meta["rating_mean"], errors="coerce").to_numpy(float)

        self.skill_cols = [c for c in feats.columns if c.startswith("skill__")]
        self.skill_ids = [c[len("skill__"):] for c in self.skill_cols]
        self.skill_mat = feats[self.skill_cols].to_numpy(np.float32)

        users = pd.read_parquet(paths.WORLD / "users.parquet")
        latent = pd.read_parquet(paths.WORLD / "user_latent_state.parquet")
        self._mastery: Dict[str, Dict[str, float]] = {}
        for r in latent.itertuples():
            self._mastery.setdefault(r.user_id, {})[r.skill_id] = float(r.mastery)
        self._users = users

    def reset(self, user_id: str) -> LearnerState:
        r = self._users[self._users["user_id"] == user_id].iloc[0]
        goal = set(r["goal_skill_ids"]) if r["goal_skill_ids"] is not None else set()
        return LearnerState(
            user_id=user_id, start_level=int(r["start_level"]),
            weekly_hours=float(r["weekly_hours"]), goal_skills=goal,
            mastery=dict(self._mastery.get(user_id, {})))

    def users(self) -> List[str]:
        return self._users["user_id"].tolist()

    def need_overlap(self, state: LearnerState, arm: int) -> float:
        """Fraction of the learner's still-missing goal skills this course teaches."""
        needed = state.needed()
        if not needed:
            return 0.0
        i = self.pos[int(arm)]
        taught = {self.skill_ids[j] for j in np.nonzero(self.skill_mat[i])[0]}
        return len(taught & needed) / len(needed)

    def terminal_reward(self, state: LearnerState) -> float:
        """Paid once at episode end, in proportion to goal attainment.

        Sparse by design: a bandit gets no signal for the whole episode and
        then one lump, which it structurally cannot attribute to any
        particular action. A planner can.
        """
        return reward.terminal(self.spec, state.goal_progress())

    def reward(self, state: LearnerState, arm: int, topic_sim: float,
               progress_delta: float = 0.0) -> float:
        i = self.pos[int(arm)]
        lvl = self.level[i]
        hrs = self.hours[i]
        rat = self.rating[i]
        comp = reward.score(
            self.spec,
            topic_sim=np.array([topic_sim]),
            course_level=np.array([lvl]), level_known=np.array([not np.isnan(lvl)]),
            course_hours=np.array([hrs]), hours_known=np.array([not np.isnan(hrs)]),
            rating=np.array([rat]), rating_known=np.array([not np.isnan(rat)]),
            user_level=state.level(), weekly_hours=state.weekly_hours,
            need_overlap=np.array([self.need_overlap(state, arm)]),
            progress_delta=np.array([progress_delta]))
        r = float(comp["total"][0])
        # Repeating a course the learner already took is worthless.
        if int(arm) in state.taken:
            r -= 0.5
        return r

    def step(self, state: LearnerState, arm: int, topic_sim: float
             ) -> Tuple[float, LearnerState, Dict[str, Any]]:
        """Apply an action: reward, then the state transition.

        The transition is what makes this an MDP. Taking a course raises
        mastery of the skills it teaches, scaled by how well the course
        matches the learner's level -- a course far beyond them teaches less.
        """
        i = self.pos[int(arm)]
        lvl = self.level[i]
        gap = 0.0 if np.isnan(lvl) else abs(lvl - state.level())
        absorption = float(np.clip(1.0 - 0.35 * gap, 0.15, 1.0))

        before = state.goal_progress()
        taught = np.nonzero(self.skill_mat[i])[0]
        for j in taught:
            sid = self.skill_ids[j]
            cur = state.mastery.get(sid, 0.0)
            state.mastery[sid] = float(min(1.0, cur + absorption * 0.45 * (1.0 - cur)))
        state.taken.add(int(arm))
        # Reward is computed AFTER the transition so the dense progress term
        # can see what this action actually achieved.
        r = self.reward(state, arm, topic_sim,
                        progress_delta=state.goal_progress() - before)
        info = {"absorption": round(absorption, 3),
                "progress_before": round(before, 3),
                "progress_after": round(state.goal_progress(), 3),
                "skills_touched": int(len(taught))}
        return r, state, info
