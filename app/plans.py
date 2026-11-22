"""Serving-side study plans: retrieve, generate, score, explain.

This is the online twin of stage 17. It must produce the SAME feature
vector as `pipeline.stage17_bestofn.DEFAULT_FEATURES`, in the same order,
or the weights learned offline do not apply to what a student is shown.

Where it differs from stage 17, and why: stage 17 reads a simulated
learner's hidden mastery out of `world/`. A real student has no such
record, so everything here is built from what they tell us plus the frozen
catalog. That is the firewall holding in the one place it actually matters
-- a served plan cannot consult an oracle, because for a real learner no
oracle exists.

The rationale shown alongside each plan is assembled from measured facts
(levels, hours, goal overlap), not written by a second model call. A
generated explanation could describe a plan that was not the plan, and the
whole point of showing it is that the student's feedback becomes better
informed rather than differently misled.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app import online_reward
from pipeline import llm, paths, plan_generator

SHORTLIST = 25
N_PLANS = 6
PLAN_LEN = 5
GOAL_SKILLS = 6          # how many skill labels a free-text goal maps onto

# A query is in scope when it matches the frozen skill taxonomy, which is
# the catalog's own definition of what it covers. Course-level similarity
# does NOT work: "how to trade stocks" scores 0.722 against the nearest
# course, above "become a backend web developer" at 0.702, because the
# scope classifier let finance courses into the pool.
#
# Measured over 14 in-domain and 8 out-of-domain queries, the bands very
# nearly touch: in-domain floor 0.506 ("become a data engineer"),
# out-of-domain ceiling 0.477 ("how to cook Italian food"). An earlier
# 6-vs-6 probe suggested a 0.141 gap; the wider set says 0.029. The
# separation is real for clearly unrelated topics and thin at the border.
#
# So the threshold sits just under the in-domain floor rather than midway.
# The errors are not symmetric: turning away a real participant costs a
# session, while an over-generous plan costs one comparison the person can
# reject. With a handful of testers the first is much the worse.
DOMAIN_FIT_MIN = 0.49

# Above the floor but below this, the catalog covers the topic thinly.
# Worth saying so rather than presenting a stretch as a confident answer.
DOMAIN_FIT_THIN = 0.57

# Must match pipeline.stage17_bestofn.DEFAULT_FEATURES exactly, in order.
FEATURES = ["mean_topic", "mean_too_hard", "mean_too_easy", "goal_coverage",
            "mean_rating", "pos_weighted_difficulty", "bias"]

LEVEL_NAME = {0: "beginner", 1: "intermediate", 2: "advanced"}

# 16 of the 1,899 served titles carry U+FFFD, baked in upstream at scrape
# time (reports/catalog_summary.md). The bytes are gone, but what they stood
# for is usually recoverable from context, and showing a participant
# "The Data Scientist<?>s Toolbox" costs trust that the study needs.
#
# Display only. The stored title keeps the damage, so the catalog still
# reports honestly on how much of it there is.
_MOJIBAKE = [
    (re.compile(r"(?<=[A-Za-z])\ufffd(?=s\b)"), "'"),          # Scientist?s
    (re.compile(r"(?<=[A-Za-z])\ufffd(?=(ll|ve|re|t|m|d)\b)"), "'"),
    (re.compile(r"\s\ufffd\s"), " \u2014 "),                   # spaced dash
    # Paired quotes. Both delimiters must sit against whitespace, which is
    # what keeps this from swallowing the span between two lost accents:
    # in "Aspectos b?sicos ... planificaci?n" the first ? follows a letter,
    # so it is not read as an opening quote.
    (re.compile(r"(?:(?<=\s)|^)\ufffd([A-Za-z][^\ufffd]{0,40}?)\ufffd(?=\s|$)"),
     "\u2018\\1\u2019"),
    (re.compile(r"(?<=[a-z])\ufffd(?=[A-Z])"), " "),            # in?SQL
    (re.compile(r"(?<=[A-Za-z])\ufffd(?=\s|$)"), "\u00ae"),      # Intel?, OpenVINO?
]


def display_title(title: Any) -> str:
    """Repair what the lost byte almost certainly was; leave the rest alone."""
    t = "" if title is None or (isinstance(title, float) and pd.isna(title)) else str(title)
    if "\ufffd" not in t:
        return t
    for pat, rep in _MOJIBAKE:
        t = pat.sub(rep, t)
    return re.sub(r"\s{2,}", " ", t).strip()


def plan_id(session_id: str, round_index: int, k: int) -> str:
    h = hashlib.sha256(f"{session_id}:{round_index}:{k}".encode()).hexdigest()[:10]
    return f"plan_{h}"


def load_reward_model(prereg: str = "poc-v6-stratified") -> Optional[Dict[str, Any]]:
    """The weights stage 17 fitted, if they have been written.

    Absent, plans are still generated and shown -- they just are not
    scored. That is the right failure: a study can collect preferences
    without a model, but must never show a plan ranked by weights whose
    standardisation did not travel with them.
    """
    p = paths.AGENT / f"reward_model_{prereg}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if list(d.get("feature_names") or []) != FEATURES:
        raise ValueError(
            f"reward model feature order {d.get('feature_names')} does not match "
            f"the serving order {FEATURES}; the weights would be misapplied")
    return d


class PlanService:
    """Wraps a loaded Catalog with the generate-and-explain path."""

    def __init__(self, catalog, model: Optional[Dict[str, Any]] = None) -> None:
        self.cat = catalog
        self.model = model
        # theta was fitted on standardised features, so mu and sd have to be
        # applied here too. Scoring raw features with these weights would
        # produce plausible numbers that rank the wrong plan first.
        self.theta = np.asarray(model["theta"], float) if model else None
        self.mu = np.asarray(model["standardise_mu"], float) if model else None
        self.sd = np.asarray(model["standardise_sd"], float) if model else None
        self._skill_vecs: Optional[np.ndarray] = None
        self._skill_labels: List[str] = []
        self.prior = self.theta.copy() if self.theta is not None else None
        self._live: Optional[np.ndarray] = None
        self._live_at = -1          # number of choices the cache was built on

    def live_theta(self, lr: float = online_reward.DEFAULT_LR,
                   anchor: float = online_reward.DEFAULT_ANCHOR) -> np.ndarray:
        """The prior with every real comparison folded in.

        Replayed from the event log rather than held in memory: two
        workers cannot drift apart, a restart loses nothing, and the
        inspector gets the whole history for free.
        """
        if self.prior is None:
            return None
        from app import store
        choices = store.store().read(store.CHOICE)
        if self._live is not None and self._live_at == len(choices):
            return self._live
        pairs = online_reward.pairs_from_events(
            store.store().read(store.PLANS_SHOWN), choices, self.mu, self.sd)
        theta, _ = online_reward.replay(self.prior, pairs, lr, anchor)
        self._live, self._live_at = theta, len(choices)
        return theta

    def score(self, f: np.ndarray, theta: Optional[np.ndarray] = None
              ) -> Optional[float]:
        t = self.theta if theta is None else theta
        if t is None:
            return None
        z = (f - self.mu) / np.where(self.sd < 1e-9, 1.0, self.sd)
        return float(np.dot(t, z))

    # ------------------------------------------------------------- goal set
    def _ensure_skill_vectors(self) -> None:
        """Embed the 188 skill labels once, so a free-text goal can be
        mapped onto the same skill vocabulary the reward model was fit on."""
        if self._skill_vecs is not None:
            return
        labels = sorted(self.cat.skills[self.cat.skills["tier"] == "skill"]["label"]
                        .dropna().unique()) if "tier" in self.cat.skills \
            else sorted(self.cat.skills["label"].dropna().unique())
        raw = llm.embed(list(labels), llm.config()["models"]["embed"])
        v = raw @ self.cat.components.T
        self._skill_vecs = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        self._skill_labels = list(labels)

    def goal_skills(self, query_vec: np.ndarray, k: int = GOAL_SKILLS) -> List[str]:
        self._ensure_skill_vectors()
        sims = self._skill_vecs @ query_vec
        return [self._skill_labels[i] for i in np.argsort(-sims)[:k]]

    def domain_fit(self, query_vec: np.ndarray,
                   k: int = GOAL_SKILLS) -> Tuple[float, List[str]]:
        """How well a query matches what the catalog actually teaches.

        The mean of the top k rather than the single best match: one
        spurious hit should not admit a whole query. "I want to learn
        Spanish" matches the leaf `Language Learning` at 0.638 -- high
        enough to pass a top-1 test, and nowhere near it on the mean.
        """
        self._ensure_skill_vectors()
        sims = self._skill_vecs @ query_vec
        top = np.argsort(-sims)[:k]
        return float(np.mean(sims[top])), [self._skill_labels[i] for i in top]

    # ------------------------------------------------------------ shortlist
    def shortlist(self, query_vec: np.ndarray, n: int = SHORTLIST
                  ) -> Tuple[List[Dict[str, Any]], Dict[int, int], np.ndarray]:
        topic = self.cat.vecs @ query_vec
        top = np.argsort(-topic)[:n]
        rows, id_to_arm = [], {}
        for i, p in enumerate(top):
            arm = int(self.cat.arm_order[p])
            c = self.cat.courses.loc[arm]
            id_to_arm[i] = arm
            rows.append({
                "id": i,
                "title": display_title(c["title_normalized"])[:70],
                "level": LEVEL_NAME.get(
                    None if pd.isna(c["level_canonical"]) else int(c["level_canonical"])),
                "hours": None if pd.isna(c["effort_hours_total"])
                         else round(float(c["effort_hours_total"]), 1),
                "rating": None if pd.isna(c["rating_mean"])
                          else round(float(c["rating_mean"]), 2),
            })
        return rows, id_to_arm, topic

    # --------------------------------------------------------------features
    def features(self, arms: List[int], topic: np.ndarray, user_level: float,
                 goal: Sequence[str]) -> np.ndarray:
        """The seven observable features, in DEFAULT_FEATURES order."""
        pos = [self.cat.arm_pos[int(a)] for a in arms]
        rows = self.cat.courses.loc[arms]
        lvl = pd.to_numeric(rows["level_canonical"], errors="coerce").to_numpy(float)
        rat = pd.to_numeric(rows["rating_mean"], errors="coerce").to_numpy(float)
        filled = np.nan_to_num(lvl, nan=float(user_level))
        gap = filled - float(user_level)
        n = len(arms)

        taught = set()
        for cid in rows["course_id"]:
            taught.update(self.cat.skills_by_course.get(cid, []))
        need = set(goal)

        w = np.arange(1, n + 1, dtype=float)
        return np.array([
            float(np.mean(topic[pos])),
            float(np.mean(np.clip(gap, 0, None))),
            float(np.mean(np.clip(-gap, 0, None))),
            float(len(taught & need) / max(len(need), 1)),
            float(np.nanmean(rat)) if np.isfinite(rat).any() else 0.0,
            float((filled * w).sum() / (w.sum() * 2.0)),
            1.0,
        ])

    # -------------------------------------------------------------rationale
    def rationale(self, arms: List[int], strategy: str, user_level: int,
                  weekly_hours: float, goal: Sequence[str]) -> Dict[str, Any]:
        """Plain-language grounds for showing this plan.

        Every clause is a measured fact about the courses in it. Nothing
        here is generated prose, so the student is reacting to the plan
        rather than to a persuasive description of it.
        """
        rows = self.cat.courses.loc[arms]
        lvl = pd.to_numeric(rows["level_canonical"], errors="coerce").to_numpy(float)
        hrs = pd.to_numeric(rows["effort_hours_total"], errors="coerce").to_numpy(float)
        known = lvl[~np.isnan(lvl)]
        total_hours = float(np.nansum(hrs)) if np.isfinite(hrs).any() else None

        taught = set()
        for cid in rows["course_id"]:
            taught.update(self.cat.skills_by_course.get(cid, []))
        covered = sorted(set(goal) & taught)

        points: List[str] = []
        if known.size:
            lo, hi = int(known.min()), int(known.max())
            if lo == hi:
                points.append(f"Every course is {LEVEL_NAME[lo]} level.")
            elif known.size > 1 and all(b >= a for a, b in zip(known, known[1:])):
                # Only claim a ramp when the difficulty never goes backwards.
                # Saying "builds up" about a plan that bounces between levels
                # would describe a plan the student is not actually looking at.
                points.append(f"Difficulty only goes up: {LEVEL_NAME[lo]} through "
                              f"to {LEVEL_NAME[hi]}.")
            else:
                order = " → ".join(LEVEL_NAME[int(x)] for x in known)
                points.append(f"Difficulty moves around rather than climbing: {order}.")
            above = int((known > user_level).sum())
            if above:
                points.append(
                    f"{above} of {len(arms)} {'course sits' if above == 1 else 'courses sit'} "
                    "above the level you described — a stretch, not a warm-up.")
        if total_hours:
            weeks = total_hours / max(weekly_hours, 1e-6)
            points.append(f"About {total_hours:.0f} hours in total, "
                          f"roughly {weeks:.0f} weeks at {weekly_hours:g} h/week.")
        if covered:
            points.append(f"Covers {len(covered)} of the {len(goal)} topics we matched "
                          f"to your goal: {', '.join(covered[:4])}"
                          f"{'…' if len(covered) > 4 else ''}.")
        else:
            points.append("Does not directly cover the topics we matched to your "
                          "goal — it may be groundwork instead.")
        return {"strategy": strategy, "points": points,
                "total_hours": None if total_hours is None else round(total_hours, 1),
                "covered_goal_skills": covered}

    # ----------------------------------------------------------------- main
    def build(self, query: str, user_level: int, weekly_hours: float,
              session_id: str, round_index: int, background: str = "",
              budget=None, on_generate=None) -> Optional[Dict[str, Any]]:
        qv = self.cat.embed_query(query)
        fit, goal = self.domain_fit(qv)
        if fit < DOMAIN_FIT_MIN:
            # Refused before generation, so an off-topic query costs an
            # embedding rather than a plan set. The catalog genuinely does
            # not cover this; handing back five finance courses would be a
            # worse answer than saying so.
            return {"out_of_scope": True, "fit": round(fit, 4), "nearest": goal}
        shortlist, id_to_arm, topic = self.shortlist(qv)

        profile = {"goal": query, "level": int(user_level),
                   "weekly_hours": float(weekly_hours), "background": background}
        # Charged here, immediately before the paid call and after the
        # domain gate, so an off-topic query costs an embedding instead of
        # a generation -- and a generation that fails still counts, because
        # it was still billed.
        if on_generate is not None:
            on_generate()
        raw = plan_generator.generate(profile, shortlist, N_PLANS, PLAN_LEN, budget)
        if not raw:
            return None

        live = self.live_theta()
        plans = []
        for k, item in enumerate(raw):
            arms = [id_to_arm[c] for c in item["course_ids"] if c in id_to_arm]
            if len(arms) < 2:
                continue
            f = self.features(arms, topic, user_level, goal)
            pid = plan_id(session_id, round_index, k)
            rows = self.cat.courses.loc[arms]
            plans.append({
                "plan_id": pid,
                "rank": k,
                "strategy": item.get("strategy") or f"Option {k + 1}",
                "arms": [int(a) for a in arms],
                "courses": [{
                    "arm_index": int(a),
                    "title": display_title(r["title_normalized"]),
                    "institution": None if pd.isna(r["institution"]) else str(r["institution"]),
                    "url": None if pd.isna(r["url"]) else str(r["url"]),
                    "level": None if pd.isna(r["level_canonical"])
                             else LEVEL_NAME.get(int(r["level_canonical"])),
                    "hours": None if pd.isna(r["effort_hours_total"])
                             else round(float(r["effort_hours_total"]), 1),
                    "rating": None if pd.isna(r["rating_mean"])
                              else round(float(r["rating_mean"]), 2),
                } for a, r in zip(arms, rows.to_dict("records"))],
                "features": [round(float(x), 6) for x in f],
                "score": None if live is None else round(self.score(f, live), 6),
                "score_prior": None if self.prior is None
                               else round(self.score(f, self.prior), 6),
                "rationale": self.rationale(arms, item.get("strategy") or "",
                                            user_level, weekly_hours, goal),
            })
        if len(plans) < 2:
            return None
        return {"plans": plans, "goal_skills": goal, "fit": round(fit, 4),
                "thin_coverage": fit < DOMAIN_FIT_THIN,
                "shortlist_arms": [int(a) for a in id_to_arm.values()],
                "feature_names": FEATURES}


def pair_schedule(n: int, seed: int) -> List[Tuple[int, int]]:
    """Every unordered pair of n plans, in a seeded order.

    All 15 pairs of 6 plans, shuffled per participant so that whichever
    pairs a student does not reach are not the same pairs every time.
    """
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    np.random.default_rng(seed).shuffle(pairs)
    return [tuple(p) for p in pairs]
