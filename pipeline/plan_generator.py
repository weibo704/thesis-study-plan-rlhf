"""LLM study-plan generator, grounded in the real catalog.

Generates N DIVERSE candidate plans per learner in a single call. One call
rather than N keeps cost linear in learners and, more importantly, lets the
model see its own alternatives and make them genuinely different -- N
independent calls at temperature 0 would return near-identical plans and
best-of-N would have nothing to choose between.

Grounding: the model picks from a supplied shortlist of real courses and
returns their ids. It cannot invent a course. Anything it returns that is
not on the shortlist is dropped rather than repaired, so a malformed plan
is visible as a short plan instead of being silently patched.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import llm

SYSTEM = (
    "You design study plans. For one learner you produce several DISTINCT "
    "plans so they can be compared.\n\n"
    "You receive the learner's goal, level, weekly hours and background, "
    "plus a shortlist of real courses with id, level, hours, rating and the "
    "skills each teaches.\n\n"
    "Rules:\n"
    "1. Each plan is an ORDERED sequence of course ids from the shortlist. "
    "Order matters: earlier courses should build toward later ones.\n"
    "2. Use only ids from the shortlist. Never invent a course. Never repeat "
    "an id within one plan.\n"
    "3. Make the plans genuinely different from each other -- different "
    "strategies, not reorderings of the same picks. For example: one that "
    "builds foundations first, one that goes straight at the goal, one that "
    "fits a tight time budget, one that maximises coverage of missing "
    "skills.\n"
    "4. Respect the learner's weekly hours. A plan whose total far exceeds "
    "what they have is a bad plan however good the courses are.\n"
    "5. `strategy` names the approach in 2-5 words. `rationale` is at most "
    "15 words.\n"
    'Return JSON: {"plans": [{"strategy": "...", "course_ids": [3, 11, 7], '
    '"rationale": "..."}]}'
)


def generate(profile: Dict[str, Any], shortlist: List[Dict[str, Any]],
             n_plans: int, plan_length: int,
             budget: Optional[llm.Budget] = None) -> List[Dict[str, Any]]:
    payload = {
        "learner": profile,
        "how_many_plans": n_plans,
        "courses_per_plan": plan_length,
        "shortlist": shortlist,
    }
    out = llm.chat_json(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        model=llm.config()["models"]["extract"], budget=budget,
        role="plan_generator")
    if not out:
        return []

    valid_ids = {c["id"] for c in shortlist}
    plans: List[Dict[str, Any]] = []
    for item in (out.get("plans") or [])[:n_plans]:
        ids, seen = [], set()
        for cid in (item.get("course_ids") or []):
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            # Drop anything off-shortlist rather than repairing it: a
            # malformed plan should be visible as a short plan.
            if cid in valid_ids and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        if not ids:
            continue
        plans.append({
            "strategy": str(item.get("strategy", ""))[:60],
            "rationale": str(item.get("rationale", ""))[:160],
            "course_ids": ids[:plan_length],
        })
    return plans
