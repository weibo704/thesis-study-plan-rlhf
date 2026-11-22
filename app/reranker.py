"""LLM re-ranker that reasons over candidates and accumulated feedback.

The embedding scorer is a frozen matrix multiply: it matches topic and
nothing else. This stage reads the learner's actual constraints and their
past thumbs-down reasons, and reorders a shortlist accordingly.

Two kinds of learning are in play and they are NOT the same thing:

  in-context (here)   re-reads the feedback history on every request and
                      reasons over it. Adapts instantly, learns nothing
                      permanently -- remove the history and it reverts.
  parameter (LinUCB)  updates weights from observed reward. Slower, but it
                      compounds and transfers to learners never seen.

Propensity is preserved: the model returns SCORES, not a chosen item, and
the slate is sampled from a softmax over those scores. Taking the model's
top pick would log a propensity of 1.0 and destroy off-policy evaluation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline import llm

TOP_K = 15          # shortlist handed to the model
TEMPERATURE = 0.35  # softmax temperature over returned scores

SYSTEM = (
    "You re-rank a shortlist of courses for one specific learner.\n\n"
    "You receive the learner's goal, stated level, weekly hours and "
    "background; a summary of what they have REJECTED before and why; and "
    "a numbered shortlist with each course's level, duration, rating and "
    "skills.\n\n"
    "Rules:\n"
    "1. Score every course from 0.0 to 1.0. Use the full range — scores "
    "that are all similar carry no information.\n"
    "2. Honour the rejection history. If they repeatedly said things were "
    "too advanced, push advanced courses down even when the topic matches "
    "perfectly.\n"
    "3. Respect the time budget. A 40-hour course for someone with 2 hours "
    "a week is a bad recommendation however well it fits otherwise.\n"
    "4. Prefer courses that teach what they still LACK over ones covering "
    "what they already know.\n"
    "5. Never invent a course. Score only the ids given.\n"
    "6. `reason` is at most 12 words, addressed to the learner.\n"
    'Return JSON: {"ranked": [{"id": 3, "score": 0.82, '
    '"reason": "..."}]}'
)


def feedback_summary(records: List[Dict[str, Any]], limit: int = 12) -> Dict[str, Any]:
    """Condense the thumbs history into something a prompt can use."""
    if not records:
        return {"n": 0, "rejected_reasons": [], "reason_counts": {}}
    downs = [r for r in records if r.get("verdict") == "down"]
    ups = [r for r in records if r.get("verdict") == "up"]
    counts: Dict[str, int] = {}
    for r in downs:
        c = r.get("reason_category") or "other"
        counts[c] = counts.get(c, 0) + 1
    return {
        "n": len(records),
        "n_rejected": len(downs),
        "n_accepted": len(ups),
        "reason_counts": counts,
        "rejected_reasons": [r.get("reason_text") for r in downs[-limit:]
                             if r.get("reason_text")],
    }


def rerank(profile: Dict[str, Any], candidates: List[Dict[str, Any]],
           history: Dict[str, Any]) -> Optional[Dict[int, Dict[str, Any]]]:
    """Return {candidate_index: {score, reason}} or None if unavailable."""
    shortlist = []
    for i, c in enumerate(candidates[:TOP_K]):
        shortlist.append({
            "id": i,
            "title": c.get("title"),
            "family": c.get("family"),
            "level": {0: "beginner", 1: "intermediate", 2: "advanced"}.get(c.get("level")),
            "hours": c.get("hours"),
            "weeks": c.get("weeks"),
            "rating": c.get("rating"),
            "skills": (c.get("skills") or [])[:6],
        })
    payload = {"learner": profile, "feedback_history": history,
               "shortlist": shortlist}
    out = llm.chat_json(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        model=llm.config()["models"]["extract"], role="reranker")
    if not out:
        return None

    scored: Dict[int, Dict[str, Any]] = {}
    for item in (out.get("ranked") or []):
        try:
            idx = int(item.get("id"))
            sc = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(shortlist):
            scored[idx] = {"score": max(0.0, min(1.0, sc)),
                           "reason": str(item.get("reason", ""))[:120]}
    return scored or None


def sample_slate(scored: Dict[int, Dict[str, Any]], n_slots: int,
                 rng: np.random.Generator) -> Dict[str, Any]:
    """Softmax over the model's scores, then sample without replacement.

    Taking the model's argmax would log propensity 1.0 and make
    inverse-propensity scoring degenerate -- permanently, since propensities
    cannot be reconstructed after the fact.
    """
    ids = sorted(scored)
    s = np.array([scored[i]["score"] for i in ids], dtype=float)
    logits = (s - s.max()) / max(TEMPERATURE, 1e-6)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    k = min(n_slots, len(ids))
    picks = rng.choice(len(ids), size=k, replace=False, p=probs)
    return {"order": [ids[p] for p in picks],
            "probs": {ids[j]: float(probs[j]) for j in range(len(ids))},
            "prob_vector": probs.tolist(),
            "candidate_ids": ids}
