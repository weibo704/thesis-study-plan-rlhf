"""Interaction and feedback logging.

Append-only JSONL under data/agent/. Two record types:

  impression  one per recommendation turn. Records the candidate set, the
              FULL action-probability vector, the sampled slate and each
              shown item's propensity.
  feedback    one per thumbs up/down, with free-text reason and a
              categorised label.

Why the full probability vector and not just the chosen propensity: with a
small candidate set it costs nothing to store, and it keeps IPS, SNIPS and
doubly-robust estimators all available later. Propensities cannot be
reconstructed after the fact -- if they are not logged at decision time,
off-policy evaluation is impossible, permanently.

Direction of dependency: these logs live in agent/ and are read later to
ground world/reward_spec. That is agent -> world, which the firewall
permits. The banned direction is world -> agent.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline import paths

LOG_DIR = paths.AGENT / "feedback"
IMPRESSIONS = LOG_DIR / "impressions.jsonl"
FEEDBACK = LOG_DIR / "feedback.jsonl"
_LOCK = threading.Lock()

# Canonical thumbs-down reasons. These are the categories the reward spec
# will be grounded in, so the vocabulary is fixed here rather than inferred
# later from whatever free text happened to arrive.
REASON_CATEGORIES = [
    "too_advanced",     # prerequisite gap -- the asymmetric penalty case
    "too_basic",
    "too_long",
    "too_short",
    "wrong_topic",
    "low_quality",
    "already_known",
    "other",
]

CATEGORISE_SYSTEM = (
    "Classify why a learner rejected a recommended course.\n"
    "Choose exactly one category from: " + ", ".join(REASON_CATEGORIES) + ".\n"
    "Use 'other' only when none genuinely fit.\n"
    'Return JSON: {"category": "...", "confidence": 0.0-1.0}'
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_impression(
    *, session_id: str, turn_index: int, profile_text: str,
    refinements: List[str], candidate_arms: List[int],
    action_probs: List[float], shown: List[Dict[str, Any]],
    policy_id: str, temperature: float, seed: int,
    dataset_version: str, spec_version: str,
) -> str:
    event_id = str(uuid.uuid4())
    _append(IMPRESSIONS, {
        "event_id": event_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "created_at": _now(),
        "profile_text": profile_text,
        # Refinements accumulate within a session, so turns are NOT
        # independent draws. Recorded explicitly so analysis cannot
        # mistake a refined turn for a fresh one.
        "refinements": refinements,
        "policy_id": policy_id,
        "temperature": temperature,
        "seed": seed,
        "candidate_set": candidate_arms,
        "action_prob_vector": [round(p, 6) for p in action_probs],
        "shown": shown,
        "dataset_version": dataset_version,
        "feature_spec_version": spec_version,
    })
    return event_id


def log_feedback(
    *, impression_event_id: str, session_id: str, arm_index: int,
    slot_position: int, propensity: float, verdict: str,
    reason_text: Optional[str], reason_category: Optional[str],
    category_confidence: Optional[float],
) -> str:
    event_id = str(uuid.uuid4())
    _append(FEEDBACK, {
        "event_id": event_id,
        "impression_event_id": impression_event_id,
        "session_id": session_id,
        "created_at": _now(),
        "arm_index": int(arm_index),
        "slot_position": int(slot_position),
        "propensity": float(propensity),
        "verdict": verdict,                       # "up" | "down"
        "reason_text": reason_text,
        "reason_category": reason_category,
        "category_confidence": category_confidence,
    })
    return event_id


def categorise_reason(reason_text: str, course_title: str = "") -> Dict[str, Any]:
    """Map free text onto the fixed category vocabulary via a cached LLM call."""
    from pipeline import llm
    if not reason_text or not reason_text.strip():
        return {"category": None, "confidence": None}
    out = llm.chat_json(
        [{"role": "system", "content": CATEGORISE_SYSTEM},
         {"role": "user", "content": json.dumps(
             {"course": course_title, "reason": reason_text}, ensure_ascii=False)}],
        model=llm.config()["models"]["extract"])
    if not out:
        return {"category": "other", "confidence": 0.0}
    cat = str(out.get("category", "other"))
    if cat not in REASON_CATEGORIES:
        cat = "other"
    try:
        conf = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"category": cat, "confidence": conf}


def records(session_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Raw feedback rows, newest last. Scoped to a session when given."""
    if not FEEDBACK.exists():
        return []
    rows = [json.loads(l) for l in FEEDBACK.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if session_id:
        rows = [r for r in rows if r.get("session_id") == session_id]
    return rows[-limit:]


def summary() -> Dict[str, Any]:
    """Counts used to ground the reward spec."""
    if not FEEDBACK.exists():
        return {"n_feedback": 0, "by_verdict": {}, "by_reason": {}}
    rows = [json.loads(l) for l in FEEDBACK.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    return {
        "n_feedback": len(df),
        "n_sessions": int(df["session_id"].nunique()) if len(df) else 0,
        "by_verdict": df["verdict"].value_counts().to_dict() if len(df) else {},
        "by_reason": (df[df["verdict"] == "down"]["reason_category"]
                      .value_counts().to_dict() if len(df) else {}),
    }


def export_parquet() -> Dict[str, int]:
    """Materialise the JSONL logs as parquet for analysis."""
    out = {}
    for name, path in (("impressions", IMPRESSIONS), ("feedback", FEEDBACK)):
        if not path.exists():
            out[name] = 0
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            out[name] = 0
            continue
        df = pd.DataFrame(rows)
        df.to_parquet(LOG_DIR / f"{name}.parquet", index=False)
        out[name] = len(df)
    return out
