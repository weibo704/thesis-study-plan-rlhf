"""Append-only event storage, local or on AWS.

The study records three kinds of event: a plan set that was generated and
shown, a pairwise choice between two of those plans, and a reason given in
words. They are the thesis's primary data, so the two requirements are that
a write either lands or raises, and that nothing is ever updated in place.

Two backends behind one interface:

  jsonl     one file per event kind under data/agent/feedback/. The local
            default, diffable, and what every existing analysis reads.
  dynamodb  one table, partitioned by participant and sorted by event time.
            Used when THESIS_STORE=dynamodb, which is what the App Runner
            service sets.

The DynamoDB table is deliberately a single table with a generic shape
rather than one table per event kind: the access pattern is "give me every
event for this participant" and "give me everything, once, for export", and
both are cheaper against one table.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from pipeline import paths

LOCAL_DIR = paths.AGENT / "feedback"
_LOCK = threading.Lock()

# Event kinds. Fixed here rather than passed as free strings so a typo
# cannot silently create a fourth kind that no analysis ever reads.
PLANS_SHOWN = "plans_shown"
CHOICE = "choice"
REASON = "reason"
OUT_OF_SCOPE = "out_of_scope"
KINDS = (PLANS_SHOWN, CHOICE, REASON, OUT_OF_SCOPE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------- JSON safety
def jsonable(obj: Any) -> Any:
    """numpy scalars and Decimals do not survive json.dumps untouched."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    return obj


def _decimalise(obj: Any) -> Any:
    """DynamoDB rejects float; it wants Decimal."""
    if isinstance(obj, dict):
        return {k: _decimalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_decimalise(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _decimalise(obj.item())
        except (ValueError, AttributeError):
            pass
    return obj


# ------------------------------------------------------------------ backends
class JsonlStore:
    """One append-only file per event kind."""

    name = "jsonl"

    def __init__(self, directory: Path = LOCAL_DIR) -> None:
        self.dir = Path(directory)

    def path(self, kind: str) -> Path:
        return self.dir / f"{kind}.jsonl"

    def put(self, kind: str, record: Dict[str, Any]) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        p = self.path(kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(jsonable(record), ensure_ascii=False)
        with _LOCK:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())      # the data is the thesis; pay the fsync

    def read(self, kind: str) -> List[Dict[str, Any]]:
        p = self.path(kind)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue               # a torn final line must not kill a read
        return out

    def count(self, kind: str) -> int:
        return len(self.read(kind))


class DynamoStore:
    """Single table: partition by participant, sort by time + event id.

    Sorting on `created_at#event_id` rather than `created_at` alone keeps
    two events written in the same second from colliding, which at one
    write per click is not hypothetical.
    """

    name = "dynamodb"

    def __init__(self, table_name: Optional[str] = None) -> None:
        import boto3                                   # only needed on AWS
        self.table_name = table_name or os.environ["THESIS_TABLE"]
        self._table = boto3.resource("dynamodb").Table(self.table_name)

    def put(self, kind: str, record: Dict[str, Any]) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        item = _decimalise(dict(record))
        item["pk"] = f"P#{record.get('participant_id', 'anon')}"
        item["sk"] = f"{record['created_at']}#{record['event_id']}"
        item["kind"] = kind
        self._table.put_item(Item=item)

    def read(self, kind: str) -> List[Dict[str, Any]]:
        """Full scan. Only used for export, never on a request path."""
        from boto3.dynamodb.conditions import Attr
        out, kwargs = [], {"FilterExpression": Attr("kind").eq(kind)}
        while True:
            page = self._table.scan(**kwargs)
            out.extend(jsonable(i) for i in page.get("Items", []))
            if "LastEvaluatedKey" not in page:
                return out
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def count(self, kind: str) -> int:
        return len(self.read(kind))


_store = None


def store():
    """The process-wide store, chosen by THESIS_STORE."""
    global _store
    if _store is None:
        kind = os.environ.get("THESIS_STORE", "jsonl").lower()
        _store = DynamoStore() if kind == "dynamodb" else JsonlStore()
    return _store


def reset_for_test(instance=None) -> None:
    global _store
    _store = instance


# -------------------------------------------------------------- event writers
def log_plans_shown(*, participant_id: str, session_id: str, round_index: int,
                    query: str, level: int, weekly_hours: float,
                    plans: List[Dict[str, Any]], shortlist_arms: List[int],
                    goal_skills: List[str], policy_id: str,
                    dataset_version: str, spec_version: str,
                    domain_fit: Optional[float] = None) -> str:
    """The full candidate set, logged before anything is shown.

    Every generated plan is recorded, not just the pair displayed, so the
    analysis can tell "this plan lost" from "this plan was never offered".
    """
    event_id = new_id()
    store().put(PLANS_SHOWN, {
        "event_id": event_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "round_index": int(round_index),
        "created_at": now(),
        "query": query,
        "level": int(level),
        "weekly_hours": float(weekly_hours),
        "goal_skills": list(goal_skills),
        # Kept on every accepted query, not just refusals, so the coverage
        # distribution can be read off the log rather than guessed at.
        "domain_fit": None if domain_fit is None else float(domain_fit),
        "shortlist_arms": [int(a) for a in shortlist_arms],
        "plans": plans,
        "policy_id": policy_id,
        "dataset_version": dataset_version,
        "feature_spec_version": spec_version,
    })
    return event_id


def log_choice(*, plans_event_id: str, participant_id: str, session_id: str,
               round_index: int, left_plan_id: str, right_plan_id: str,
               chosen_plan_id: str, shown_order: List[str],
               seconds_on_screen: Optional[float]) -> str:
    """One pairwise preference: exactly what Bradley-Terry consumes.

    `shown_order` records which plan sat on the left, because position bias
    is real and an analysis that ignores it will read it as preference.
    """
    event_id = new_id()
    store().put(CHOICE, {
        "event_id": event_id,
        "plans_event_id": plans_event_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "round_index": int(round_index),
        "created_at": now(),
        "left_plan_id": left_plan_id,
        "right_plan_id": right_plan_id,
        "chosen_plan_id": chosen_plan_id,
        "shown_order": list(shown_order),
        "seconds_on_screen": seconds_on_screen,
    })
    return event_id


def log_reason(*, choice_event_id: str, participant_id: str, session_id: str,
               plan_id: str, reason_text: str, reason_categories: List[str],
               free_text_only: bool) -> str:
    event_id = new_id()
    store().put(REASON, {
        "event_id": event_id,
        "choice_event_id": choice_event_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "created_at": now(),
        "plan_id": plan_id,
        "reason_text": reason_text,
        "reason_categories": list(reason_categories),
        "free_text_only": bool(free_text_only),
    })
    return event_id


def log_out_of_scope(*, participant_id: str, session_id: str, query: str,
                     fit: float, nearest_skills: List[str]) -> str:
    """A query the catalog does not cover.

    Worth keeping rather than dropping: what students ask for and the item
    space cannot answer is a measurement of the catalog's coverage, and
    the thesis currently has no evidence on that at all.
    """
    event_id = new_id()
    store().put(OUT_OF_SCOPE, {
        "event_id": event_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "created_at": now(),
        "query": query,
        "domain_fit": float(fit),
        "nearest_skills": list(nearest_skills),
    })
    return event_id


def summary() -> Dict[str, int]:
    s = store()
    return {k: s.count(k) for k in KINDS}
