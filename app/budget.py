"""A hard ceiling on paid generation, enforced server-side.

The API budget for this study is fixed and not replaceable, and the study
link is public. Watching a cost dashboard is a human process with a long
feedback loop; a crawler that finds the endpoint at 2am does not wait for
it. So the cap lives in front of the model call, not in a monitoring tab.

Two limits, both on plan generation (the only paid call a student can
trigger):

  per participant   stops one browser from running the study repeatedly
  global per day    the backstop that bounds the worst case

Local runs count in memory. On AWS the global counter is a DynamoDB atomic
increment, so two App Runner instances cannot each spend the whole budget.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Dict, Tuple

# Deliberately low. A study of a few dozen students needs hundreds of
# generations, not thousands, and the cap is easier to raise on purpose
# than a spent budget is to recover.
PER_PARTICIPANT_PER_DAY = int(os.environ.get("THESIS_CAP_PARTICIPANT", "12"))
GLOBAL_PER_DAY = int(os.environ.get("THESIS_CAP_GLOBAL", "400"))

_lock = threading.Lock()
_local: Dict[str, int] = {}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class BudgetExceeded(Exception):
    def __init__(self, scope: str, limit: int) -> None:
        self.scope, self.limit = scope, limit
        super().__init__(f"{scope} limit of {limit} reached")


def _bump_local(key: str, limit: int, scope: str) -> int:
    with _lock:
        n = _local.get(key, 0) + 1
        if n > limit:
            raise BudgetExceeded(scope, limit)
        _local[key] = n
        return n


def _bump_dynamo(key: str, limit: int, scope: str) -> int:
    """Atomic increment, then check. Over-counting a rejected call is the
    safe direction to be wrong in."""
    import boto3
    from botocore.exceptions import ClientError
    table = boto3.resource("dynamodb").Table(os.environ["THESIS_TABLE"])
    try:
        r = table.update_item(
            Key={"pk": "COUNTER", "sk": key},
            UpdateExpression="ADD #n :one",
            ExpressionAttributeNames={"#n": "n"},
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW")
        n = int(r["Attributes"]["n"])
    except ClientError:
        # A counter that cannot be read must not become a way to bypass the
        # cap, so a failure here refuses the call.
        raise BudgetExceeded(scope, limit)
    if n > limit:
        raise BudgetExceeded(scope, limit)
    return n


def charge(participant_id: str) -> Tuple[int, int]:
    """Record one paid generation, or raise BudgetExceeded.

    Charged BEFORE the model call. A call that is made and then fails still
    costs money, so the counter has to move first.
    """
    day = _today()
    bump = _bump_dynamo if os.environ.get("THESIS_STORE", "").lower() == "dynamodb" \
        else _bump_local
    g = bump(f"GLOBAL#{day}", GLOBAL_PER_DAY, "daily study-wide")
    try:
        p = bump(f"P#{participant_id}#{day}", PER_PARTICIPANT_PER_DAY, "per-person daily")
    except BudgetExceeded:
        raise
    return p, g


def reset_for_test() -> None:
    with _lock:
        _local.clear()
