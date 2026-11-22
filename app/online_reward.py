"""Online Bradley-Terry: each real preference moves the recommendation.

The offline model is fitted on 1,800 simulated comparisons. Two people
will not produce 1,800 of anything, and a batch refit on thirty pairs
from a cold start would sit near zero and change nothing visible. So:

  1. Start from the simulated weights as a PRIOR rather than from zero.
     Thirty real comparisons cannot identify seven weights on their own,
     but they can say where the simulated ones were wrong.

  2. Apply one gradient step per comparison, with the step size exposed
     as a dial. The Bradley-Terry gradient is self-scaling -- a pair the
     model already gets right moves it barely at all, a pair it gets
     wrong moves it a lot -- so "aggressive" sharpens disagreement rather
     than amplifying noise uniformly.

  3. Anchor back to the prior with a pull term, so one stray click cannot
     wipe out the offline fit.

        theta <- theta + lr * [ (1 - p) * x  -  anchor * (theta - prior) ]
        where x = phi(chosen) - phi(rejected),  p = sigmoid(theta . x)

No mutable server state. Theta is REPLAYED from the event log on every
read, so two App Runner workers cannot hold diverging copies, a restart
loses nothing, and the inspector can show the whole history for free.
At a few hundred events the replay costs microseconds.

What this is not: a way to get a statistically meaningful fit out of a
small sample. It is a way to make the mechanism legible and to let a
handful of real comparisons visibly reshape what gets recommended.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Defaults chosen so a single comparison the model got wrong moves the
# ranking of a six-plan set noticeably, without one click dominating ten.
DEFAULT_LR = 0.35
DEFAULT_ANCHOR = 0.02


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def replay(prior: np.ndarray, pairs: Sequence[Dict[str, Any]],
           lr: float = DEFAULT_LR, anchor: float = DEFAULT_ANCHOR
           ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Fold every comparison into the prior, in order, keeping the trace.

    `pairs` carries standardised feature vectors for the chosen and
    rejected plan. The trace is what the inspector renders: for each step,
    what the model believed, how surprised it was, and what moved.
    """
    theta = np.asarray(prior, float).copy()
    history: List[Dict[str, Any]] = []
    for i, p in enumerate(pairs):
        x = np.asarray(p["chosen"], float) - np.asarray(p["rejected"], float)
        before = theta.copy()
        pred = sigmoid(float(theta @ x))
        grad = (1.0 - pred) * x - anchor * (theta - prior)
        theta = theta + lr * grad
        history.append({
            "index": i,
            "event_id": p.get("event_id"),
            "participant_id": p.get("participant_id"),
            "created_at": p.get("created_at"),
            # Probability the model assigned to the plan the person picked,
            # BEFORE seeing their answer. Low means it was surprised, and a
            # surprised model is one that learns something.
            "p_before": round(pred, 4),
            "p_after": round(sigmoid(float(theta @ x)), 4),
            "surprise": round(1.0 - pred, 4),
            "delta": [round(float(d), 4) for d in (theta - before)],
            "theta": [round(float(t), 4) for t in theta],
        })
    return theta, history


def score(theta: np.ndarray, features: Sequence[float],
          mu: Sequence[float], sd: Sequence[float]) -> float:
    f = np.asarray(features, float)
    z = (f - np.asarray(mu, float)) / np.where(np.asarray(sd, float) < 1e-9, 1.0,
                                               np.asarray(sd, float))
    return float(np.asarray(theta, float) @ z)


def standardise(features: Sequence[float], mu: Sequence[float],
                sd: Sequence[float]) -> np.ndarray:
    f = np.asarray(features, float)
    s = np.asarray(sd, float)
    return (f - np.asarray(mu, float)) / np.where(s < 1e-9, 1.0, s)


def rank_change(prior_theta: np.ndarray, live_theta: np.ndarray,
                plans: List[Dict[str, Any]], mu, sd) -> List[Dict[str, Any]]:
    """How the running order of one plan set changed, prior vs now.

    This is the claim the whole study rests on, made checkable on a single
    screen: these comparisons moved THIS plan from fourth to first.
    """
    rows = []
    for p in plans:
        f = p.get("features")
        if not f:
            continue
        rows.append({
            "plan_id": p.get("plan_id"),
            "strategy": p.get("strategy"),
            "before": score(prior_theta, f, mu, sd),
            "after": score(live_theta, f, mu, sd),
        })
    if not rows:
        return []
    order_before = sorted(range(len(rows)), key=lambda i: -rows[i]["before"])
    order_after = sorted(range(len(rows)), key=lambda i: -rows[i]["after"])
    rank_before = {j: r + 1 for r, j in enumerate(order_before)}
    rank_after = {j: r + 1 for r, j in enumerate(order_after)}
    out = []
    for i, r in enumerate(rows):
        out.append({
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in r.items()},
            "rank_before": rank_before[i],
            "rank_after": rank_after[i],
            "moved": rank_before[i] - rank_after[i],
        })
    return sorted(out, key=lambda r: r["rank_after"])


def pairs_from_events(plans_shown: List[Dict[str, Any]],
                      choices: List[Dict[str, Any]],
                      mu: Sequence[float], sd: Sequence[float]
                      ) -> List[Dict[str, Any]]:
    """Join choices to the plan sets they came from, standardised.

    Choices whose plan set is missing are dropped rather than guessed at;
    a preference with no features attached teaches nothing.
    """
    feats: Dict[str, List[float]] = {}
    for ev in plans_shown:
        for p in ev.get("plans", []):
            if p.get("plan_id") and p.get("features"):
                feats[p["plan_id"]] = [float(x) for x in p["features"]]

    out = []
    for ch in sorted(choices, key=lambda c: c.get("created_at") or ""):
        chosen = ch.get("chosen_plan_id")
        rejected = (ch.get("right_plan_id") if chosen == ch.get("left_plan_id")
                    else ch.get("left_plan_id"))
        if chosen not in feats or rejected not in feats:
            continue
        out.append({
            "event_id": ch.get("event_id"),
            "participant_id": ch.get("participant_id"),
            "created_at": ch.get("created_at"),
            "chosen_plan_id": chosen, "rejected_plan_id": rejected,
            "chosen": standardise(feats[chosen], mu, sd),
            "rejected": standardise(feats[rejected], mu, sd),
        })
    return out
