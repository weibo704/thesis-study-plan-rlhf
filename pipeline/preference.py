"""Learn a reward model from FEEDBACK, not from the oracle scalar.

Stage 17's first reward model regressed directly on the environment's
return. That is an oracle: it hands the model the exact quantity it is
being asked to predict. Real feedback is nothing like that -- a learner
says "this one, not that one", or "too advanced", and never emits a number.

So this module learns the way RLHF does, from PAIRWISE PREFERENCES under a
Bradley-Terry model:

    P(a preferred over b) = sigmoid(theta . (phi_a - phi_b))

fit by logistic regression on the feature DIFFERENCE. The model never sees
a reward value, only which of two plans a simulated learner preferred, so
it has to infer what drives preference from the comparisons alone.

Simulated feedback is deliberately noisy. A learner comparing two similar
plans picks almost at random; the wider the true gap, the more reliable
they are. A model trained on noiseless preferences would look far better
than one trained on anything a real person produces.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Which reward component dominated a rejection, mapped to the vocabulary
# the UI already collects.
COMPONENT_TO_REASON = {
    "too_hard": "too_advanced",
    "too_easy": "too_basic",
    "time": "too_long",
    "topic": "wrong_topic",
    "goal": "wrong_topic",
    "quality": "low_quality",
}


def simulate_choice(ret_a: float, ret_b: float, sensitivity: float,
                    rng: np.random.Generator) -> int:
    """Which plan a simulated learner prefers. 1 if a, 0 if b.

    Logistic in the true return gap: near-identical plans are a coin flip,
    clearly different ones are reliably ranked. Noiseless preferences would
    flatter the model relative to anything a real learner produces.
    """
    p = 1.0 / (1.0 + np.exp(-sensitivity * (ret_a - ret_b)))
    return int(rng.random() < p)


def build_pairs(groups: Dict[str, List[Dict[str, Any]]], sensitivity: float,
                rng: np.random.Generator, max_pairs_per_group: int = 6
                ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """All within-learner plan pairs, labelled by simulated preference.

    Pairs are within a learner only. Comparing plans across learners would
    leak learner difficulty into the label -- the model would learn who is
    easy to please rather than what makes a plan good.
    """
    X, y, meta = [], [], []
    for uid, items in groups.items():
        if len(items) < 2:
            continue
        idx = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]
        rng.shuffle(idx)
        for i, j in idx[:max_pairs_per_group]:
            a, b = items[i], items[j]
            pref_a = simulate_choice(a["true_return"], b["true_return"],
                                     sensitivity, rng)
            fa, fb = np.asarray(a["features"]), np.asarray(b["features"])
            # Bradley-Terry on the feature difference.
            X.append(fa - fb)
            y.append(pref_a)
            meta.append({"user_id": uid, "a": i, "b": j,
                         "true_gap": a["true_return"] - b["true_return"],
                         "chose_better": int(pref_a == int(
                             a["true_return"] > b["true_return"]))})
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), meta


def fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0,
        iters: int = 300, lr: float = 0.5) -> np.ndarray:
    """Logistic regression by gradient descent, L2 regularised.

    No intercept: a preference model is scale- and shift-invariant in the
    features, and a constant cannot express "a is better than b".
    """
    theta = np.zeros(X.shape[1])
    n = max(len(X), 1)
    for _ in range(iters):
        z = X @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = X.T @ (p - y) / n + lam * theta / n
        theta -= lr * grad
    return theta


def accuracy(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    pred = (X @ theta > 0).astype(int)
    return float((pred == y).mean())


def reason_for_rejection(components: Dict[str, float]) -> str:
    """The complaint a learner would voice, from the worst-scoring term."""
    negatives = {k: v for k, v in components.items() if v < 0}
    if not negatives:
        return "other"
    worst = min(negatives, key=negatives.get)
    return COMPONENT_TO_REASON.get(worst, "other")
