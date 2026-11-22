"""Turn real student choices into a fitted reward model.

This closes the loop the thesis is about. Offline, preference labels come
from `preference.simulate_choice` -- a logistic function of the true
return, which is a stand-in for a person. This reads what people actually
chose and fits the same Bradley-Terry model on it, so the headline result
can be restated as "RL from human feedback" rather than "from simulated
feedback".

Input is either the local JSONL the dev server writes, or a DynamoDB scan:

    aws dynamodb scan --table-name <T> --output json > study_events.json
    python -m pipeline.import_study_events study_events.json

With no argument it reads data/agent/feedback/*.jsonl.

The fit here deliberately reuses `preference.fit` unchanged. If real
preferences needed a different estimator than simulated ones, the offline
result would not have been evidence about the real thing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipeline import paths, preference
from pipeline.stage17_bestofn import DEFAULT_FEATURES


def _unwrap(v: Any) -> Any:
    """DynamoDB's typed JSON ({'S': 'x'}) -> plain Python."""
    if not isinstance(v, dict) or len(v) != 1:
        return v
    (t, inner), = v.items()
    if t == "S":
        return inner
    if t == "N":
        f = float(inner)
        return int(f) if f.is_integer() else f
    if t == "BOOL":
        return inner
    if t == "NULL":
        return None
    if t == "L":
        return [_unwrap(x) for x in inner]
    if t == "M":
        return {k: _unwrap(x) for k, x in inner.items()}
    return v


def load_events(scan_file: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """Events grouped by kind, from a DynamoDB scan or the local JSONL."""
    out: Dict[str, List[Dict]] = defaultdict(list)
    if scan_file:
        doc = json.loads(Path(scan_file).read_text(encoding="utf-8"))
        items = doc.get("Items", doc if isinstance(doc, list) else [])
        for raw in items:
            item = {k: _unwrap(v) for k, v in raw.items()}
            kind = item.get("kind")
            if kind:
                out[kind].append(item)
    else:
        for kind in ("plans_shown", "choice", "reason"):
            p = paths.AGENT / "feedback" / f"{kind}.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            out[kind].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return out


def build_real_pairs(events: Dict[str, List[Dict]]
                     ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Feature differences and the label a person actually gave.

    One row per choice: phi(chosen) - phi(rejected), label 1. Both
    orientations are emitted, chosen-minus-rejected labelled 1 and the
    reverse labelled 0, so the fit cannot pick up a sign convention as
    signal the way a one-sided set would.
    """
    by_plan: Dict[str, Dict[str, Any]] = {}
    for ev in events.get("plans_shown", []):
        for p in ev.get("plans", []):
            if p.get("plan_id") and p.get("features"):
                by_plan[p["plan_id"]] = {
                    "features": [float(x) for x in p["features"]],
                    "participant_id": ev.get("participant_id"),
                    "query": ev.get("query"),
                }

    X, y, meta, skipped = [], [], [], 0
    for ch in events.get("choice", []):
        chosen, left, right = (ch.get("chosen_plan_id"), ch.get("left_plan_id"),
                               ch.get("right_plan_id"))
        rejected = right if chosen == left else left
        if chosen not in by_plan or rejected not in by_plan:
            skipped += 1          # a choice whose plan set was never logged
            continue
        a = np.asarray(by_plan[chosen]["features"], float)
        b = np.asarray(by_plan[rejected]["features"], float)
        X.append(a - b); y.append(1)
        X.append(b - a); y.append(0)
        meta.append({
            "participant_id": ch.get("participant_id"),
            "chosen_plan_id": chosen, "rejected_plan_id": rejected,
            "seconds_on_screen": ch.get("seconds_on_screen"),
            # Which side the winner sat on. If choices track position
            # rather than content this is where it shows up.
            "chose_left": chosen == left,
        })
    if skipped:
        print(f"[import] {skipped} choice(s) skipped: no matching plan set",
              file=sys.stderr)
    return np.asarray(X, float), np.asarray(y, int), meta


def report(events: Dict[str, List[Dict]], out_path: Optional[Path] = None) -> str:
    shown = events.get("plans_shown", [])
    choices = events.get("choice", [])
    reasons = events.get("reason", [])
    participants = {e.get("participant_id") for e in choices} - {None}

    X, y, meta = build_real_pairs(events)
    lines = [
        "# Real study feedback", "",
        f"- participants: **{len(participants)}**",
        f"- plan sets generated: **{len(shown)}**",
        f"- pairwise choices: **{len(choices)}**",
        f"- reasons given: **{len(reasons)}**",
        f"- usable preference pairs: **{len(meta)}**", "",
    ]

    if meta:
        left = np.mean([m["chose_left"] for m in meta])
        lines += [
            "## Position bias", "",
            f"The left-hand plan was chosen **{left:.0%}** of the time. "
            "At 50% position is not driving choices; far from it and the "
            "labels are partly about layout rather than content.", "",
        ]
        secs = [m["seconds_on_screen"] for m in meta
                if isinstance(m["seconds_on_screen"], (int, float))]
        if secs:
            fast = np.mean([s < 3 for s in secs])
            lines += [
                f"Median time on screen **{np.median(secs):.1f}s**; "
                f"**{fast:.0%}** of choices came in under 3 seconds. A high "
                "share of very fast choices is worth a look — two five-course "
                "plans take longer than that to read.", "",
            ]

    if len(meta) >= 20:
        n = len(meta)
        cut = int(n * 0.7)
        tr = np.r_[np.arange(0, cut * 2)]
        te = np.r_[np.arange(cut * 2, n * 2)]
        theta = preference.fit(X[tr], y[tr])
        acc = preference.accuracy(theta, X[te], y[te])
        lines += [
            "## Fitted on real preferences", "",
            f"Bradley-Terry on {cut} training choices, "
            f"held-out accuracy **{acc:.3f}** on {n - cut}.", "",
            "| feature | weight |", "|---|---|",
        ]
        for name, w in zip(DEFAULT_FEATURES, theta):
            lines.append(f"| `{name}` | {w:+.3f} |")
        lines += [
            "",
            "Compare these against the simulated fit in "
            "`reports/bestofn_poc-v6-stratified.md`. Agreement on the sign "
            "and rough magnitude of `mean_too_hard` is the single most "
            "informative check: it is what the simulated learners were most "
            "sure about.", "",
        ]
    else:
        lines += [
            "## Not enough yet", "",
            f"**{len(meta)}** usable pairs. A Bradley-Terry fit over seven "
            "features needs on the order of a hundred before the weights "
            "mean anything; at five comparisons each that is roughly 20 "
            "participants. Collect more before reading anything into them.", "",
        ]

    text = "\n".join(lines)
    (out_path or (paths.REPORTS / "real_feedback.md")).write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan_file", nargs="?", type=Path,
                    help="DynamoDB scan JSON; omit to read the local JSONL")
    a = ap.parse_args()
    print(report(load_events(a.scan_file)))


if __name__ == "__main__":
    main()
