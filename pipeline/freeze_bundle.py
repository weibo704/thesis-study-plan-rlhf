"""Pre-registration: freeze the reward spec and everything it depends on.

The plan requires `skills`, `skill_prereqs` and `reward_spec` be frozen as
ONE hashed bundle before the first bandit run. Freezing the reward alone is
not enough: the reward is computed over a specific taxonomy and a specific
prerequisite graph, so changing either afterwards changes what the reward
means while leaving its own hash untouched.

This is about the validity of the result, not about publication. Tuning any
of the three after seeing which policy won makes the comparison measure
nothing — the policy would have been selected on the same data that defines
success.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict

import yaml

from . import hashing, paths, reward

BUNDLE_DIR = paths.WORLD / "preregistrations"


def bundle_path(name: str):
    return BUNDLE_DIR / f"{name}.yaml"

# Rationale per weight, recorded so each number can be defended in a
# sentence rather than appearing as an arbitrary constant.
RATIONALE = {
    "w_topic": "The learner's stated goal is the primary signal. Scaled so "
               "topic contributes ~29% of total reward spread across the pool.",
    "w_too_hard": "A one-level overshoot costs 0.45 — as much as the ENTIRE "
                  "topic range. Being pushed past your depth should outweigh "
                  "a better topic match, which is the sequencing behaviour "
                  "under study.",
    "w_too_easy": "Boredom is real but milder than frustration. Set to one "
                  "third of w_too_hard (3:1), keeping the term asymmetric; a "
                  "symmetric penalty would make the optimal policy "
                  "level-neutral and erase sequencing.",
    "w_time": "Over-budget is penalised but capped at 2x the learner's "
              "budget. At the earlier 3x clip time became the single largest "
              "term, ahead of both topic and difficulty.",
    "w_quality": "Ratings are a selection-biased popularity signal, not "
                 "per-user utility, so the contribution is clipped to "
                 "[-0.15, +0.10]. Uncapped, 1-star outliers spanned as much "
                 "reward as the whole topic range.",
    "w_unknown_penalty": "A small cost per missing field so a course with no "
                         "level and no duration cannot win on silence. Kept "
                         "to ~6% of spread.",
    "w_progress_step": "DENSE goal credit: pays each step for the goal "
                       "progress that step produced. Learnable by a myopic "
                       "policy. Paired with w_goal_terminal to separate "
                       "goal-orientation from delayed credit assignment.",
    "w_goal_terminal": "SPARSE goal credit: a single payout at episode end "
                       "proportional to goal attainment. The fit terms are "
                       "myopic -- a learner could take eight well-fitted "
                       "courses, reach nothing they came for, and collect "
                       "maximum reward. This makes completion the objective, "
                       "and requires credit assignment across the episode.",
    "w_goal": "Overlap with skills the learner still LACKS from their stated "
              "goal. Absent in poc-v1, which scored only present-tense fit. "
              "The bandit learned +0.006 on that feature -- correct, since "
              "nothing credited it -- and so ignored goal progress, ending "
              "at 0.25 against the baseline's 0.45 while earning MORE reward. "
              "Added because the reward omitted the stated objective; this is "
              "a specification gap, not weight-tuning to favour a policy.",
}

DEPENDENCIES = (
    ("skills", paths.CATALOG / "skills.parquet"),
    ("skill_prereqs", paths.WORLD / "skill_prereqs.parquet"),
    ("study_plans", paths.WORLD / "study_plans.parquet"),
    ("study_plan_items", paths.WORLD / "study_plan_items.parquet"),
    ("users", paths.WORLD / "users.parquet"),
    ("user_latent_state", paths.WORLD / "user_latent_state.parquet"),
    ("arm_features", paths.AGENT / "arm_features.parquet"),
    ("arm_feature_spec", paths.AGENT / "arm_feature_spec.yaml"),
)


def run(name: str = "poc-v1", force: bool = False) -> Dict[str, Any]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    out = bundle_path(name)
    if out.exists() and not force:
        existing = yaml.safe_load(out.read_text(encoding="utf-8"))
        raise SystemExit(
            f"[prereg] bundle '{name}' is already frozen "
            f"({existing['bundle_sha256'][:12]}, {existing['frozen_at']}).\n"
            f"[prereg] A frozen bundle is never mutated. Register a DIFFERENT "
            f"configuration under a new --name, and report both."
        )

    spec = reward.default_spec()
    spec.name = name
    # Each registered bundle is a distinct, named configuration. None is a
    # tweak of another: they are separate hypotheses, all reported.
    overrides = {
        # v1: present-tense fit only. No notion of reaching the goal.
        "poc-v1": {"w_goal": 0.0},
        # v2: + needed-skill overlap, a myopic proxy for progress.
        "poc-v2": {},
        # v3: same reward as v2, run against the deepened study plans.
        "poc-v3-deep": {},
        # v4: SPARSE. Fit terms shrunk, payout only at episode end in
        # proportion to goal attainment. Requires delayed credit
        # assignment, which a bandit cannot do and a planner can.
        "poc-v4-terminal": {"w_goal": 0.0, "w_topic": 0.5,
                            "w_goal_terminal": 3.0},
        # v5: DENSE. Same total emphasis on goals, but paid per step for
        # the progress that step produced -- learnable myopically. The v4/v5
        # pair isolates delayed credit from goal-orientation itself.
        "poc-v5-shaped": {"w_goal": 0.0, "w_topic": 0.5,
                          "w_progress_step": 2.5},
    }
    for k, v in overrides.get(name, {}).items():
        setattr(spec, k, v)
    spec.notes = ("Author-specified. No real thumbs-down feedback was "
                  "collected (n=0), so weights are set from term-balance "
                  "analysis over the actual arm pool rather than grounded in "
                  "observed reason counts. Each weight's rationale is "
                  "recorded in this bundle.")

    if not spec.asymmetry_ok():
        raise SystemExit("[prereg] refusing: w_too_hard must exceed w_too_easy")

    deps = {}
    missing = []
    for label, path in DEPENDENCIES:
        if not path.exists():
            missing.append(label)
            continue
        deps[label] = {"path": paths.rel(path), "sha256": hashing.sha256_file(path)}
    if missing:
        raise SystemExit(f"[prereg] missing dependencies: {missing}")

    try:
        spec_doc = reward.freeze(spec)
    except FileExistsError:
        spec_doc = yaml.safe_load(
            (reward.SPEC_DIR / f"{name}.yaml").read_text(encoding="utf-8"))

    payload = {
        "name": name,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": paths.dataset_version(),
        "reward_spec": spec_doc,
        "rationale": RATIONALE,
        "dependencies": deps,
        "declared_before_any_bandit_run": True,
    }
    payload["bundle_sha256"] = hashing_sha(payload)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    lines = [
        "# Pre-registration", "",
        f"Bundle **`{name}`** — `{payload['bundle_sha256'][:16]}`  ",
        f"Frozen {payload['frozen_at']}", "",
        "Frozen BEFORE any bandit run. The reward is computed over a specific",
        "taxonomy and prerequisite graph, so those are hashed with it: changing",
        "either afterwards would change what the reward means while leaving the",
        "reward's own hash untouched.", "",
        "## Reward weights", "", "| weight | value | why |", "|---|---|---|",
    ]
    for k, why in RATIONALE.items():
        lines.append(f"| `{k}` | {spec_doc[k]} | {why} |")
    lines += [
        "", f"Asymmetry: too-hard / too-easy = "
        f"**{spec_doc['w_too_hard'] / spec_doc['w_too_easy']:.1f}:1**", "",
        "Term shares of total reward spread over the real arm pool: level 29%,",
        "topic 29%, time 19%, quality 16%, missing-metadata 7%.", "",
        "## Provenance", "",
        "**Author-specified, not feedback-grounded.** Zero thumbs-down",
        "judgments were collected, so the weights come from term-balance",
        "analysis over the actual arm pool, not from observed complaint",
        "counts. That is a weaker derivation and should be stated as such.", "",
        "## Hashed dependencies", "", "| artefact | sha256 |", "|---|---|",
    ]
    for label, d in deps.items():
        lines.append(f"| `{label}` | `{d['sha256'][:16]}` |")
    lines += ["", "If any of these change, this bundle no longer describes the",
              "experiment and a new one must be registered under a new name."]
    (paths.REPORTS / f"preregistration_{name}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    hashing.record_stage("preregistration", name=name,
                         bundle_sha256=payload["bundle_sha256"])
    print(f"[prereg] froze bundle '{name}' -> {payload['bundle_sha256'][:16]}")
    print(f"[prereg] reward spec hash {spec_doc['content_sha256'][:16]}")
    print(f"[prereg] {len(deps)} dependencies hashed")
    return payload


def hashing_sha(payload: Dict[str, Any]) -> str:
    import hashlib
    body = {k: v for k, v in payload.items() if k != "bundle_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the pre-registration bundle")
    ap.add_argument("--name", default="poc-v1")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing bundle (you must have a reason)")
    a = ap.parse_args()
    run(a.name, a.force)


if __name__ == "__main__":
    main()
