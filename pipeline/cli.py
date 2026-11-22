"""Pipeline runner.

    python -m pipeline.cli run              # all implemented stages
    python -m pipeline.cli run --from 3     # resume from a stage
    python -m pipeline.cli run --only 5
    python -m pipeline.cli status

Stages 6-11 (taxonomy, skills_assign, scope, prereqs, features, validate)
are declared here but not yet implemented; `status` shows what exists.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Dict, List, NamedTuple, Optional

from . import paths


class Stage(NamedTuple):
    number: int
    name: str
    runner: Optional[Callable[[], object]]
    summary: str


def _load(module: str) -> Callable[[], object]:
    def _run() -> object:
        mod = __import__(f"pipeline.{module}", fromlist=["run"])
        return mod.run()
    return _run


STAGES: List[Stage] = [
    Stage(1, "fetch", _load("stage01_fetch"), "download sources, record provenance"),
    Stage(2, "parse", _load("stage02_parse"), "detect encoding, repair mojibake"),
    Stage(3, "normalize", _load("stage03_normalize"), "map onto the canonical schema"),
    Stage(4, "merge", _load("stage04_merge"), "donor fill + field provenance"),
    Stage(5, "dedup", _load("stage05_dedup"), "three-band cascade + union-find"),
    Stage(6, "taxonomy", _load("stage06_taxonomy"), "extract, merge, prune, freeze"),
    Stage(7, "skills_assign", _load("stage07_skills"), "extract skills (deterministic pass)"),
    Stage(8, "scope", _load("stage08_scope"), "in/out of scope + attrition funnel"),
    Stage(9, "prereqs", _load("stage09_prereqs"), "agent-proposed skill DAG + study_plans"),
    Stage(10, "features", _load("stage10_features"), "freeze arm_index, SVD projection, matrices"),
    Stage(11, "validate", _load("stage11_validate"), "all assertions + manifest + datasheet"),
    Stage(12, "simulator", _load("stage12_simulator"), "synthetic learners (world/ oracle)"),
    Stage(13, "baseline", _load("stage13_baseline"), "content-based baseline policy"),
    Stage(14, "bandit", _load("stage14_bandit"), "LinUCB vs baseline"),
    Stage(15, "planner", _load("stage15_planner"), "MDP planner vs bandit vs baseline"),
    Stage(16, "experiment", _load("stage16_experiment"), "multi-seed paired experiment"),
    Stage(17, "bestofn", _load("stage17_bestofn"), "RL over LLM-generated study plans"),
]

BY_NUMBER: Dict[int, Stage] = {s.number: s for s in STAGES}


def cmd_run(args: argparse.Namespace) -> int:
    if args.only:
        selected = [BY_NUMBER[n] for n in args.only]
    else:
        start = args.start or 1
        selected = [s for s in STAGES if s.number >= start]
    selected = [s for s in selected if s.number <= (args.to or 99)]

    failed = 0
    for stage in selected:
        if stage.runner is None:
            print(f"\n=== stage {stage.number} {stage.name}: NOT IMPLEMENTED "
                  f"({stage.summary}) ===")
            continue
        print(f"\n=== stage {stage.number} {stage.name} ===")
        started = time.time()
        try:
            stage.runner()
        except SystemExit as exc:
            print(f"[cli] stage {stage.number} halted: {exc}", file=sys.stderr)
            failed += 1
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[cli] stage {stage.number} FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            failed += 1
            break
        print(f"[cli] stage {stage.number} done in {time.time() - started:.1f}s")
    return 1 if failed else 0


def cmd_status(_: argparse.Namespace) -> int:
    print(f"dataset_version : {paths.dataset_version()}")
    print(f"data root       : {paths.DATA}")
    print(f"sources enabled : {', '.join(paths.enabled_sources())}\n")
    print(f"{'#':>2}  {'stage':<14} {'impl':<6} artefacts")
    for stage in STAGES:
        impl = "yes" if stage.runner else "-"
        print(f"{stage.number:>2}  {stage.name:<14} {impl:<6} {stage.summary}")
    print("\nartefacts present:")
    for directory in (paths.STAGING, paths.CATALOG, paths.WORLD, paths.AGENT, paths.REPORTS):
        if not directory.exists():
            continue
        for item in sorted(directory.iterdir()):
            if item.is_file():
                print(f"  {paths.rel(item):<48} {item.stat().st_size:>10,} B")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run pipeline stages")
    run_p.add_argument("--from", dest="start", type=int, help="first stage")
    run_p.add_argument("--to", type=int, help="last stage")
    run_p.add_argument("--only", type=int, nargs="*", help="run exactly these stages")
    run_p.set_defaults(func=cmd_run)

    sub.add_parser("status", help="show stage and artefact status").set_defaults(func=cmd_status)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
