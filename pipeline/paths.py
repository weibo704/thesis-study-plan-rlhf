"""Canonical paths and the frozen dataset version.

Everything downstream resolves locations through this module so that the
three-namespace layout stays enforceable by inspection.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Load repo-local .env before anything reads os.environ. override=False so
# a variable already exported in the shell wins over the file.
load_dotenv(ROOT / ".env", override=False)

# THESIS_DATA_ROOT lets tests and fixture runs redirect all generated
# artefacts to a scratch directory. Config is always read from the repo.
_DATA_ROOT = Path(os.environ.get("THESIS_DATA_ROOT", ROOT)).resolve()

CONFIG = ROOT / "config"
DATA = _DATA_ROOT / "data"
RAW = DATA / "raw"
STAGING = DATA / "staging"
CATALOG = DATA / "catalog"
WORLD = DATA / "world"
AGENT = DATA / "agent"
REPORTS = _DATA_ROOT / "reports"

SOURCES_YAML = CONFIG / "sources.yaml"
FAMILIES_YAML = CONFIG / "families.yaml"

# Namespaces the firewall assertion cares about. agent/ must never take an
# input from world/ -- see docs/ingestion-plan.md section 2.
FIREWALL_FORBIDDEN = {"agent": {"world"}}


@functools.lru_cache(maxsize=1)
def load_sources() -> Dict[str, Any]:
    with open(SOURCES_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def load_families() -> Dict[str, Any]:
    with open(FAMILIES_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dataset_version() -> str:
    return str(load_sources()["dataset_version"])


def enabled_sources() -> Dict[str, Dict[str, Any]]:
    """Sources that participate in the build (Udemy is disabled by default)."""
    return {
        sid: spec
        for sid, spec in load_sources()["sources"].items()
        if spec.get("enabled", True)
    }


def raw_dir(source_id: str) -> Path:
    return RAW / source_id


def raw_file(source_id: str) -> Path:
    spec = load_sources()["sources"][source_id]
    return raw_dir(source_id) / spec["filename"]


def staging_file(source_id: str) -> Path:
    return STAGING / f"{source_id}.parquet"


def ensure_dirs() -> None:
    for d in (RAW, STAGING, CATALOG, WORLD, AGENT, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    """Path relative to the repo root when possible, else to the data root.

    Manifest keys must stay stable whether the build writes into the repo or
    into a scratch root, so this never raises.
    """
    for base in (ROOT, _DATA_ROOT):
        try:
            return str(Path(path).resolve().relative_to(base))
        except ValueError:
            continue
    return str(Path(path).resolve())


def llm_api_key() -> str:
    """Return the OpenAI key, or an empty string if unset.

    Never log, print, or write this value into an artefact.
    """
    return os.environ.get("OPENAI_API_KEY", "").strip()


def has_llm() -> bool:
    return bool(llm_api_key())


# --- namespace firewall -------------------------------------------------
# agent/ artefacts must never derive from world/. Enforced by recording
# every parquet read and asserting on the actual paths touched, rather than
# grepping source text -- a grep cannot tell a reference from a read, and
# trips over the assertion that implements it.
_READS: list = []


def load_parquet(path: Path):
    """Read a parquet file, recording the path for the firewall check."""
    import pandas as pd
    _READS.append(Path(path).resolve())
    return pd.read_parquet(path)


def reset_read_log() -> None:
    _READS.clear()


# The firewall bans the ORACLE'S STRUCTURE, not the user's own context.
#
# A contextual bandit receives a context describing the learner by
# definition -- in a real system the user types their goal, states their
# level and says how many hours they have. That is the policy's input, not
# a leak.
#
# What must stay hidden is the oracle's MODEL of how skills relate, and the
# reward itself. If the policy can see prerequisite coverage while the
# reward credits prerequisite satisfaction, the bandit wins by construction
# and the comparison measures nothing. Prerequisite satisfaction has to be
# a latent property the policy discovers from observable features.
ORACLE_FILES = ("skill_prereqs", "study_plans", "study_plan_items", "reward_specs")

# Columns of world/users.parquet that a real recommender could never see.
ORACLE_USER_COLUMNS = ("readiness", "persistence", "novelty_pref",
                       "price_sensitivity", "prereq_gap", "seed")

OBSERVABLE_USER_COLUMNS = ("user_id", "start_level", "weekly_hours",
                           "goal_labels", "goal_family", "background",
                           "motivation")


def assert_no_world_reads(stage: str) -> None:
    """Strict form: no world/ reads at all. For build stages like features."""
    world = WORLD.resolve()
    leaked = [p for p in _READS if world in p.parents or p == world]
    if leaked:
        raise AssertionError(
            f"FIREWALL VIOLATION in {stage}: read from world/ -> "
            f"{[str(p.name) for p in leaked]}"
        )


def assert_no_oracle_reads(stage: str) -> None:
    """Policy form: user context is allowed, oracle structure is not."""
    leaked = [p for p in _READS
              if any(name in p.name or name in str(p.parent.name)
                     for name in ORACLE_FILES)]
    if leaked:
        raise AssertionError(
            f"FIREWALL VIOLATION in {stage}: a policy read oracle structure -> "
            f"{sorted({p.name for p in leaked})}. Prerequisite structure must "
            f"stay latent; seeing it while the reward credits it makes the "
            f"comparison vacuous."
        )


def read_log() -> list:
    return list(_READS)
