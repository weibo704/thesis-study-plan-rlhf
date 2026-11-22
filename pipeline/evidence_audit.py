"""Check every headline number in the thesis against its source.

The advisor's first request: resolve the numerical differences and say
which report supports each result. This recomputes the claims from the
committed parquet and report files, compares them to what the draft
actually says, and lists every disagreement.

A claim is one of:
  data   recomputed from parquet under data/
  report parsed out of a generated report under reports/
Each carries the file a reader should check.

    python -m pipeline.evidence_audit

Writes reports/evidence_audit.md and exits non-zero if anything mismatches.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline import paths

DRAFT = Path(__file__).resolve().parent.parent / "docs" / "thesis-draft.md"
PREREG = "poc-v6-stratified"


def _norm(s: str) -> str:
    return (str(s).replace("\u2212", "-").replace("\u2013", "-")
                  .replace("\u2014", "-").replace("\u00b1", "+/-"))


class Claim:
    def __init__(self, section: str, text: str, value: Any, source: str,
                 compute: Optional[Callable[[], Any]] = None, tol: float = 5e-4):
        self.section, self.text, self.value = section, text, value
        self.source, self.compute, self.tol = source, compute, tol
        self.actual: Any = None
        self.status = "unchecked"
        self.note = ""

    def check(self, draft: str) -> None:
        # 1. is the number actually present in the draft, as written?
        #    Typographic minus (U+2212) and en-dash are normalised to ASCII
        #    first; the draft uses the typographic form in prose and the
        #    reports use ASCII, and that difference is not a discrepancy.
        in_draft = _norm(self.text) in _norm(draft)
        # 2. does the source still produce it?
        if self.compute is not None:
            try:
                self.actual = self.compute()
            except Exception as e:                       # noqa: BLE001
                self.status = "ERROR"; self.note = f"{type(e).__name__}: {e}"
                return
            if isinstance(self.value, float):
                ok_val = abs(float(self.actual) - self.value) <= self.tol
            elif isinstance(self.value, str):
                ok_val = _norm(self.actual) == _norm(self.value)
            else:
                ok_val = self.actual == self.value
        else:
            ok_val, self.actual = True, "(not recomputed)"

        if not in_draft and not ok_val:
            self.status = "MISMATCH"; self.note = "absent from draft AND source disagrees"
        elif not in_draft:
            self.status = "MISSING"; self.note = "source agrees, but this string is not in the draft"
        elif not ok_val:
            self.status = "MISMATCH"; self.note = f"draft says {self.value}, source gives {self.actual}"
        else:
            self.status = "ok"


def _sel() -> pd.DataFrame:
    return pd.read_parquet(paths.AGENT / f"bestofn_selection_{PREREG}.parquet")


def _piv(metric="return") -> pd.DataFrame:
    return _sel().pivot(index="user_id", columns="policy", values=metric)


def _paired(a: str, b: str, metric="return"):
    p = _piv(metric)
    return (p[a] - p[b]).dropna()


def _exp() -> pd.DataFrame:
    return pd.read_parquet(paths.AGENT / f"experiment_{PREREG}.parquet")


def _stage16(a: str, b: str):
    """Per (learner, seed) — the pairing the stage-16 report uses."""
    ex = _exp()
    tot = ex.groupby(["policy", "user_id", "seed"])["reward"].sum().unstack(0)
    return (tot[a] - tot[b]).dropna()


def _report(name: str) -> str:
    p = paths.REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def claims() -> List[Claim]:
    import json
    rm = json.loads((paths.AGENT / f"reward_model_{PREREG}.json").read_text())
    C = []

    # ---- item space -------------------------------------------------
    C.append(Claim("3.1/3.2", "1,899", 1899, "data/agent/arm_features.parquet",
                   lambda: len(pd.read_parquet(paths.AGENT / "arm_features.parquet"))))
    C.append(Claim("3.3", "188", 188, "data/catalog/skills.parquet",
                   lambda: int((pd.read_parquet(paths.CATALOG / "skills.parquet")
                                .pipe(lambda d: d[d.tier == "skill"] if "tier" in d else d)).shape[0])))
    C.append(Claim("3.3", "254", 254, "data/world/skill_prereqs.parquet",
                   lambda: len(pd.read_parquet(paths.WORLD / "skill_prereqs.parquet"))))
    C.append(Claim("3.3", "122", 122, "data/world/study_plans.parquet",
                   lambda: len(pd.read_parquet(paths.WORLD / "study_plans.parquet"))))
    C.append(Claim("3.4", "500", 500, "data/world/users.parquet",
                   lambda: len(pd.read_parquet(paths.WORLD / "users.parquet"))))

    # ---- stage 17 policy means --------------------------------------
    for name, pol, val in [("the LLM's first plan", "sample_1", 0.5985),
                           ("random of six", "random_of_n", 0.4045),
                           ("feedback model", "best_of_n_preference", 0.7121),
                           ("oracle-regressed", "best_of_n_regression", 0.6819),
                           ("ceiling", "best_of_n_oracle", 0.8438)]:
        C.append(Claim("4.4", f"{val:.4f}", val,
                       f"data/agent/bestofn_selection_{PREREG}.parquet",
                       lambda p=pol: round(float(_piv()[p].mean()), 4)))

    # ---- stage 17 contrasts -----------------------------------------
    for lab, a, b, mean, ci in [
            ("feedback RL over raw LLM", "best_of_n_preference", "sample_1", 0.1136, 0.0745),
            ("scoring over sampling", "best_of_n_preference", "random_of_n", 0.3075, 0.0995),
            ("more candidates alone", "random_of_n", "sample_1", -0.1940, 0.1001),
            ("headroom remaining", "best_of_n_oracle", "best_of_n_preference", 0.1317, 0.0465)]:
        C.append(Claim("4.4", f"{mean:+.4f}", mean, f"reports/bestofn_{PREREG}.md",
                       lambda a=a, b=b: round(float(_paired(a, b).mean()), 4)))
        C.append(Claim("4.4", f"±{ci:.4f}", ci, f"reports/bestofn_{PREREG}.md",
                       lambda a=a, b=b: round(float(1.96 * _paired(a, b).std(ddof=1)
                                                    / np.sqrt(len(_paired(a, b)))), 4)))

    # ---- the win-rate decomposition ---------------------------------
    d = _paired("best_of_n_preference", "sample_1")
    C.append(Claim("4.4", "24%", 24, "data/agent/bestofn_selection parquet",
                   lambda: int(round(float((d == 0).mean()) * 100))))
    C.append(Claim("4.4", "67%", 67, "data/agent/bestofn_selection parquet",
                   lambda: int(round(float((d[d != 0] > 0).mean()) * 100))))
    C.append(Claim("4.4", "+0.1489", 0.1489, "data/agent/bestofn_selection parquet",
                   lambda: round(float(d[d != 0].mean()), 4)))

    # ---- learned weights --------------------------------------------
    for name, val in zip(rm["feature_names"], rm["theta"]):
        pretty = f"{val:+.3f}"
        C.append(Claim("4.5", pretty, round(val, 3),
                       f"data/agent/reward_model_{PREREG}.json",
                       lambda v=val: round(float(v), 3)))
    hard = dict(zip(rm["feature_names"], rm["theta"]))
    C.append(Claim("4.5", "4.6", 4.6, "ratio of the two difficulty weights",
                   lambda: round(abs(hard["mean_too_hard"] / hard["mean_too_easy"]), 1),
                   tol=0.05))
    C.append(Claim("4.3", "0.602", 0.602, f"data/agent/reward_model_{PREREG}.json",
                   lambda: round(float(rm["heldout_pair_accuracy"]), 3)))
    C.append(Claim("3.8", "64%", 64, f"data/agent/reward_model_{PREREG}.json",
                   lambda: int(round(float(rm["label_noise"]) * 100))))

    # ---- stage 16 ----------------------------------------------------
    for lab, a, b, mean, ci, win in [
            ("learning", "linucb", "baseline", 0.2042, 0.0130, 80),
            ("planning", "planner", "linucb", 0.0640, 0.0206, 53)]:
        C.append(Claim("4.7", f"{mean:+.4f}", mean, f"reports/experiment_{PREREG}.md",
                       lambda a=a, b=b: round(float(_stage16(a, b).mean()), 4)))
        C.append(Claim("4.7", f"±{ci:.4f}", ci, f"reports/experiment_{PREREG}.md",
                       lambda a=a, b=b: round(float(
                           1.96 * _stage16(a, b).std(ddof=1) / np.sqrt(len(_stage16(a, b)))), 4)))
        C.append(Claim("4.7", f"{win}%", win, f"reports/experiment_{PREREG}.md",
                       lambda a=a, b=b: int(round(float((_stage16(a, b) > 0).mean()) * 100))))

    # ---- numbers that live only in a report --------------------------
    def from_report(fname: str, pattern: str):
        def f():
            m = re.search(pattern, _report(fname))
            return m.group(1) if m else None
        return f

    for sec, txt, fname, pat in [
            ("4.2", "+1.9828", "stage1_evidence.md", r"retrieval \|[^|]*\|\s*([+-][\d.]+)"),
            ("4.2", "+0.3707", "stage1_evidence.md", r"generation \|[^|]*\|\s*([+-][\d.]+)"),
            ("4.2", "+0.2872", "stage1_evidence.md", r"ordering \|[^|]*\|\s*([+-][\d.]+)"),
            ("4.8", "+0.0549", "ablation_randomized_graph.md", r"`linucb` \| real − random \| ([+-][\d.]+)"),
            ("4.8", "+0.0552", "ablation_randomized_graph.md", r"`planner` \| real − random \| ([+-][\d.]+)"),
            ("4.8", "+0.0347", "ablation_randomized_graph.md", r"`baseline` \| real − random \| ([+-][\d.]+)")]:
        C.append(Claim(sec, txt, txt, f"reports/{fname}", from_report(fname, pat)))

    C.append(Claim("4.2", "0.255", "0.255", "reports/stage1_evidence.md",
                   from_report("stage1_evidence.md", r"within one learner[^|]*\|\s*([\d.]+)")))
    C.append(Claim("4.2", "0.035", "0.035", "reports/stage1_evidence.md",
                   from_report("stage1_evidence.md", r"across two learners[^|]*\|\s*([\d.]+)")))
    return C


def cross_report() -> List[Dict[str, str]]:
    """Quantities that two generated reports both state, and disagree on.

    These are the ones that embarrass a thesis: both files are correct
    for what they computed, and a reader who opens both sees a
    contradiction nobody flagged.
    """
    issues = []
    stat, exp = _report("statistical_analysis.md"), _report(f"experiment_{PREREG}.md")
    if stat and exp:
        for lab, pat_s, pat_e in [
                ("learning win rate",
                 r"learning \(bandit over fixed scorer\) \| \d+ \|[^|]*\|[^|]*\|[^|]*\|\s*(\d+)%",
                 r"\| learning \|[^|]*\|[^|]*\|\s*(\d+)%"),
                ("planning win rate",
                 r"planning \(lookahead over bandit\) \| \d+ \|[^|]*\|[^|]*\|[^|]*\|\s*(\d+)%",
                 r"\| planning \|[^|]*\|[^|]*\|\s*(\d+)%")]:
            a = re.search(pat_s, stat); b = re.search(pat_e, exp)
            if a and b and a.group(1) != b.group(1):
                issues.append({
                    "quantity": lab,
                    "statistical_analysis.md": a.group(1) + "%",
                    f"experiment_{PREREG}.md": b.group(1) + "%",
                    "why": "paired per learner (n=500) vs per learner-seed (n=2,500); "
                           "means agree, dispersion does not",
                    "thesis uses": b.group(1) + "% (per learner-seed)"})
    return issues


def run() -> int:
    draft = DRAFT.read_text(encoding="utf-8")
    cs = claims()
    for c in cs:
        c.check(draft)
    bad = [c for c in cs if c.status != "ok"]

    out = ["# Evidence audit", "",
           "`python -m pipeline.evidence_audit` · every headline number in the "
           "draft, recomputed from its source and compared to what the draft "
           "says.", "",
           f"**{len(cs) - len(bad)} of {len(cs)} claims verified.**", ""]
    xr = cross_report()
    if xr:
        out += ["## Quantities two reports disagree on", "",
                "Both files are correct for what they computed. The risk is a "
                "reader opening both and finding a contradiction nobody "
                "flagged, so each is reconciled here and the thesis cites one "
                "consistently.", "",
                "| quantity | statistical_analysis.md | experiment report | why they differ | thesis uses |",
                "|---|---|---|---|---|"]
        for i in xr:
            out.append(f"| {i['quantity']} | {i['statistical_analysis.md']} | "
                       f"{i[f'experiment_{PREREG}.md']} | {i['why']} | {i['thesis uses']} |")
        out.append("")

    if bad:
        out += ["## Discrepancies", "",
                "| § | draft says | source gives | source | note |", "|---|---|---|---|---|"]
        for c in bad:
            out.append(f"| {c.section} | `{c.text}` | `{c.actual}` | `{c.source}` | "
                       f"{c.status}: {c.note} |")
        out.append("")
    out += ["## Verified claims and the file that supports each", "",
            "| § | claim | source |", "|---|---|---|"]
    for c in cs:
        mark = "" if c.status == "ok" else " ⚠"
        out.append(f"| {c.section} | `{c.text}`{mark} | `{c.source}` |")
    out.append("")
    (paths.REPORTS / "evidence_audit.md").write_text("\n".join(out), encoding="utf-8")

    print(f"{len(cs) - len(bad)}/{len(cs)} verified; "
          f"{len(xr)} cross-report disagreement(s) reconciled")
    for c in bad:
        print(f"  [{c.status}] §{c.section} '{c.text}' -> {c.actual}  ({c.note})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
