"""One consolidated statistical account of every contrast in the thesis.

The per-stage reports each give means, intervals and win rates for their
own experiment. What none of them gives is a single table with effect
sizes alongside, and effect size is the quantity that separates "this
cleared significance because n was 2,500" from "this matters".

Every contrast here is PAIRED: each learner is evaluated under every
policy, so the unit of analysis is the within-learner difference. That
removes variance from learners differing in goal depth, starting level
and hour budget, which is most of the variance there is.

Reads only committed parquet under data/agent/. No API calls.

    python -m pipeline.statistical_analysis
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline import paths

PREREG = "poc-v6-stratified"


def paired(a: pd.Series, b: pd.Series, label: str, contrast: str) -> Dict:
    """Paired contrast with interval, effect size and win rate.

    Cohen's d here is the paired form: mean difference over the standard
    deviation OF THE DIFFERENCES, not of the raw scores. Using the raw
    spread would understate the effect, because the pairing is exactly
    what removes between-learner variance.
    """
    d = (a - b).dropna()
    n = len(d)
    if n < 2:
        return {}
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    ci = 1.96 * sd / np.sqrt(n)
    # Ties are real: a selector that picks the plan the LLM already put
    # first produces an exact zero. Counting those as losses understates
    # the win rate, so both are reported.
    ties = int((d == 0).sum())
    nz = d[d != 0]
    return {
        "label": label, "contrast": contrast, "n": n,
        "mean": mean, "ci95": ci, "sd": sd,
        "d": mean / sd if sd > 0 else float("nan"),
        "win": float((d > 0).mean()),
        "win_excl_ties": float((nz > 0).mean()) if len(nz) else float("nan"),
        "ties": ties,
        "established": abs(mean) > ci,
    }


def _fmt(r: Dict) -> str:
    mag = abs(r["d"])
    size = ("negligible" if mag < 0.2 else "small" if mag < 0.5
            else "medium" if mag < 0.8 else "large")
    tie = f"{r['ties']}" if r["ties"] else "—"
    return (f"| {r['contrast']} | {r['n']} | {r['mean']:+.4f} | ±{r['ci95']:.4f} | "
            f"{r['d']:+.3f} ({size}) | {r['win']:.0%} | {tie} | "
            f"{'**established**' if r['established'] else 'not established'} |")


HEAD = ("| contrast | n | difference | 95% CI | Cohen's d | win rate | ties | verdict |\n"
        "|---|---|---|---|---|---|---|---|")


def run(prereg: str = PREREG) -> str:
    out: List[str] = [
        f"# Consolidated statistical analysis — `{prereg}`", "",
        "`python -m pipeline.statistical_analysis` · computed from the parquet "
        "under `data/agent/` · no API calls.", "",
        "Every contrast is paired within learner. Cohen's *d* is the paired "
        "form — mean difference over the standard deviation of the "
        "differences — because the pairing is what removes between-learner "
        "variance. A 95% interval spanning zero is *not established* "
        "regardless of sign.", "",
        "**Why effect size is reported alongside the interval.** With 2,500 "
        "learner-runs almost any non-zero effect clears significance. The "
        "interval says an effect is real; *d* says whether it matters.", "",
    ]

    # ---------------- stage 17: selection policies ----------------
    sel_path = paths.AGENT / f"bestofn_selection_{prereg}.parquet"
    if sel_path.exists():
        sel = pd.read_parquet(sel_path)
        piv = sel.pivot(index="user_id", columns="policy", values="return")
        prog = sel.pivot(index="user_id", columns="policy", values="progress")
        out += ["## 1. Best-of-N selection (the headline experiment)", "",
                f"{piv.shape[0]} held-out learners, six candidate plans each, "
                "identical candidates across policies.", "",
                "### Mean return by policy", "",
                "| policy | mean return | mean goal progress |", "|---|---|---|"]
        order = ["sample_1", "random_of_n", "best_of_n_preference",
                 "best_of_n_regression", "best_of_n_oracle"]
        names = {"sample_1": "the LLM's first plan",
                 "random_of_n": "a random one of its six",
                 "best_of_n_preference": "**chosen by the feedback model**",
                 "best_of_n_regression": "chosen by an oracle-regressed model",
                 "best_of_n_oracle": "the best of the six (ceiling)"}
        for p in order:
            if p in piv:
                out.append(f"| {names[p]} | {piv[p].mean():.4f} | {prog[p].mean():.4f} |")
        out += ["", "### Paired contrasts on return", "", HEAD]
        pairs = [("feedback RL over raw LLM", "best_of_n_preference", "sample_1"),
                 ("scoring over sampling", "best_of_n_preference", "random_of_n"),
                 ("more candidates alone", "random_of_n", "sample_1"),
                 ("feedback vs oracle-regressed", "best_of_n_preference", "best_of_n_regression"),
                 ("headroom remaining", "best_of_n_oracle", "best_of_n_preference")]
        head_r = None
        for lab, a, b in pairs:
            if a in piv and b in piv:
                r = paired(piv[a], piv[b], "bestofn", lab)
                if lab == "feedback RL over raw LLM":
                    head_r = r
                out.append(_fmt(r))

        if head_r:
            out += ["", "### The headline contrast, decomposed", "",
                    "A selector that picks the plan the language model already "
                    "placed first produces a difference of exactly zero, counted "
                    "as a non-win. Those ties are the model *agreeing*, not "
                    "failing, and they depress the headline win rate.", "",
                    "| quantity | value |", "|---|---|",
                    f"| learners | {head_r['n']} |",
                    f"| ties (model picked the LLM's own first plan) | "
                    f"{head_r['ties']} ({head_r['ties']/head_r['n']:.0%}) |",
                    f"| win rate, all learners | {head_r['win']:.0%} |",
                    f"| **win rate where it overrode the LLM** | "
                    f"**{head_r['win_excl_ties']:.0%}** |"]
            d = (piv["best_of_n_preference"] - piv["sample_1"]).dropna()
            nz = d[d != 0]
            if len(nz):
                out += [f"| mean gain when it overrode | {nz.mean():+.4f} |",
                        f"| mean when it won | {nz[nz>0].mean():+.4f} |",
                        f"| mean when it lost | {nz[nz<0].mean():+.4f} |"]
            out.append("")

    # ---------------- stage 16: baseline family ----------------
    exp_path = paths.AGENT / f"experiment_{prereg}.parquet"
    if exp_path.exists():
        ex = pd.read_parquet(exp_path)
        tot = (ex.groupby(["policy", "user_id", "seed"])["reward"].sum()
                 .reset_index().groupby(["policy", "user_id"])["reward"].mean()
                 .unstack(0))
        fin = (ex.sort_values("t").groupby(["policy", "user_id", "seed"])["progress"]
                 .last().reset_index()
                 .groupby(["policy", "user_id"])["progress"].mean().unstack(0))
        seeds = [int(x) for x in sorted(ex["seed"].unique())]
        out += ["## 2. Baseline policy family", "",
                f"{tot.shape[0]} learners × {len(seeds)} seeds "
                f"({seeds}) = {tot.shape[0]*len(seeds):,} learner-runs per policy.", "",
                "### Paired contrasts on total reward", "",
                "> **These differ from `reports/experiment_" + prereg + ".md`, and "
                "deliberately.** That report pairs at the (learner, seed) level, "
                "giving n = 2,500 and treating the same learner under a different "
                "seed as an independent observation. Here each learner's reward is "
                "averaged over seeds first, giving n = 500 — one row per learner, "
                "which is what a learner actually is. Averaging removes seed noise, "
                "so the intervals widen (fewer rows) while the win rates rise "
                "(less noise to flip a sign). The means are identical either way. "
                "Cite whichever you prefer, but cite one of them consistently; the "
                "thesis currently quotes the per-seed figures.", "", HEAD]
        for lab, a, b in [("learning (bandit over fixed scorer)", "linucb", "baseline"),
                          ("planning (lookahead over bandit)", "planner", "linucb"),
                          ("learning + planning", "planner", "baseline")]:
            if a in tot and b in tot:
                out.append(_fmt(paired(tot[a], tot[b], "stage16", lab)))
        out += ["", "### The same contrasts on goal progress", "",
                "Reported because it is a metric the policies do **not** "
                "optimise, which is what makes it informative: every policy "
                "that earns more reward reaches less of the learner's goal.",
                "", HEAD]
        for lab, a, b in [("learning (bandit over fixed scorer)", "linucb", "baseline"),
                          ("planning (lookahead over bandit)", "planner", "linucb"),
                          ("learning + planning", "planner", "baseline")]:
            if a in fin and b in fin:
                out.append(_fmt(paired(fin[a], fin[b], "stage16-progress", lab)))
        out += ["", "### Per-seed means, to show the result is not one lucky draw", "",
                "| seed | " + " | ".join(sorted(ex.policy.unique())) + " |",
                "|---" * (1 + ex.policy.nunique()) + "|"]
        per = ex.groupby(["seed", "policy", "user_id"])["reward"].sum().groupby(
            ["seed", "policy"]).mean().unstack(1)
        for s in per.index:
            out.append(f"| {s} | " + " | ".join(f"{per.loc[s, p]:.4f}"
                                                for p in sorted(ex.policy.unique())) + " |")
        out.append("")

    out += ["## 3. How to read a 'not established' row", "",
            "It means the interval spans zero, so the sign is not reliable at "
            "this sample size. It does **not** mean the effect is zero. §4.6 of "
            "the thesis shows the converse case: several selector variants "
            "differ by amounts the experiment cannot resolve, and the honest "
            "conclusion there is a ceiling on the evaluation rather than on the "
            "method.", "",
            "## 4. Contrasts computed elsewhere", "",
            "| result | where |", "|---|---|",
            "| retrieval / generation / ordering decomposition | `reports/stage1_evidence.md` |",
            "| selector ceiling, label-noise and feature ablations | `reports/stage1_diagnosis.md` |",
            "| randomized-graph ablation (runs in a scratch root) | `reports/ablation_randomized_graph.md` |",
            "| real collected preferences, once any exist | `reports/real_feedback.md` |", ""]

    text = "\n".join(out)
    (paths.REPORTS / "statistical_analysis.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(run())
