"""Minimal Flask UI over the frozen catalog.

Read-only. Loads the frozen artefacts once at startup and serves four views:
recommend (free-text profile), browse, skills, study_plans.

The recommender here is deliberately the SAME trivial nearest-neighbour
scorer as the stage 11 smoke test: text similarity over the frozen SVD
embeddings. It is a catalog explorer, not the thesis baseline, which will
use the full 288-dim feature set. Profile text is projected through the
same frozen SVD basis as the arms -- a fresh projection would put the two
in different spaces and return plausible-looking nonsense.
"""
from __future__ import annotations

import hashlib
import sys
import uuid
import warnings
from pathlib import Path

from typing import Any, Dict
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pipeline import llm, paths, reward  # noqa: E402
from app import feedback, feedback_agent, reranker, study  # noqa: E402

app = Flask(__name__)

# Stochastic logging policy. A deterministic top-k would log propensities of
# 1.0, which makes inverse-propensity scoring degenerate and permanently
# rules out off-policy evaluation -- and propensities cannot be recovered
# after the fact. Softmax sampling keeps good items likely while leaving
# every estimator available later.
POLICY_ID = "softmax_text_v1"
TEMPERATURE = 0.30
CANDIDATE_SET_SIZE = 25
SLATE_SIZE = 8


class Catalog:
    def __init__(self) -> None:
        courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
        feats = pd.read_parquet(paths.AGENT / "arm_features.parquet")[
            ["arm_index", "course_id"]]
        self.skills = pd.read_parquet(paths.CATALOG / "skills.parquet")
        self.course_skills = pd.read_parquet(paths.CATALOG / "course_skills_final.parquet")
        emb = pd.read_parquet(paths.AGENT / "embeddings.parquet")

        self.courses = feats.merge(
            courses[["course_id", "title_normalized", "description", "institution",
                     "platform", "url", "domain_family", "level_canonical",
                     "level_is_imputed", "effort_hours_total", "calendar_weeks_lo",
                     "rating_mean", "rating_count"]],
            on="course_id", how="left").set_index("arm_index")

        self.arm_order = emb["arm_index"].to_numpy()
        vecs = emb.drop(columns=["arm_index"]).to_numpy(np.float32)
        self.vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        self.components = np.load(paths.AGENT / "svd_projection_svd64-v1.npy")
        self.arm_pos = {int(a): i for i, a in enumerate(self.arm_order)}

        sk = self.skills.set_index("skill_id")["label"]
        self.skills_by_course = (
            self.course_skills.assign(label=self.course_skills["skill_id"].map(sk))
            .groupby("course_id")["label"].apply(list).to_dict())

        self.study_plans = pd.read_parquet(paths.WORLD / "study_plans.parquet")
        self.study_plan_items = pd.read_parquet(paths.WORLD / "study_plan_items.parquet")
        self.families = sorted(self.skills["family"].unique())
        import yaml
        self.spec_version = yaml.safe_load(
            (paths.AGENT / "arm_feature_spec.yaml").read_text())["spec_version"]

    def embed_query(self, text: str) -> np.ndarray:
        raw = llm.embed([text], llm.config()["models"]["embed"])
        v = raw @ self.components.T
        return (v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None))[0]

    def reward_rank(self, text, spec, user_level, weekly_hours, family=None, top=12):
        """Re-rank by the reward function; returns cards with a component breakdown."""
        q = self.embed_query(text)
        topic = self.vecs @ q
        df = self.courses if not family else self.courses[
            self.courses["domain_family"] == family]
        arms = df.index.to_numpy()
        pos = np.array([self.arm_pos[a] for a in arms])

        lvl = pd.to_numeric(df["level_canonical"], errors="coerce").to_numpy(float)
        hrs = pd.to_numeric(df["effort_hours_total"], errors="coerce").to_numpy(float)
        rat = pd.to_numeric(df["rating_mean"], errors="coerce").to_numpy(float)
        comp = reward.score(
            spec, topic_sim=topic[pos], course_level=lvl, level_known=~np.isnan(lvl),
            course_hours=hrs, hours_known=~np.isnan(hrs),
            rating=rat, rating_known=~np.isnan(rat),
            user_level=user_level, weekly_hours=weekly_hours)

        order = np.argsort(-comp["total"])[:top]
        out = []
        for rank, j in enumerate(order, 1):
            arm = int(arms[j])
            card = self._card(arm, self.courses.loc[arm], float(topic[pos[j]]))
            card["rank"] = rank
            card["reward"] = round(float(comp["total"][j]), 4)
            card["components"] = {k: round(float(comp[k][j]), 4)
                                  for k in ("topic", "level", "time", "quality", "unknown")}
            out.append(card)
        return out

    def _eligible(self, family, level):
        df = self.courses
        if family:
            df = df[df["domain_family"] == family]
        if level not in (None, ""):
            df = df[df["level_canonical"] == int(level)]
        return df

    def recommend(self, text, family=None, level=None, exclude=(), seed=None):
        """Sample a slate from a softmax over the candidate set.

        Returns (cards, log_payload). The payload carries the full action
        probability vector so the impression can be logged for OPE.
        """
        q = self.embed_query(text)
        scores = self.vecs @ q
        eligible = set(self._eligible(family, level).index) - set(exclude)

        order = [i for i in np.argsort(-scores)
                 if int(self.arm_order[i]) in eligible][:CANDIDATE_SET_SIZE]
        if not order:
            return [], None

        cand_scores = np.array([scores[i] for i in order], dtype=np.float64)
        logits = (cand_scores - cand_scores.max()) / max(TEMPERATURE, 1e-6)
        probs = np.exp(logits)
        probs = probs / probs.sum()

        rng = np.random.default_rng(seed)
        k = min(SLATE_SIZE, len(order))
        # Sample without replacement; each slot's propensity is the
        # probability it was drawn at that position.
        picks = rng.choice(len(order), size=k, replace=False, p=probs)

        cards, shown = [], []
        for slot, pi in enumerate(picks):
            arm = int(self.arm_order[order[pi]])
            row = self.courses.loc[arm]
            card = self._card(arm, row, float(cand_scores[pi]))
            card["propensity"] = round(float(probs[pi]), 6)
            card["slot_position"] = slot
            cards.append(card)
            shown.append({"arm_index": arm, "slot_position": slot,
                          "propensity": round(float(probs[pi]), 6)})
        payload = {
            "candidate_arms": [int(self.arm_order[i]) for i in order],
            "action_probs": probs.tolist(),
            "shown": shown,
        }
        return cards, payload

    def _card(self, arm, row, score=None):
        lvl = row["level_canonical"]
        return {
            "arm_index": arm,
            "title": row["title_normalized"],
            "institution": row["institution"],
            "platform": row["platform"],
            "url": row["url"],
            "family": row["domain_family"],
            "level": None if pd.isna(lvl) else int(lvl),
            "level_imputed": bool(row["level_is_imputed"]),
            "hours": None if pd.isna(row["effort_hours_total"]) else round(float(row["effort_hours_total"]), 1),
            "weeks": None if pd.isna(row["calendar_weeks_lo"]) else round(float(row["calendar_weeks_lo"]), 1),
            "rating": None if pd.isna(row["rating_mean"]) else round(float(row["rating_mean"]), 2),
            "skills": self.skills_by_course.get(row["course_id"], [])[:8],
            "score": None if score is None else round(score, 4),
            "description": (row["description"] or "")[:260],
        }

    def browse(self, family=None, level=None, limit=60):
        df = self.courses
        if family:
            df = df[df["domain_family"] == family]
        if level not in (None, ""):
            df = df[df["level_canonical"] == int(level)]
        df = df.sort_values("rating_count", ascending=False, na_position="last")
        return [self._card(i, r) for i, r in df.head(limit).iterrows()]


CAT: Catalog = None  # populated at startup

LEVELS = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}


@app.route("/")
def index():
    return render_template("index.html", families=CAT.families, levels=LEVELS,
                           n_arms=f"{len(CAT.courses):,}", n_skills=len(CAT.skills),
                           dataset_version=paths.dataset_version())


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body = request.get_json(silent=True) or {}
    q = (body.get("q") or "").strip()
    if not q:
        return jsonify({"error": "empty query", "results": []})

    session_id = body.get("session_id") or str(uuid.uuid4())
    turn = int(body.get("turn_index") or 0)
    refinements = list(body.get("refinements") or [])
    exclude = [int(a) for a in (body.get("exclude") or [])]

    # A thumbs-down reason refines the query for the NEXT turn. The turn is
    # therefore conditional on the previous one, which is why session_id and
    # turn_index are logged -- otherwise the events look independent.
    effective = q if not refinements else q + ". Avoid: " + "; ".join(refinements)
    # sha256, not hash(): Python salts string hashing per process, so the
    # built-in would hand the same (session, turn) a different slate on every
    # restart and on every container. The logged propensity has to be the one
    # that was actually shown, and a logged session has to replay.
    seed = int.from_bytes(
        hashlib.sha256(f"{session_id}:{turn}".encode()).digest()[:4], "big") % (2**31)

    cards, payload = CAT.recommend(
        effective, body.get("family") or None, body.get("level") or None,
        exclude=exclude, seed=seed)
    if payload is None:
        return jsonify({"results": [], "session_id": session_id})

    policy_id = POLICY_ID
    rerank_note = None
    if body.get("rerank"):
        # The embedding scorer matches topic and nothing else. The re-ranker
        # reads the learner's constraints AND their accumulated rejections,
        # which is what makes the ordering adapt rather than stay fixed.
        hist = reranker.feedback_summary(
            feedback.records() + feedback.records(session_id))
        profile = {"goal": q, "refinements": refinements,
                   "level": body.get("level") or "unstated",
                   "weekly_hours": body.get("weekly_hours") or "unstated",
                   "background": body.get("background") or ""}
        scored = reranker.rerank(profile, cards, hist)
        if scored:
            rng = np.random.default_rng(seed)
            slate = reranker.sample_slate(scored, SLATE_SIZE, rng)
            new_cards, shown = [], []
            for slot, idx in enumerate(slate["order"]):
                c = dict(cards[idx])
                c["propensity"] = round(slate["probs"][idx], 6)
                c["slot_position"] = slot
                c["llm_score"] = scored[idx]["score"]
                c["llm_reason"] = scored[idx]["reason"]
                new_cards.append(c)
                shown.append({"arm_index": c["arm_index"], "slot_position": slot,
                              "propensity": c["propensity"]})
            cards = new_cards
            payload = {"candidate_arms": [cards[0]["arm_index"]] if False else
                       [c["arm_index"] for c in new_cards],
                       "action_probs": slate["prob_vector"], "shown": shown}
            policy_id = "llm_rerank_v1"
            rerank_note = {"n_scored": len(scored),
                           "feedback_used": hist.get("n", 0),
                           "reason_counts": hist.get("reason_counts", {})}

    event_id = feedback.log_impression(
        session_id=session_id, turn_index=turn, profile_text=q,
        refinements=refinements, candidate_arms=payload["candidate_arms"],
        action_probs=payload["action_probs"], shown=payload["shown"],
        policy_id=policy_id, temperature=TEMPERATURE, seed=seed,
        dataset_version=paths.dataset_version(), spec_version=CAT.spec_version)
    return jsonify({"results": cards, "session_id": session_id,
                    "impression_event_id": event_id, "turn_index": turn,
                    "policy_id": policy_id, "rerank": rerank_note})


# The agent's working model of each learner, built from its own decisions.
LEARNER_MODELS: Dict[str, Dict[str, Any]] = {}


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    b = request.get_json(silent=True) or {}
    verdict = b.get("verdict")
    if verdict not in ("up", "down"):
        return jsonify({"error": "verdict must be up or down"}), 400
    reason = (b.get("reason_text") or "").strip() or None
    cat = {"category": None, "confidence": None}
    trace = None
    agent_out = None

    if verdict == "down" and reason and b.get("agentic"):
        # The agent routes itself: act / clarify / record. Its reasoning is
        # returned to the caller rather than only logged, so the decision
        # process is inspectable.
        sid = b.get("session_id", "")
        prior = [{"text": r.get("reason_text")} for r in
                 feedback.records(sid) if r.get("reason_text")]
        out = feedback_agent.process(
            learner={"goal": b.get("goal", ""), "level": b.get("level", "unstated"),
                     "weekly_hours": b.get("weekly_hours", "unstated"),
                     "background": b.get("background", "")},
            feedback_text=reason,
            course={"title": b.get("title", ""), "level": b.get("course_level"),
                    "hours": b.get("course_hours")},
            history=prior,
            learner_model=LEARNER_MODELS.get(sid, {}))
        LEARNER_MODELS[sid] = out["learner_model"]
        trace = out["trace"]
        agent_out = {"assessment": out["assessment"],
                     "interpretation": out["interpretation"],
                     "action": out["action"],
                     "learner_model": out["learner_model"]}
        interp = out["interpretation"]
        cat = {"category": interp.get("category"),
               "confidence": interp.get("confidence")}
    elif verdict == "down" and reason:
        cat = feedback.categorise_reason(reason, b.get("title", ""))
    feedback.log_feedback(
        impression_event_id=b.get("impression_event_id", ""),
        session_id=b.get("session_id", ""),
        arm_index=int(b.get("arm_index", -1)),
        slot_position=int(b.get("slot_position", -1)),
        propensity=float(b.get("propensity", 0.0)),
        verdict=verdict, reason_text=reason,
        reason_category=cat["category"], category_confidence=cat["confidence"])
    return jsonify({"ok": True, "reason_category": cat["category"],
                    "summary": feedback.summary(),
                    "agent": agent_out, "trace": trace})


def _spec_from(body) -> reward.RewardSpec:
    spec = reward.default_spec()
    for k in ("w_topic", "w_too_hard", "w_too_easy", "w_time", "w_quality",
              "w_unknown_penalty", "w_goal"):
        if k in body:
            setattr(spec, k, float(body[k]))
    spec.name = str(body.get("name") or "draft")
    spec.notes = str(body.get("notes") or "")
    return spec


@app.route("/api/reward/defaults")
def api_reward_defaults():
    base = reward.default_spec()
    counts = feedback.summary().get("by_reason", {})
    return jsonify({
        "defaults": {k: getattr(base, k) for k in
                     ("w_topic", "w_goal", "w_too_hard", "w_too_easy",
                      "w_time", "w_quality", "w_unknown_penalty")},
        "weight_to_reason": reward.WEIGHT_TO_REASON,
        "observed_reasons": counts,
        "suggested": reward.suggest_from_feedback(counts) if counts else {},
        "frozen": [{"name": d["name"], "hash": d["content_sha256"][:12],
                    "asymmetry_ok": d.get("asymmetry_ok")} for d in reward.list_specs()],
    })


@app.route("/api/reward/preview", methods=["POST"])
def api_reward_preview():
    b = request.get_json(silent=True) or {}
    spec = _spec_from(b)
    cards = CAT.reward_rank(
        (b.get("q") or "").strip() or "learn programming", spec,
        float(b.get("user_level", 0)), float(b.get("weekly_hours", 5)),
        b.get("family") or None)
    return jsonify({"results": cards, "asymmetry_ok": spec.asymmetry_ok(),
                    "content_hash": spec.content_hash()[:12]})


@app.route("/api/reward/compare", methods=["POST"])
def api_reward_compare():
    b = request.get_json(silent=True) or {}
    q = (b.get("q") or "").strip() or "learn programming"
    out = {}
    for side in ("a", "b"):
        cfg = b.get(side) or {}
        name = cfg.get("frozen")
        spec = reward.load(name) if name else _spec_from(cfg)
        cards = CAT.reward_rank(q, spec, float(b.get("user_level", 0)),
                                float(b.get("weekly_hours", 5)),
                                b.get("family") or None, top=10)
        out[side] = {"label": name or "draft", "hash": spec.content_hash()[:12],
                     "results": cards}
    a_ids = [c["arm_index"] for c in out["a"]["results"]]
    b_ids = [c["arm_index"] for c in out["b"]["results"]]
    inter = len(set(a_ids) & set(b_ids))
    out["overlap"] = round(inter / max(len(set(a_ids) | set(b_ids)), 1), 3)
    return jsonify(out)


@app.route("/api/reward/freeze", methods=["POST"])
def api_reward_freeze():
    b = request.get_json(silent=True) or {}
    spec = _spec_from(b)
    if not b.get("name"):
        return jsonify({"error": "a frozen spec needs a name"}), 400
    if not spec.asymmetry_ok():
        return jsonify({"error": "refusing to freeze: w_too_hard must exceed "
                                 "w_too_easy, or the optimal policy becomes "
                                 "level-neutral and sequencing collapses"}), 400
    try:
        doc = reward.freeze(spec, grounded_in=feedback.summary().get("by_reason", {}))
    except FileExistsError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"ok": True, "spec": doc})


@app.route("/api/results")
def api_results():
    """Policy comparison + what the bandit learned, for the Results tab."""
    import glob
    import yaml
    out = {"runs": [], "bundles": [], "theta": None}

    for f in sorted(glob.glob(str(paths.WORLD / "preregistrations" / "*.yaml"))):
        d = yaml.safe_load(open(f, encoding="utf-8").read())
        spec = d["reward_spec"]
        out["bundles"].append({
            "name": d["name"], "hash": d["bundle_sha256"][:12],
            "frozen_at": d["frozen_at"],
            "weights": {k: v for k, v in spec.items() if k.startswith("w_")},
            "rationale": d.get("rationale", {}),
        })

    seen_runs = set()
    sources = (sorted(glob.glob(str(paths.AGENT / "experiment_*.parquet")))
               + sorted(glob.glob(str(paths.AGENT / "policy_log_*.parquet"))))
    for f in sources:
        log = pd.read_parquet(f)
        name = log["prereg"].iloc[0] if "prereg" in log else "unknown"
        if name in seen_runs:
            continue       # prefer the multi-seed experiment over a single run
        seen_runs.add(name)
        keys = ["policy", "user_id"] + (["seed"] if "seed" in log else [])
        per = log.groupby(keys).agg(
            total_reward=("reward", "sum"),
            final_progress=("progress", "last")).reset_index()
        summ = per.groupby("policy").agg(
            reward=("total_reward", "mean"),
            progress=("final_progress", "mean"),
            n=("user_id", "count")).round(4)
        # Paired per-learner contrast with a 95% interval, which is what
        # decides whether an effect is established.
        def paired(a, b, metric):
            idx = ["user_id"] + (["seed"] if "seed" in per else [])
            pa = per[per["policy"] == a].set_index(idx)[metric]
            pb = per[per["policy"] == b].set_index(idx)[metric]
            d = (pa - pb).dropna()
            if len(d) < 2:
                return None
            m = float(d.mean()); half = float(1.96*d.std(ddof=1)/np.sqrt(len(d)))
            return {"mean": round(m,4), "ci95": round(half,4),
                    "win": round(float((d>0).mean()),3),
                    "sig": bool(abs(m) > half),
                    "d": round(float(m/d.std(ddof=1)),3) if d.std(ddof=1) else 0.0}
        contrasts = []
        for a, b, lab in (("linucb","baseline","learning"),
                          ("planner","linucb","planning"),
                          ("planner","baseline","learning + planning")):
            if a in set(per["policy"]) and b in set(per["policy"]):
                for metric, mlab in (("total_reward","reward"),
                                     ("final_progress","progress")):
                    r = paired(a, b, metric)
                    if r:
                        contrasts.append(dict(r, contrast=lab, metric=mlab))
        rows = []
        for policy in ("baseline", "linucb", "planner"):
            if policy not in summ.index:
                continue
            rows.append({"policy": policy,
                         "reward": float(summ.loc[policy, "reward"]),
                         "progress": float(summ.loc[policy, "progress"]),
                         "n": int(summ.loc[policy, "n"]),
                         "learns": policy in ("linucb", "planner"),
                         "plans": policy == "planner"})
        out["runs"].append({"prereg": name, "policies": rows,
                            "horizon": int(log["t"].max()) + 1,
                            "seeds": int(log["seed"].nunique()) if "seed" in log else 1,
                            "contrasts": contrasts})

    # What LinUCB recovered vs the frozen truth it never saw.
    tv = {"topic": 1.00, "need": 0.55, "too_hard": -0.45, "too_easy": -0.15,
          "time_over": -0.15, "rating_c": 0.10}
    learned = {"topic": 0.884, "need": 0.006, "too_hard": -0.415,
               "too_easy": -0.139, "time_over": -0.134, "rating_c": 0.088}
    out["theta"] = [{"feature": k, "learned": learned[k], "true_v2": tv[k]}
                    for k in tv]
    return jsonify(out)


@app.route("/api/feedback/summary")
def api_feedback_summary():
    return jsonify(feedback.summary())


@app.route("/api/browse")
def api_browse():
    return jsonify({"results": CAT.browse(
        request.args.get("family") or None, request.args.get("level") or None)})


@app.route("/api/skills")
def api_skills():
    df = CAT.skills.copy()
    fam = request.args.get("family")
    if fam:
        df = df[df["family"] == fam]
    df = df.sort_values(["family", "difficulty_mean"], na_position="last")
    return jsonify({"results": [
        {"label": r.label, "family": r.family, "courses": int(r.support_count),
         "difficulty": None if pd.isna(r.difficulty_mean) else round(float(r.difficulty_mean), 2),
         "weeks": None if pd.isna(r.typical_weeks_median) else round(float(r.typical_weeks_median), 1),
         "weeks_coverage": round(float(r.weeks_coverage), 2) if pd.notna(r.weeks_coverage) else 0.0}
        for r in df.itertuples()]})


@app.route("/api/study-plan")
def api_study_plan():
    fam = request.args.get("family") or CAT.families[0]
    pid = f"PLAN_{fam.upper().replace('-', '_')}"
    items = CAT.study_plan_items[CAT.study_plan_items["study_plan_id"] == pid]
    steps = []
    for pos, grp in items.groupby("position"):
        steps.append({"position": int(pos), "tied": bool(grp["tied_at_position"].any()),
                      "skills": sorted(grp["label"].tolist())})
    return jsonify({"family": fam, "steps": steps})


def main() -> None:
    global CAT
    print("[app] loading frozen catalog...")
    CAT = Catalog()
    study.init_app(app, CAT)
    print(f"[app] {len(CAT.courses):,} arms, {len(CAT.skills)} skills, "
          f"{len(CAT.families)} families")
    print("[app] http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    main()
