"""The student-facing study: generate plans, collect pairwise preferences.

Mounted separately from the researcher console because the two have
opposite goals. The console exposes every knob; this shows one question at
a time and nothing that would tell a student what answer we are hoping for.

What it collects, and why that shape: a student compares two generated
study plans and says which they would rather take. That is exactly the
input `pipeline.preference.build_pairs` consumes, so real choices drop
straight into the place currently filled by `simulate_choice`. Thumbs-up
and thumbs-down on a single plan would be weaker -- Bradley-Terry learns
from which of two won, and a rating has no opponent.

Server state is deliberately nil. The generated set is logged in full the
moment it is made, the browser walks the pair schedule, and each choice
posts back the ids. Two App Runner instances therefore need share nothing.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid
from typing import Any, Dict, List

from flask import Blueprint, jsonify, redirect, render_template, request

from app import budget, online_reward, plans as planlib, store
from pipeline import llm, paths

bp = Blueprint("study", __name__)

ROUNDS = 5                  # comparisons asked of one participant
MAX_QUERY_CHARS = 400
MAX_REASON_CHARS = 600

# The fixed vocabulary, shared with the offline analysis. Free text is kept
# too, but a closed set is what makes the reasons countable.
REASONS = [
    ("too_advanced", "Too advanced for me"),
    ("too_basic", "Too basic"),
    ("too_long", "Takes too long"),
    ("wrong_topic", "Not what I want to learn"),
    ("bad_order", "Wrong order"),
    ("low_quality", "Courses look weak"),
    ("other", "Something else"),
]
REASON_KEYS = {k for k, _ in REASONS}

_SERVICE: planlib.PlanService = None       # set by init_app


def init_app(app, catalog) -> None:
    global _SERVICE
    _SERVICE = planlib.PlanService(catalog, planlib.load_reward_model())
    app.register_blueprint(bp)


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit]


def _participant(body: Dict[str, Any]) -> str:
    """A pseudonymous id the browser keeps. No name, no email, nothing
    identifying -- enough to group one person's comparisons, and nothing
    that could re-identify them."""
    pid = _clean(str(body.get("participant_id") or ""), 64)
    return pid if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", pid or "") else f"anon-{uuid.uuid4().hex[:12]}"


@bp.route("/study")
def page():
    """One app. The model tab appears only with a valid token."""
    return render_template("app.html", rounds=ROUNDS, reasons=REASONS,
                           features=planlib.FEATURES,
                           show_inspect=_inspect_allowed())


@bp.route("/api/study/start", methods=["POST"])
def start():
    body = request.get_json(silent=True) or {}
    query = _clean(body.get("query"), MAX_QUERY_CHARS)
    if len(query) < 8:
        return jsonify({"error": "Tell us a bit more about what you want to learn."}), 400
    try:
        level = int(body.get("level", 0))
        hours = float(body.get("weekly_hours", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid level or hours."}), 400
    if level not in (0, 1, 2) or not (0.5 <= hours <= 60):
        return jsonify({"error": "Invalid level or hours."}), 400

    participant = _participant(body)
    session_id = f"{participant}-{uuid.uuid4().hex[:8]}"

    # The spend cap is charged inside build(), at the moment of the paid
    # call, so a query the catalog cannot serve costs an embedding rather
    # than a plan set.
    over_budget = {}

    def _charge():
        try:
            budget.charge(participant)
        except budget.BudgetExceeded as e:
            over_budget["scope"] = e.scope
            raise

    try:
        built = _SERVICE.build(query, level, hours, session_id, 0,
                               budget=llm.Budget(4, "study"), on_generate=_charge)
    except budget.BudgetExceeded:
        return jsonify({
            "error": f"The study has hit its {over_budget['scope']} limit. "
                     "Thank you for your interest — please try again tomorrow."}), 429

    if built and built.get("out_of_scope"):
        store.log_out_of_scope(
            participant_id=participant, session_id=session_id, query=query,
            fit=built["fit"], nearest_skills=built["nearest"])
        return jsonify({
            "out_of_scope": True,
            "error": "This catalog only covers computer science, software "
                     "development and data science. We don't have courses "
                     "for that topic, so any plan we built would be a bad "
                     "one. Try a goal in those areas instead."}), 422

    if not built:
        return jsonify({"error": "We could not build plans for that just now. "
                                 "Try rephrasing your goal."}), 503

    ev = store.log_plans_shown(
        participant_id=participant, session_id=session_id, round_index=0,
        query=query, level=level, weekly_hours=hours,
        plans=built["plans"], shortlist_arms=built["shortlist_arms"],
        goal_skills=built["goal_skills"], policy_id="plan_bestofn_v1",
        domain_fit=built.get("fit"),
        dataset_version=paths.dataset_version(),
        spec_version=_SERVICE.cat.spec_version)

    seed = int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:4], "big")
    sched = planlib.pair_schedule(len(built["plans"]), seed)[:ROUNDS]

    # Scores are withheld from the browser on purpose. Telling a student
    # which plan the model preferred would turn their judgement into
    # agreement, and agreement is not a preference.
    public = [{k: v for k, v in p.items() if k not in ("score", "features")}
              for p in built["plans"]]
    return jsonify({
        "participant_id": participant, "session_id": session_id,
        "plans_event_id": ev, "plans": public,
        "pairs": [list(p) for p in sched],
        "goal_skills": built["goal_skills"], "rounds": len(sched),
        "thin_coverage": bool(built.get("thin_coverage")),
    })


@bp.route("/api/study/choose", methods=["POST"])
def choose():
    body = request.get_json(silent=True) or {}
    need = ("plans_event_id", "left_plan_id", "right_plan_id", "chosen_plan_id")
    if not all(_clean(str(body.get(k) or ""), 80) for k in need):
        return jsonify({"error": "missing fields"}), 400
    left = _clean(str(body["left_plan_id"]), 80)
    right = _clean(str(body["right_plan_id"]), 80)
    chosen = _clean(str(body["chosen_plan_id"]), 80)
    if chosen not in (left, right):
        return jsonify({"error": "chosen plan was not one of the two shown"}), 400

    secs = body.get("seconds_on_screen")
    try:
        secs = None if secs is None else round(float(secs), 2)
    except (TypeError, ValueError):
        secs = None

    ev = store.log_choice(
        plans_event_id=_clean(str(body["plans_event_id"]), 80),
        participant_id=_participant(body),
        session_id=_clean(str(body.get("session_id") or ""), 80),
        round_index=int(body.get("round_index") or 0),
        left_plan_id=left, right_plan_id=right, chosen_plan_id=chosen,
        shown_order=[left, right], seconds_on_screen=secs)
    return jsonify({"choice_event_id": ev})


@bp.route("/api/study/reason", methods=["POST"])
def reason():
    body = request.get_json(silent=True) or {}
    cats = [c for c in (body.get("reason_categories") or []) if c in REASON_KEYS]
    text = _clean(body.get("reason_text"), MAX_REASON_CHARS)
    if not cats and not text:
        return jsonify({"skipped": True})
    ev = store.log_reason(
        choice_event_id=_clean(str(body.get("choice_event_id") or ""), 80),
        participant_id=_participant(body),
        session_id=_clean(str(body.get("session_id") or ""), 80),
        plan_id=_clean(str(body.get("plan_id") or ""), 80),
        reason_text=text, reason_categories=cats,
        free_text_only=bool(text and not cats))
    return jsonify({"reason_event_id": ev})


def _inspect_allowed() -> bool:
    """The inspector shows the model's internals and every comparison made.

    Not secret, but not something to leave on a public URL either. When
    THESIS_INSPECT_TOKEN is set (it is, in the deployed service) the page
    needs ?k=<token>; locally, with no token configured, it is open.
    """
    want = os.environ.get("THESIS_INSPECT_TOKEN", "")
    if not want:
        return True
    got = request.args.get("k", "")
    return hmac.compare_digest(got, want)


@bp.route("/inspect")
def inspect_page():
    """Kept so older links still work; the view now lives in a tab."""
    if not _inspect_allowed():
        return ("Not found", 404)
    q = request.query_string.decode()
    return redirect(f"/study?tab=model{'&' + q if q else ''}", code=302)


@bp.route("/api/study/inspect")
def inspect_data():
    """Prior weights, every update, and what each one moved.

    The whole state is derived from the event log on read, so this is a
    true account of how the live model got where it is rather than a
    summary written alongside it.
    """
    if not _inspect_allowed():
        return ("Not found", 404)
    svc = _SERVICE
    if svc is None or svc.prior is None:
        return jsonify({"error": "no reward model loaded"}), 503
    try:
        lr = float(request.args.get("lr", online_reward.DEFAULT_LR))
        anchor = float(request.args.get("anchor", online_reward.DEFAULT_ANCHOR))
    except ValueError:
        return jsonify({"error": "lr and anchor must be numbers"}), 400
    lr = max(0.0, min(lr, 2.0))
    anchor = max(0.0, min(anchor, 1.0))

    shown = store.store().read(store.PLANS_SHOWN)
    choices = store.store().read(store.CHOICE)
    reasons = store.store().read(store.REASON)
    pairs = online_reward.pairs_from_events(shown, choices, svc.mu, svc.sd)
    theta, history = online_reward.replay(svc.prior, pairs, lr, anchor)

    # Reason text, hung off the choice it belongs to, so the trace reads
    # as "they said this, and the model moved that way".
    by_choice = {}
    for r in reasons:
        by_choice.setdefault(r.get("choice_event_id"), []).append(
            {"text": r.get("reason_text"),
             "categories": r.get("reason_categories") or []})
    plan_meta = {}
    for ev in shown:
        for pl in ev.get("plans", []):
            plan_meta[pl.get("plan_id")] = {
                "strategy": pl.get("strategy"),
                "titles": [c.get("title") for c in pl.get("courses", [])],
            }
    by_pair = {p["event_id"]: p for p in pairs}
    for h in history:
        pr = by_pair.get(h["event_id"], {})
        h["chosen"] = plan_meta.get(pr.get("chosen_plan_id"), {})
        h["rejected"] = plan_meta.get(pr.get("rejected_plan_id"), {})
        h["reasons"] = by_choice.get(h["event_id"], [])

    # Rank movement on the most recent plan set: the claim made checkable.
    latest = sorted(shown, key=lambda e: e.get("created_at") or "")[-1] if shown else None
    movement, query = [], None
    if latest:
        movement = online_reward.rank_change(
            svc.prior, theta, latest.get("plans", []), svc.mu, svc.sd)
        query = latest.get("query")

    return jsonify({
        "features": planlib.FEATURES,
        "prior": [round(float(x), 4) for x in svc.prior],
        "live": [round(float(x), 4) for x in theta],
        "lr": lr, "anchor": anchor,
        "n_choices": len(choices), "n_usable_pairs": len(pairs),
        "n_participants": len({c.get("participant_id") for c in choices} - {None}),
        "history": history,
        "movement": movement, "movement_query": query,
        "defaults": {"lr": online_reward.DEFAULT_LR,
                     "anchor": online_reward.DEFAULT_ANCHOR},
    })


@bp.route("/api/study/health")
def health():
    """Liveness for App Runner, plus a cheap look at how collection is going."""
    ok = _SERVICE is not None and _SERVICE.cat is not None
    out = {"ok": ok, "store": store.store().name,
           "scored": _SERVICE is not None and _SERVICE.theta is not None}
    if request.args.get("counts"):
        out["counts"] = store.summary()
    return jsonify(out), (200 if ok else 503)
