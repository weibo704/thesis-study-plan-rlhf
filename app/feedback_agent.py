"""LangGraph agent that decides for itself how to handle feedback.

The hardcoded path this replaces was three fixed steps: classify the reason
into a closed vocabulary, drop the rejected arm, append the text to the
query. The agent had no say in whether the feedback was worth acting on,
what it actually meant for this learner, or what to do about it.

Here the agent routes itself:

    assess  -- is this feedback credible and actionable? does it contradict
               what the learner said earlier?
      |
      +-- not actionable --> record and stop (no change to the policy)
      +-- ambiguous      --> formulate a clarifying question
      +-- actionable     --> interpret -> update the learner model -> act

`interpret` is open-ended: the agent writes what it believes changed about
the learner, in its own words, rather than picking from a fixed list. The
closed vocabulary is still emitted alongside, because the preference model
and the reward spec are defined over it -- but it is now a projection of
the agent's reasoning, not the reasoning itself.

SELECTION EFFECT: when the agent declines feedback, the preference model
never sees it. Every decision is logged with its reason so that filtering
can be measured rather than silently confounding the reward model.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from pipeline import llm
from app.feedback import REASON_CATEGORIES


class FeedbackState(TypedDict, total=False):
    # inputs
    learner: Dict[str, Any]
    feedback_text: str
    course: Dict[str, Any]
    history: List[Dict[str, Any]]
    learner_model: Dict[str, Any]
    # agent's own outputs
    assessment: Dict[str, Any]
    interpretation: Dict[str, Any]
    action: Dict[str, Any]
    trace: List[Dict[str, Any]]


ASSESS_SYSTEM = (
    "You decide whether a piece of learner feedback should change how we "
    "recommend to them.\n\n"
    "You see the learner, the course they rejected, what they said, and what "
    "they have said before.\n\n"
    "Judge:\n"
    "- actionable: does this imply a concrete change to future "
    "recommendations?\n"
    "- credible: is it consistent with what they said before, or does it "
    "contradict them? Contradiction does not mean reject -- people change "
    "their minds -- but say so.\n"
    "- specific: is it about THIS course, or a general preference?\n\n"
    "Choose a route:\n"
    '  "act"      - clear enough to change behaviour now\n'
    '  "clarify"  - plausible but too vague to act on without asking\n'
    '  "record"   - note it, change nothing (venting, off-topic, or a '
    "one-off that should not move the policy)\n"
    'Return JSON: {"route": "act|clarify|record", "actionable": true, '
    '"credible": true, "contradicts_history": false, '
    '"confidence": 0.0-1.0, "why": "max 20 words"}'
)

INTERPRET_SYSTEM = (
    "You maintain a working model of one learner, updated from their "
    "feedback.\n\n"
    "Write what you now believe about them in your own words -- do not pick "
    "from a list. Then, separately, project that belief onto the fixed "
    "categories the downstream reward model is defined over.\n\n"
    f"Categories: {', '.join(REASON_CATEGORIES)}\n\n"
    'Return JSON: {"belief": "one sentence, what you now think about this '
    'learner", "changed": "what is different from before, or none", '
    '"category": "<one category>", "constraints": {"max_level": 0-2 or null, '
    '"max_hours": number or null, "avoid_topics": ["..."]}, '
    '"confidence": 0.0-1.0}'
)

CLARIFY_SYSTEM = (
    "The learner gave feedback too vague to act on. Write ONE short question "
    "that would resolve the ambiguity. Ask about the thing that would most "
    "change the recommendation.\n"
    'Return JSON: {"question": "...", "resolves": "max 12 words"}'
)


def _step(state: FeedbackState, name: str, data: Dict[str, Any]) -> None:
    state.setdefault("trace", []).append({"node": name, **data})


def assess(state: FeedbackState) -> FeedbackState:
    out = llm.chat_json(
        [{"role": "system", "content": ASSESS_SYSTEM},
         {"role": "user", "content": json.dumps({
             "learner": state.get("learner", {}),
             "course_rejected": state.get("course", {}),
             "they_said": state.get("feedback_text", ""),
             "said_before": state.get("history", [])[-6:],
         }, ensure_ascii=False)}],
        model=llm.config()["models"]["extract"], role="feedback_agent")
    out = out or {"route": "record", "why": "assessment unavailable",
                  "confidence": 0.0}
    route = out.get("route")
    if route not in ("act", "clarify", "record"):
        out["route"] = "record"
    state["assessment"] = out
    _step(state, "assess", {"route": out["route"], "why": out.get("why", ""),
                            "confidence": out.get("confidence")})
    return state


def interpret(state: FeedbackState) -> FeedbackState:
    out = llm.chat_json(
        [{"role": "system", "content": INTERPRET_SYSTEM},
         {"role": "user", "content": json.dumps({
             "learner": state.get("learner", {}),
             "current_model": state.get("learner_model", {}),
             "course_rejected": state.get("course", {}),
             "they_said": state.get("feedback_text", ""),
             "assessment": state.get("assessment", {}),
         }, ensure_ascii=False)}],
        model=llm.config()["models"]["extract"], role="feedback_agent")
    out = out or {"belief": "", "category": "other", "confidence": 0.0}
    if out.get("category") not in REASON_CATEGORIES:
        out["category"] = "other"
    state["interpretation"] = out
    _step(state, "interpret", {"belief": out.get("belief", ""),
                               "category": out.get("category"),
                               "changed": out.get("changed")})
    return state


def apply_update(state: FeedbackState) -> FeedbackState:
    """Fold the agent's belief into the persistent learner model."""
    model = dict(state.get("learner_model") or {})
    interp = state.get("interpretation", {})
    cons = interp.get("constraints") or {}
    for key in ("max_level", "max_hours"):
        if cons.get(key) is not None:
            model[key] = cons[key]
    avoid = set(model.get("avoid_topics", [])) | set(cons.get("avoid_topics") or [])
    if avoid:
        model["avoid_topics"] = sorted(avoid)
    beliefs = list(model.get("beliefs", []))
    if interp.get("belief"):
        beliefs.append(interp["belief"])
    model["beliefs"] = beliefs[-8:]
    state["learner_model"] = model
    state["action"] = {"kind": "updated_model", "applied": True,
                       "category": interp.get("category"),
                       "constraints": cons}
    _step(state, "apply", {"constraints": cons, "n_beliefs": len(beliefs)})
    return state


def clarify(state: FeedbackState) -> FeedbackState:
    out = llm.chat_json(
        [{"role": "system", "content": CLARIFY_SYSTEM},
         {"role": "user", "content": json.dumps({
             "learner": state.get("learner", {}),
             "they_said": state.get("feedback_text", ""),
             "course_rejected": state.get("course", {}),
         }, ensure_ascii=False)}],
        model=llm.config()["models"]["extract"], role="feedback_agent")
    out = out or {"question": "Could you say a bit more about what did not fit?"}
    state["action"] = {"kind": "ask_clarification", "applied": False,
                       "question": out.get("question")}
    _step(state, "clarify", {"question": out.get("question")})
    return state


def record_only(state: FeedbackState) -> FeedbackState:
    """Logged, but deliberately NOT fed to the policy or reward model."""
    state["action"] = {"kind": "recorded_no_change", "applied": False,
                       "why": state.get("assessment", {}).get("why", "")}
    _step(state, "record", {"why": state["action"]["why"]})
    return state


def _route(state: FeedbackState) -> str:
    return state.get("assessment", {}).get("route", "record")


def build_graph():
    g = StateGraph(FeedbackState)
    g.add_node("assess", assess)
    g.add_node("interpret", interpret)
    g.add_node("apply", apply_update)
    g.add_node("clarify", clarify)
    g.add_node("record", record_only)
    g.set_entry_point("assess")
    g.add_conditional_edges("assess", _route,
                            {"act": "interpret", "clarify": "clarify",
                             "record": "record"})
    g.add_edge("interpret", "apply")
    g.add_edge("apply", END)
    g.add_edge("clarify", END)
    g.add_edge("record", END)
    return g.compile()


_GRAPH = None


def process(learner: Dict[str, Any], feedback_text: str, course: Dict[str, Any],
            history: List[Dict[str, Any]], learner_model: Dict[str, Any]
            ) -> Dict[str, Any]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final = _GRAPH.invoke({
        "learner": learner, "feedback_text": feedback_text, "course": course,
        "history": history or [], "learner_model": learner_model or {},
        "trace": [],
    })
    return {
        "assessment": final.get("assessment", {}),
        "interpretation": final.get("interpretation", {}),
        "action": final.get("action", {}),
        "learner_model": final.get("learner_model", learner_model or {}),
        "trace": final.get("trace", []),
    }
