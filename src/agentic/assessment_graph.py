"""Minimal LangGraph wrapper around the deterministic assess-fast pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_assess_graph(
    execute_assess: Callable[[str], dict[str, Any] | None],
):
    """
    Compile a single-node graph that runs `execute_assess(encounter_id)` and records a short trace.

    `execute_assess` must return the same dict as `/assess-fast` or None if the encounter is missing.
    """
    from typing import TypedDict

    from langgraph.graph import END, StateGraph

    class AgentState(TypedDict, total=False):
        encounter_id: str
        result: dict[str, Any] | None
        agent_trace: list[str]

    def node_assess(state: AgentState):
        r = execute_assess(state["encounter_id"])
        return {"result": r, "agent_trace": ["langgraph:deterministic_assess"]}

    g = StateGraph(AgentState)
    g.add_node("assess", node_assess)
    g.set_entry_point("assess")
    g.add_edge("assess", END)
    return g.compile()
