from langgraph.graph import StateGraph, START, END

from agent.state import AgentState, get_initial_state
from agent.nodes import (
    retrieve_node,
    diagnose_node,
    check_complete_node,
    report_node,
)


# ---------------------------------------------------------------------------
# Conditional edge logic
# ---------------------------------------------------------------------------

def _route_after_check(state: AgentState) -> str:
    """
    Route to 'report' if all dimensions are assessed,
    otherwise end the turn and wait for the next user message.
    """
    if state.get("diagnosis_complete", False):
        return "report"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

# Register nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("diagnose", diagnose_node)
builder.add_node("check_complete", check_complete_node)
builder.add_node("report", report_node)

# Edges
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "diagnose")
builder.add_edge("diagnose", "check_complete")
builder.add_conditional_edges(
    "check_complete",
    _route_after_check,
    {
        "report": "report",
        END: END,
    },
)
builder.add_edge("report", END)

# Compile
pharmaai_graph = builder.compile()

# Re-export get_initial_state for convenience so app.py only needs
# to import from agent.graph
__all__ = ["pharmaai_graph", "get_initial_state"]

