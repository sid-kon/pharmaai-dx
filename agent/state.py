from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


DIMENSIONS = [
    "Data Readiness",
    "Governance & Ownership",
    "Regulatory Alignment",
    "Change Management",
    "Technical Architecture Fit",
    "Pilot-to-Scale Design",
]

SEVERITY_LEVELS = ["High", "Medium", "Low", "Not assessed"]


class AgentState(TypedDict):
    # Full conversation history (user + assistant turns)
    messages: Annotated[list, add_messages]

    # The user's description of their pharma AI initiative
    initiative_description: str

    # RAG chunks retrieved from Pinecone for the current query
    retrieved_chunks: list[str]

    # Keys are the 6 dimension names, values are severity strings
    dimension_scores: dict[str, str]

    # False until the agent has assessed all 6 dimensions
    diagnosis_complete: bool

    # Empty string until diagnosis_complete is True
    final_report: str


def get_initial_state() -> AgentState:
    """Return a fresh AgentState with all defaults."""
    return AgentState(
        messages=[],
        initiative_description="",
        retrieved_chunks=[],
        dimension_scores={dim: "Not assessed" for dim in DIMENSIONS},
        diagnosis_complete=False,
        final_report="",
    )
