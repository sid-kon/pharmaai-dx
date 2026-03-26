from langchain_core.tools import tool
from knowledge_base.pinecone_client import query_pinecone


@tool
def retrieve_pharma_ai_knowledge(query: str) -> str:
    """
    Retrieve authoritative knowledge chunks from the PharmaAI Dx
    knowledge base stored in Pinecone.

    This tool searches across 15 primary sources covering six AI
    implementation failure dimensions:
      - Data Readiness
      - Governance & Ownership
      - Regulatory Alignment
      - Change Management
      - Technical Architecture Fit
      - Pilot-to-Scale Design

    Sources include FDA/EMA regulatory guidance, IBM Watson failure
    case studies, Google ML engineering frameworks (technical debt,
    ML test score), McKinsey AI adoption surveys, NHS AI Lab
    implementation guidance, FAIR data principles, and 21 CFR Part 11.

    Use this tool whenever you need citable evidence to support a
    diagnostic finding on any of the six dimensions.

    Args:
        query: A natural language search query describing what
               information you need (e.g. "FDA SaMD classification
               requirements for AI", "ML technical debt patterns",
               "clinical end-user adoption failure").

    Returns:
        Retrieved knowledge chunks joined by separators, ready to
        cite in a diagnostic response. Returns a message indicating
        no results if the knowledge base returns nothing.
    """
    chunks = query_pinecone(query, top_k=5)

    if not chunks:
        return (
            "No relevant chunks retrieved from the knowledge base "
            "for this query. Proceed with general knowledge."
        )

    return "\n---\n".join(chunks)

