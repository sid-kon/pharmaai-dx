import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState, DIMENSIONS, SEVERITY_LEVELS
from knowledge_base.pinecone_client import query_pinecone

load_dotenv()

# ---------------------------------------------------------------------------
# Shared LLM instance
# ---------------------------------------------------------------------------

_llm = None


def _get_llm() -> ChatAnthropic:
    """Lazy-load the ChatAnthropic model."""
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0.2,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    return _llm


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Load the system prompt from prompts/system_prompt.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "You are PharmaAI Dx, an expert diagnostic agent for pharma AI initiatives."


def _extract_json_block(text: str) -> dict:
    """
    Extract and parse the first JSON block from a model response.
    Falls back to a safe default structure if parsing fails.
    """
    # Try to find a ```json ... ``` block first
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try to find a raw { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return {}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def _merge_scores(
    existing: dict[str, str],
    new_scores: dict[str, str],
) -> dict[str, str]:
    """
    Merge new dimension scores into existing ones.
    Never overwrite an established High/Medium/Low with 'Not assessed'.
    """
    merged = dict(existing)
    for dim, severity in new_scores.items():
        if dim not in DIMENSIONS:
            continue
        if severity not in SEVERITY_LEVELS:
            continue
        # Only update if the existing score is still 'Not assessed'
        # OR the new score is a real assessment (not 'Not assessed')
        if merged.get(dim) == "Not assessed" or severity != "Not assessed":
            merged[dim] = severity
    return merged


def _get_last_user_message(state: AgentState) -> str:
    """Extract the content of the most recent HumanMessage."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return state.get("initiative_description", "")


# ---------------------------------------------------------------------------
# Node 1: retrieve_node
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve relevant RAG chunks from Pinecone based on the latest
    user message.
    """
    query = _get_last_user_message(state)

    if not query:
        return {"retrieved_chunks": []}

    chunks = query_pinecone(query, top_k=5)
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Node 2: diagnose_node
# ---------------------------------------------------------------------------

def diagnose_node(state: AgentState) -> dict:
    """
    Call Claude to assess failure dimensions based on the user's
    initiative description and retrieved RAG chunks.

    Returns updated dimension_scores and a conversational reply.
    """
    system_prompt = _load_system_prompt()
    initiative = state.get("initiative_description", "")
    chunks = state.get("retrieved_chunks", [])
    current_scores = state.get("dimension_scores", {})

    # Identify which dimensions still need assessment
    unassessed = [
        dim for dim, score in current_scores.items()
        if score == "Not assessed"
    ]
    assessed = [
        f"{dim}: {score}"
        for dim, score in current_scores.items()
        if score != "Not assessed"
    ]

    # Build RAG context block
    rag_context = ""
    if chunks:
        rag_context = (
            "\n\n## Retrieved Knowledge Base Evidence\n"
            + "\n---\n".join(chunks)
        )

    # Build the full diagnostic prompt
    user_prompt = f"""## Initiative Description
{initiative}
{rag_context}

## Current Dimension Assessment Status
Already assessed:
{chr(10).join(assessed) if assessed else "None yet"}

Still requiring assessment:
{chr(10).join(unassessed) if unassessed else "All dimensions assessed"}

## Your Task
Based on the initiative description and retrieved evidence above:
1. Assess as many of the unassessed dimensions as the information allows.
2. Ask a targeted follow-up question if critical information is missing
   for any dimension.
3. Return your response ONLY as a JSON block in this exact format:

```json
{{
  "dimension_scores": {{
    "Data Readiness": "High|Medium|Low|Not assessed",
    "Governance & Ownership": "High|Medium|Low|Not assessed",
    "Regulatory Alignment": "High|Medium|Low|Not assessed",
    "Change Management": "High|Medium|Low|Not assessed",
    "Technical Architecture Fit": "High|Medium|Low|Not assessed",
    "Pilot-to-Scale Design": "High|Medium|Low|Not assessed"
  }},
  "response": "Your conversational reply to the user, including findings and any follow-up questions."
}}
```

Important: Only include dimensions in dimension_scores where you have
enough information to assess. Set others to "Not assessed".
"""

    llm = _get_llm()
    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    response_text = result.content if hasattr(result, "content") else str(result)
    parsed = _extract_json_block(response_text)

    # Extract scores and response text from parsed JSON
    new_scores = parsed.get("dimension_scores", {})
    reply = parsed.get("response", response_text)

    # Merge scores — never downgrade an existing assessment
    updated_scores = _merge_scores(current_scores, new_scores)

    return {
        "messages": [AIMessage(content=reply)],
        "dimension_scores": updated_scores,
    }


# ---------------------------------------------------------------------------
# Node 3: check_complete_node
# ---------------------------------------------------------------------------

def check_complete_node(state: AgentState) -> dict:
    """
    Check whether all 6 dimensions have been assessed.
    Sets diagnosis_complete to True if none remain as 'Not assessed'.
    """
    scores = state.get("dimension_scores", {})
    all_assessed = all(
        score != "Not assessed" for score in scores.values()
    )
    return {"diagnosis_complete": all_assessed}


# ---------------------------------------------------------------------------
# Node 4: report_node
# ---------------------------------------------------------------------------

def report_node(state: AgentState) -> dict:
    """
    Generate the final structured markdown diagnostic report once all
    6 dimensions have been assessed.
    """
    scores = state.get("dimension_scores", {})
    messages = state.get("messages", [])
    initiative = state.get("initiative_description", "")

    # Build conversation summary for context
    convo_summary = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages[-10:]  # last 10 messages for context
    ])

    # Sort dimensions by severity for the report
    severity_order = {"High": 0, "Medium": 1, "Low": 2, "Not assessed": 3}
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: severity_order.get(x[1], 3),
    )

    scores_table = "\n".join([
        f"| {dim} | {sev} |"
        for dim, sev in sorted_scores
    ])

    report_prompt = f"""Generate a structured PharmaAI Dx diagnostic report in markdown format.

## Initiative Description
{initiative}

## Dimension Scores
{chr(10).join([f"{dim}: {sev}" for dim, sev in sorted_scores])}

## Conversation Context (last 10 exchanges)
{convo_summary}

Produce the report in EXACTLY this structure:

# PharmaAI Dx — Diagnostic Report

## Executive Summary
[2-3 sentences summarising the overall risk profile of this initiative
and the most critical failure dimensions identified.]

## Dimension Scores

| Dimension | Severity | Key Signals Identified |
|-----------|----------|----------------------|
{scores_table.replace('|  |', '| [Key signals from the conversation] |')}

[Fill in the Key Signals Identified column based on the conversation context.
Each cell should contain 1-2 specific signals observed, not generic statements.]

## Priority Recommendations
[List the top 3 recommendations ranked by severity. Each recommendation must:
- Name the specific dimension it addresses
- Cite a specific failure signal observed
- Provide a concrete, actionable next step]

### 1. [Highest severity dimension]
### 2. [Second priority]
### 3. [Third priority]

## Suggested Next Steps
[3-5 concrete next steps the client should take in the next 30 days,
specific to this initiative's risk profile.]

---
*Report generated by PharmaAI Dx | Powered by Agilsisum Consulting*
"""

    llm = _get_llm()
    result = llm.invoke([
        {"role": "user", "content": report_prompt},
    ])

    report_markdown = result.content if hasattr(result, "content") else str(result)

    return {
        "final_report": report_markdown,
        "messages": [AIMessage(content="Your full diagnostic report is ready. You can view and download it below.")],
    }

