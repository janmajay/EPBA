from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.summarization_agent.src.agent import get_summarizer
from shared.src.a2a_models import (
    AgentCard, AgentSkill, AgentProvider, AgentCapabilities, AgentInterface,
    Message,
)
from shared.src.a2a_server import create_a2a_router

app = FastAPI()

class SummarizeRequest(BaseModel):
    query: str
    sql_result: str
    vector_result: str

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    agent = get_summarizer()
    response = agent.summarize(request.query, request.sql_result, request.vector_result)
    return {"answer": response}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ─── A2A Protocol ────────────────────────────────────────────────────────────

summarizer_agent_card = AgentCard(
    name="Summarization Agent",
    description="Synthesizes a comprehensive clinical answer from structured (SQL) and unstructured (Vector) data sources. Flags conflicting information, provides citations, and generates a unified patient summary.",
    version="1.0.0",
    provider=AgentProvider(organization="EPBA"),
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["application/json", "text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="clinical-answer-synthesis",
            name="Clinical Answer Synthesis",
            description="Takes a query plus SQL and Vector agent results (via structured JSON data parts) and synthesizes a single, comprehensive clinical answer using GPT-4o-mini.",
            tags=["summarization", "synthesis", "clinical", "llm", "multi-source"],
            examples=[
                "Summarize patient Abdul's complete medical profile from both database and reports",
            ],
            inputModes=["application/json"],
            outputModes=["text/plain"],
        )
    ],
)


async def _process_summarizer_message(message: Message) -> str:
    """
    Handles A2A messages for the summarization agent.
    Expects either:
      - A single text part (treated as query with empty results)
      - A data part with {query, sql_result, vector_result}
    """
    query = ""
    sql_result = "N/A"
    vector_result = "N/A"

    for part in message.parts:
        if part.data is not None:
            query = part.data.get("query", query)
            sql_result = part.data.get("sql_result", sql_result)
            vector_result = part.data.get("vector_result", vector_result)
        elif part.text is not None and not query:
            query = part.text

    # Extract trace_id and parent_id from metadata
    trace_id = message.metadata.get("langfuse_trace_id") if message.metadata else None
    parent_id = message.metadata.get("langfuse_parent_id") if message.metadata else None

    agent = get_summarizer()
    return agent.summarize(query, sql_result, vector_result, trace_id=trace_id, parent_observation_id=parent_id)


a2a_router = create_a2a_router(
    summarizer_agent_card,
    process_full_message=_process_summarizer_message,
)
app.include_router(a2a_router)
