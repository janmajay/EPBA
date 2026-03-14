from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from services.orchestrator.src.graph import create_graph
from shared.src.a2a_models import (
    AgentCard, AgentSkill, AgentProvider, AgentCapabilities, AgentInterface,
)
from shared.src.a2a_server import create_a2a_router
import os

app = FastAPI()

# Compile graph once
workflow = create_graph()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    final_answer: str
    sql_result: str
    vector_result: str
    timings: Optional[Dict[str, float]] = None

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Initial state
        initial_state = {
            "query": request.query,
            "sql_result": "",
            "vector_result": "",
            "final_answer": "",
            "timings": {}
        }
        
        # Invoke graph
        result = await workflow.ainvoke(initial_state)
        
        return QueryResponse(
            final_answer=result["final_answer"],
            sql_result=result["sql_result"],
            vector_result=result["vector_result"],
            timings=result.get("timings", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ─── A2A Protocol ────────────────────────────────────────────────────────────

orchestrator_agent_card = AgentCard(
    name="Orchestrator",
    description="The central coordinator of the EPBA system. Uses LangGraph to orchestrate parallel calls to SQL and Vector agents, then synthesizes results via the Summarization Agent. Provides a complete patient intelligence answer from a single natural language query.",
    version="1.0.0",
    provider=AgentProvider(organization="EPBA"),
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="patient-intelligence-query",
            name="Patient Intelligence Query",
            description="Accepts a natural language question about a patient, queries both structured (SQL) and unstructured (Vector) data in parallel, and returns a synthesized clinical answer with per-agent timings.",
            tags=["orchestration", "multi-agent", "langgraph", "patient-query", "rag"],
            examples=[
                "Give me the details about patient Abdul, his medical history and recent reports",
                "What medications is patient Sarah taking and are there any related clinical notes?",
            ],
            inputModes=["text/plain"],
            outputModes=["text/plain", "application/json"],
        )
    ],
)


async def _process_orchestrator_message(query: str) -> str:
    """Invoke the LangGraph workflow and return the final answer."""
    initial_state = {
        "query": query,
        "sql_result": "",
        "vector_result": "",
        "final_answer": "",
        "timings": {}
    }
    result = await workflow.ainvoke(initial_state)
    return result["final_answer"]


a2a_router = create_a2a_router(
    orchestrator_agent_card,
    process_message=_process_orchestrator_message,
)
app.include_router(a2a_router)
