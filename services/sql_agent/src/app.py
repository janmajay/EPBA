from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.sql_agent.src.agent import get_sql_agent
from shared.src.a2a_models import (
    AgentCard, AgentSkill, AgentProvider, AgentCapabilities, AgentInterface,
    Message,
)
from shared.src.a2a_server import create_a2a_router

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_db(request: QueryRequest):
    agent = get_sql_agent()
    response = agent.query(request.query)
    return {"answer": response}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ─── A2A Protocol ────────────────────────────────────────────────────────────

sql_agent_card = AgentCard(
    name="SQL Agent",
    description="Queries structured patient data (demographics, encounters, conditions, medications, observations, allergies, careplans, immunizations, procedures) from the FHIR-derived SQLite database using LLM-generated SQL.",
    version="1.0.0",
    provider=AgentProvider(organization="EPBA"),
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="structured-patient-query",
            name="Structured Patient Query",
            description="Translates natural language questions into SQL queries against a patient database with 9 normalized tables. Optimized with max_iterations=5 and fail-fast logic.",
            tags=["sql", "patient-data", "structured", "fhir", "database"],
            examples=[
                "Give me the details about patient Abdul",
                "What medications is patient Sarah currently taking?",
                "List all encounters for patient John in the last year",
            ],
            inputModes=["text/plain"],
            outputModes=["text/plain"],
        )
    ],
)

async def _process_sql_message(message: Message) -> str:
    # Extract query
    query = ""
    for part in message.parts:
        if part.text:
            query = part.text
            break
            
    # Extract trace_id and parent_id from metadata
    trace_id = message.metadata.get("langfuse_trace_id") if message.metadata else None
    parent_id = message.metadata.get("langfuse_parent_id") if message.metadata else None
    
    agent = get_sql_agent()
    return agent.query(query, trace_id=trace_id, parent_observation_id=parent_id)

a2a_router = create_a2a_router(sql_agent_card, process_full_message=_process_sql_message)
app.include_router(a2a_router)
