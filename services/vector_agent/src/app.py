from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.vector_agent.src.agent import get_vector_agent
from shared.src.a2a_models import (
    AgentCard, AgentSkill, AgentProvider, AgentCapabilities, AgentInterface,
    Message,
)
from shared.src.a2a_server import create_a2a_router

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

from functools import lru_cache

@lru_cache(maxsize=1)
def _get_cached_agent():
    return get_vector_agent()

@app.post("/query")
async def query_vector(request: QueryRequest):
    agent = _get_cached_agent()
    response = agent.query(request.query)
    return {"answer": response}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ─── A2A Protocol ────────────────────────────────────────────────────────────

vector_agent_card = AgentCard(
    name="Vector Agent",
    description="Performs semantic similarity search over embedded medical reports (PDF/TXT) using ChromaDB and OpenAI embeddings. Retrieves the top-K most relevant document chunks and uses RetrievalQA to generate contextual answers.",
    version="1.0.0",
    provider=AgentProvider(organization="EPBA"),
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    skills=[
        AgentSkill(
            id="semantic-report-search",
            name="Semantic Report Search",
            description="Searches unstructured medical reports using vector similarity (text-embedding-3-small, k=3). Reports are chunked with RecursiveCharacterTextSplitter (chunk_size=600, overlap=50).",
            tags=["vector-search", "embeddings", "reports", "unstructured", "chromadb"],
            examples=[
                "What do the medical reports say about patient Abdul?",
                "Find clinical notes mentioning diabetes treatment",
                "Any recent lab reports for patient Sarah?",
            ],
            inputModes=["text/plain"],
            outputModes=["text/plain"],
        )
    ],
)

async def _process_vector_message(message: Message) -> str:
    # Extract query
    query = ""
    for part in message.parts:
        if part.text:
            query = part.text
            break
            
    # Extract trace_id and parent_id from metadata
    trace_id = message.metadata.get("langfuse_trace_id") if message.metadata else None
    parent_id = message.metadata.get("langfuse_parent_id") if message.metadata else None
    
    agent = _get_cached_agent()
    return agent.query(query, trace_id=trace_id, parent_observation_id=parent_id)

a2a_router = create_a2a_router(vector_agent_card, process_full_message=_process_vector_message)
app.include_router(a2a_router)
