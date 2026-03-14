import os
import httpx
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from shared.src.config import settings
from shared.src.logger import configure_logger, log_execution_time
from shared.src.a2a_models import (
    SendMessageRequest, Message, Part, Role,
)
from langfuse import Langfuse

logger = configure_logger("orchestrator")

# Initialize Langfuse
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST
)

# Define State
class AgentState(TypedDict):
    query: str
    sql_result: str
    vector_result: str
    final_answer: str
    timings: dict
    trace_id: str  # Added for distributed tracing

# Define Nodes
import asyncio
import time


def _build_a2a_request(query: str, trace_id: str = None) -> dict:
    """Build an A2A SendMessageRequest payload for text-based agents."""
    request = SendMessageRequest(
        message=Message(
            role=Role.USER,
            parts=[Part.from_text(query)],
            metadata={"langfuse_trace_id": trace_id} if trace_id else {}
        )
    )
    return request.model_dump()


def _build_summarizer_a2a_request(query: str, sql_result: str, vector_result: str, trace_id: str = None) -> dict:
    """Build an A2A SendMessageRequest payload for the summarizer with structured data."""
    request = SendMessageRequest(
        message=Message(
            role=Role.USER,
            parts=[
                Part.from_data({
                    "query": query,
                    "sql_result": sql_result,
                    "vector_result": vector_result,
                })
            ],
            metadata={"langfuse_trace_id": trace_id} if trace_id else {}
        )
    )
    return request.model_dump()


def _extract_a2a_result(response_json: dict) -> str:
    """Extract the text result from an A2A Task response."""
    try:
        task = response_json.get("task", {})
        status = task.get("status", {})

        # Check if task failed
        if status.get("state") == "failed":
            error_msg = "Agent task failed"
            if status.get("message"):
                parts = status["message"].get("parts", [])
                if parts and parts[0].get("text"):
                    error_msg = parts[0]["text"]
            return f"Error: {error_msg}"

        # Extract from artifacts
        artifacts = task.get("artifacts", [])
        if artifacts:
            parts = artifacts[0].get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]

        return "No answer from agent"
    except Exception as e:
        logger.error("a2a_result_extraction_error", error=str(e))
        return f"Error extracting A2A result: {str(e)}"


# Helpers (Internal) — now using A2A protocol
async def _fetch_sql(query: str, base_url: str, trace_id: str = None):
    # Derive A2A endpoint from base service URL
    a2a_url = base_url.rsplit("/", 1)[0] + "/message:send"
    logger.info("orchestrator_calling_sql_a2a", url=a2a_url)
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            payload = _build_a2a_request(query, trace_id)
            resp = await client.post(a2a_url, json=payload, timeout=30.0)
            duration = time.time() - start_time
            if resp.status_code == 200:
                result = _extract_a2a_result(resp.json())
                return result, duration
            else:
                return f"Error: {resp.text}", duration
    except Exception as e:
        logger.error("sql_call_error", error=str(e))
        return f"Connection Error: {str(e)}", time.time() - start_time

async def _fetch_vector(query: str, base_url: str, trace_id: str = None):
    a2a_url = base_url.rsplit("/", 1)[0] + "/message:send"
    logger.info("orchestrator_calling_vector_a2a", url=a2a_url)
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            payload = _build_a2a_request(query, trace_id)
            resp = await client.post(a2a_url, json=payload, timeout=30.0)
            duration = time.time() - start_time
            if resp.status_code == 200:
                val = _extract_a2a_result(resp.json())
                if "I don't know" in val or "No relevant" in val:
                     val = "No relevant documents found."
                return val, duration
            else:
                return f"Error: {resp.text}", duration
    except Exception as e:
        logger.error("vector_call_error", error=str(e))
        return f"Connection Error: {str(e)}", time.time() - start_time

# Define Nodes
async def retrieve_data(state: AgentState):
    query = state["query"]
    
    # Initialize Root Trace
    trace = langfuse.trace(
        name="EPBA Orchestration Trace",
        input=query,
        user_id="anonymous", # Or pass from frontend
        session_id=settings.LANGFUSE_SESSION_ID
    )
    state["trace_id"] = trace.id
    
    with log_execution_time(logger, "parallel_retrieval"):
        # Run SQL and Vector in parallel via A2A
        sql_future = _fetch_sql(query, settings.SQL_AGENT_URL, trace.id)
        vector_future = _fetch_vector(query, settings.VECTOR_AGENT_URL, trace.id)
        
        # Results are now tuples (data, duration)
        (sql_res, sql_time), (vector_res, vector_time) = await asyncio.gather(sql_future, vector_future)
        
    return {
        "sql_result": sql_res,
        "vector_result": vector_res,
        "timings": {
            "sql_agent": sql_time,
            "vector_agent": vector_time
        },
        "trace_id": trace.id
    }

async def call_summarizer_agent(state: AgentState):
    query = state["query"]
    trace_id = state.get("trace_id")
    base_url = settings.SUMMARIZER_AGENT_URL
    a2a_url = base_url.rsplit("/", 1)[0] + "/message:send"
    start_time = time.time()
    
    current_timings = state.get("timings", {})
    
    with log_execution_time(logger, "call_summarizer"):
        logger.info("orchestrator_calling_summarizer_a2a", url=a2a_url)
        payload = _build_summarizer_a2a_request(
            query,
            state.get("sql_result", "N/A"),
            state.get("vector_result", "N/A"),
            trace_id
        )
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(a2a_url, json=payload, timeout=60.0)
                duration = time.time() - start_time
                if resp.status_code == 200:
                    result = _extract_a2a_result(resp.json())
                    current_timings["summarizer"] = duration
                    
                    # Finalize trace in Orchestrator
                    if trace_id:
                        langfuse.trace(id=trace_id).update(output=result)
                    
                    return {
                        "final_answer": result,
                        "timings": current_timings
                    }
                else:
                    current_timings["summarizer"] = duration
                    return {
                        "final_answer": f"Summarizer Error: {resp.text}",
                        "timings": current_timings
                    }
        except Exception as e:
            logger.error("summarizer_call_error", error=str(e))
            current_timings["summarizer"] = time.time() - start_time
            return {
                "final_answer": f"Summarizer Connection Error: {str(e)}",
                "timings": current_timings
            }

# Define Graph
def create_graph():
    workflow = StateGraph(AgentState)
    
    # We consolidate retrieval into one parallel node
    workflow.add_node("retrieve_data", retrieve_data)
    workflow.add_node("summarizer", call_summarizer_agent)
    
    workflow.set_entry_point("retrieve_data")
    workflow.add_edge("retrieve_data", "summarizer")
    workflow.add_edge("summarizer", END)
    
    return workflow.compile()
