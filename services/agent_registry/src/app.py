"""
Agent Registry Service — Centralized agent discovery for EPBA.

Polls each known agent's /.well-known/agent.json endpoint,
stores Agent Cards in memory, and exposes them via REST.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.src.config import settings
from shared.src.logger import configure_logger

logger = configure_logger("agent_registry")

# ─── In-Memory Agent Store ────────────────────────────────────────────────────

class RegisteredAgent(BaseModel):
    """An agent card enriched with registry metadata."""
    agent_card: dict          # Raw Agent Card JSON
    base_url: str             # Service base URL
    status: str = "unknown"   # online / offline / unknown
    last_checked: Optional[float] = None
    response_time_ms: Optional[float] = None


# Global agent store: name → RegisteredAgent
_agent_store: dict[str, RegisteredAgent] = {}

# Known agent service URLs (derived from settings)
def _get_agent_urls() -> dict[str, str]:
    """Map agent display names to their base URLs (without path)."""
    return {
        "SQL Agent": settings.SQL_AGENT_URL.rsplit("/", 1)[0],
        "Vector Agent": settings.VECTOR_AGENT_URL.rsplit("/", 1)[0],
        "Summarization Agent": settings.SUMMARIZER_AGENT_URL.rsplit("/", 1)[0],
        "Orchestrator": settings.ORCHESTRATOR_URL.rsplit("/", 1)[0],
    }


async def _fetch_agent_card(name: str, base_url: str) -> RegisteredAgent:
    """Fetch a single agent's Agent Card."""
    card_url = f"{base_url}/.well-known/agent.json"
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(card_url, timeout=5.0)
            elapsed_ms = (time.time() - start) * 1000
            if resp.status_code == 200:
                card_data = resp.json()
                logger.info("agent_card_fetched", agent=name, url=card_url)
                return RegisteredAgent(
                    agent_card=card_data,
                    base_url=base_url,
                    status="online",
                    last_checked=time.time(),
                    response_time_ms=round(elapsed_ms, 1),
                )
            else:
                logger.warning("agent_card_fetch_failed", agent=name, status=resp.status_code)
                return RegisteredAgent(
                    agent_card={"name": name, "description": "Agent card unavailable"},
                    base_url=base_url,
                    status="offline",
                    last_checked=time.time(),
                )
    except Exception as e:
        logger.error("agent_card_fetch_error", agent=name, error=str(e))
        return RegisteredAgent(
            agent_card={"name": name, "description": f"Connection error: {str(e)}"},
            base_url=base_url,
            status="offline",
            last_checked=time.time(),
        )


async def refresh_all_agents():
    """Fetch Agent Cards from all known agents."""
    global _agent_store
    agent_urls = _get_agent_urls()
    tasks = [_fetch_agent_card(name, url) for name, url in agent_urls.items()]
    results = await asyncio.gather(*tasks)
    for name, result in zip(agent_urls.keys(), results):
        _agent_store[name] = result
    logger.info("agent_registry_refreshed", agent_count=len(_agent_store))


async def _periodic_refresh(interval: int = 30):
    """Background loop to refresh agent cards periodically."""
    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_all_agents()
        except Exception as e:
            logger.error("periodic_refresh_error", error=str(e))


# ─── FastAPI Application ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initial fetch + start background refresh."""
    # Wait a moment for other services to start
    await asyncio.sleep(2)
    await refresh_all_agents()
    # Start background refresh task
    task = asyncio.create_task(_periodic_refresh(30))
    yield
    task.cancel()


app = FastAPI(
    title="EPBA Agent Registry",
    description="Centralized agent discovery service using Google A2A protocol",
    lifespan=lifespan,
)


@app.get("/agents")
async def list_agents():
    """List all registered agent cards with their status."""
    agents = []
    for name, reg in _agent_store.items():
        agents.append({
            "name": name,
            "status": reg.status,
            "base_url": reg.base_url,
            "last_checked": reg.last_checked,
            "response_time_ms": reg.response_time_ms,
            "agent_card": reg.agent_card,
        })
    return {"agents": agents, "count": len(agents)}


@app.get("/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get a specific agent's card by name."""
    # Try exact match first
    if agent_name in _agent_store:
        reg = _agent_store[agent_name]
        return {
            "name": agent_name,
            "status": reg.status,
            "base_url": reg.base_url,
            "last_checked": reg.last_checked,
            "response_time_ms": reg.response_time_ms,
            "agent_card": reg.agent_card,
        }
    # Try case-insensitive match
    for name, reg in _agent_store.items():
        if name.lower().replace(" ", "-") == agent_name.lower().replace(" ", "-"):
            return {
                "name": name,
                "status": reg.status,
                "base_url": reg.base_url,
                "agent_card": reg.agent_card,
            }
    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")


@app.post("/agents/refresh")
async def trigger_refresh():
    """Manually trigger a refresh of all agent cards."""
    await refresh_all_agents()
    return {"message": "Agent cards refreshed", "count": len(_agent_store)}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "registered_agents": len(_agent_store),
        "online_agents": sum(1 for a in _agent_store.values() if a.status == "online"),
    }
