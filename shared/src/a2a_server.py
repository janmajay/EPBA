"""
Reusable A2A Server Router for FastAPI

Provides a factory function that creates a FastAPI APIRouter with:
  - GET /.well-known/agent.json  → serves the agent's Agent Card
  - POST /message:send           → processes a message via the agent's callback

Usage in any agent's app.py:
    from shared.src.a2a_server import create_a2a_router
    from shared.src.a2a_models import AgentCard, ...

    agent_card = AgentCard(name="My Agent", ...)

    async def process_message(query: str) -> str:
        return await my_agent_logic(query)

    a2a_router = create_a2a_router(agent_card, process_message)
    app.include_router(a2a_router)
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Awaitable, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.src.a2a_models import (
    AgentCard,
    A2ATask,
    Artifact,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    TaskState,
    TaskStatus,
    create_completed_task,
    create_failed_task,
)

# Type alias for the agent's processing function
# It receives the raw query string and returns a result string
ProcessMessageFn = Callable[[str], Awaitable[str]]

# Extended version that receives the full message for structured data
ProcessFullMessageFn = Callable[[Message], Awaitable[str]]


def create_a2a_router(
    agent_card: AgentCard,
    process_message: Optional[ProcessMessageFn] = None,
    process_full_message: Optional[ProcessFullMessageFn] = None,
) -> APIRouter:
    """
    Create a FastAPI APIRouter with A2A protocol endpoints.

    Args:
        agent_card: The AgentCard to serve at /.well-known/agent.json
        process_message: Simple callback: async (query: str) -> str
        process_full_message: Advanced callback: async (message: Message) -> str
            Use this when the agent needs access to structured data parts.
            If both are provided, process_full_message takes precedence.
    """
    router = APIRouter(tags=["A2A Protocol"])

    @router.get("/.well-known/agent.json", response_model=AgentCard)
    async def get_agent_card():
        """Serve the agent's Agent Card for discovery."""
        return agent_card

    @router.post("/message:send", response_model=SendMessageResponse)
    async def send_message(request: SendMessageRequest):
        """
        A2A message:send — process a message and return a completed Task.

        Follows the HTTP+JSON/REST binding (Section 11) of the A2A spec.
        """
        try:
            # Extract query from message parts
            incoming_message = request.message

            if process_full_message is not None:
                # Use the full-message handler (for agents needing structured data)
                result_text = await process_full_message(incoming_message)
            elif process_message is not None:
                # Extract plain text from the first text part
                query = _extract_text_from_message(incoming_message)
                result_text = await process_message(query)
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No message processor configured"},
                )

            task = create_completed_task(
                result_text=result_text,
                artifact_name=f"{agent_card.name} Result",
            )

            # Add the incoming message to task history
            task.history = [incoming_message]

            return SendMessageResponse(task=task)

        except Exception as e:
            task = create_failed_task(
                error_message=f"Agent processing failed: {str(e)}"
            )
            task.history = [request.message]

            return SendMessageResponse(task=task)

    return router


def _extract_text_from_message(message: Message) -> str:
    """Extract the text content from message parts."""
    text_parts = []
    for part in message.parts:
        if part.text is not None:
            text_parts.append(part.text)
    return " ".join(text_parts) if text_parts else ""
