"""
A2A Protocol Data Models (Google Agent-to-Agent Protocol RC v1.0)

Pydantic models for Agent Cards, Tasks, Messages, Parts, and Artifacts
following the A2A specification: https://a2a-protocol.org/latest/specification/
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Agent Card Models ───────────────────────────────────────────────────────

class AgentSkill(BaseModel):
    """Represents a distinct capability or function that an agent can perform."""
    id: str
    name: str
    description: str
    tags: list[str] = []
    examples: list[str] = []
    inputModes: list[str] = ["text/plain"]
    outputModes: list[str] = ["text/plain"]


class AgentProvider(BaseModel):
    """Represents the service provider of an agent."""
    organization: str
    url: Optional[str] = None


class AgentCapabilities(BaseModel):
    """Defines optional capabilities supported by an agent."""
    streaming: bool = False
    pushNotifications: bool = False


class AgentInterface(BaseModel):
    """Declares a URL + transport for interacting with the agent."""
    url: str
    protocolBinding: str = "HTTP+JSON"
    protocolVersion: str = "1.0"


class AgentCard(BaseModel):
    """
    A self-describing manifest for an agent.
    Served at GET /.well-known/agent.json
    """
    name: str
    description: str
    version: str = "1.0.0"
    provider: Optional[AgentProvider] = None
    capabilities: AgentCapabilities = AgentCapabilities()
    skills: list[AgentSkill] = []
    defaultInputModes: list[str] = ["text/plain"]
    defaultOutputModes: list[str] = ["text/plain"]
    supportedInterfaces: list[AgentInterface] = []
    documentationUrl: Optional[str] = None
    iconUrl: Optional[str] = None


# ─── A2A Task Protocol Models ────────────────────────────────────────────────

class Role(str, Enum):
    """Defines the sender of a message."""
    USER = "user"
    AGENT = "agent"


class TaskState(str, Enum):
    """Defines the possible lifecycle states of a Task."""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    INPUT_REQUIRED = "input-required"


class Part(BaseModel):
    """
    Represents a container for a section of communication content.
    A Part MUST contain exactly one of: text, data.
    """
    text: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    mediaType: Optional[str] = None

    @classmethod
    def from_text(cls, text: str) -> Part:
        return cls(text=text, mediaType="text/plain")

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Part:
        return cls(data=data, mediaType="application/json")


class Message(BaseModel):
    """One unit of communication between client and server."""
    messageId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    parts: list[Part]
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class TaskStatus(BaseModel):
    """Container for the status of a task."""
    state: TaskState
    message: Optional[Message] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Artifact(BaseModel):
    """Represents task outputs."""
    artifactId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    parts: list[Part] = []
    metadata: Optional[dict[str, Any]] = None


class A2ATask(BaseModel):
    """
    The core unit of action for A2A.
    Returned by message:send operations.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    artifacts: list[Artifact] = []
    history: list[Message] = []
    metadata: Optional[dict[str, Any]] = None


class SendMessageRequest(BaseModel):
    """Incoming request for POST /message:send"""
    message: Message
    configuration: Optional[dict[str, Any]] = None


class SendMessageResponse(BaseModel):
    """Response wrapper for message:send — returns a Task."""
    task: A2ATask


# ─── Helpers ──────────────────────────────────────────────────────────────────

def create_completed_task(
    result_text: str,
    artifact_name: str = "result",
    metadata: dict[str, Any] | None = None,
) -> A2ATask:
    """Helper to create a completed A2A Task with a text result."""
    return A2ATask(
        status=TaskStatus(state=TaskState.COMPLETED),
        artifacts=[
            Artifact(
                name=artifact_name,
                parts=[Part.from_text(result_text)],
            )
        ],
        metadata=metadata,
    )


def create_failed_task(error_message: str) -> A2ATask:
    """Helper to create a failed A2A Task."""
    return A2ATask(
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role=Role.AGENT,
                parts=[Part.from_text(error_message)],
            ),
        ),
    )
