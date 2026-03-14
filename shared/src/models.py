from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AgentRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None
    context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    answer: str
    source_documents: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = 0.0

class SummarizationRequest(BaseModel):
    query: str
    sql_result: str
    vector_result: str
