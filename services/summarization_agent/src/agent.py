import os
from shared.src.config import settings
from shared.src.logger import configure_logger, log_execution_time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from shared.src.evaluation import run_deepeval
import threading

import yaml

logger = configure_logger("summarization_agent")

# Initialize Langfuse
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST
)

def load_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
    try:
        with open(prompt_path, "r") as f:
            data = yaml.safe_load(f)
            return data.get("summarization_prompt", "")
    except Exception as e:
        logger.error("failed_to_load_prompts", error=str(e))
        return ""

class SummarizationService:
    def __init__(self):
        self.llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=settings.LLM_TEMPERATURE)
        self.prompt_template = load_prompt()

    def summarize(self, query: str, sql_res: str, vec_res: str, trace_id: str = None, parent_observation_id: str = None) -> str:
        with log_execution_time(logger, "summarization_execution"):
            logger.info("received_summarization_request", query=query, trace_id=trace_id)
            
            # Start Langfuse Generation Span
            generation = None
            if trace_id:
                generation = langfuse.generation(
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    name="Summarization Agent Synthesis",
                    model=settings.LLM_MODEL_NAME,
                    model_parameters={"temperature": settings.LLM_TEMPERATURE},
                    input={"query": query, "sql_result": sql_res, "vector_result": vec_res},
                )
            
            # Fallback to a basic prompt if the yaml file failed to load
            if not self.prompt_template:
                prompt_str = f"User asked: {query}\nSQL: {sql_res}\nVector: {vec_res}\nSummarize concisely."
            else:
                prompt_str = self.prompt_template.format(query=query, sql_res=sql_res, vec_res=vec_res)
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt_str)])
                logger.info("summarization_success")
                
                # Trigger async DeepEval evaluation via a separate thread
                try:
                    # Use threading to ensure we don't block the FastAPI event loop
                    eval_thread = threading.Thread(
                        target=run_deepeval,
                        args=(query, response.content, [sql_res, vec_res], trace_id)
                    )
                    eval_thread.start()
                except Exception as eval_e:
                    logger.error("failed_to_trigger_eval", error=str(eval_e))
                
                if generation:
                    generation.update(output=response.content)
                
                return response.content
            except Exception as e:
                logger.error("summarization_failed", error=str(e))
                if generation:
                    generation.update(level="ERROR", status_message=str(e))
                return f"Error generating summary: {str(e)}"

def get_summarizer():
    return SummarizationService()
