import os
from shared.src.config import settings
from shared.src.logger import configure_logger, log_execution_time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langfuse import Langfuse

logger = configure_logger("vector_agent")

# Initialize Langfuse
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST
)

class VectorAgentService:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        logger.info("initializing_vector_agent", persist_dir=persist_dir, embedding_model=settings.EMBEDDING_MODEL_NAME)
        self.embedding_function = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
        self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embedding_function)
        self.retriever = self.db.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
        self.llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=settings.LLM_TEMPERATURE)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def query(self, user_query: str, trace_id: str = None) -> str:
        with log_execution_time(logger, "vector_search_execution"):
            logger.info("received_query", query=user_query, trace_id=trace_id)
            
            # Start Langfuse Generation Span
            generation = None
            if trace_id:
                generation = langfuse.generation(
                    trace_id=trace_id,
                    name="Vector Agent RetrievalQA",
                    model=settings.LLM_MODEL_NAME,
                    model_parameters={"temperature": settings.LLM_TEMPERATURE},
                    input={"query": user_query},
                )
                
            try:
                result = self.qa_chain.invoke({"query": user_query})
                output = result["result"]
                logger.info("query_success", output_preview=output[:100])
                
                if generation:
                    generation.update(output=output)
                
                return output
            except Exception as e:
                logger.error("query_failed", error=str(e))
                if generation:
                    generation.update(level="ERROR", status_message=str(e))
                return f"Error executing Vector search: {str(e)}"

def get_vector_agent(persist_dir: str = None):
    final_dir = persist_dir or settings.VECTOR_STORE_DIR
    return VectorAgentService(final_dir)
