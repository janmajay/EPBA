import os
from shared.src.config import settings
from shared.src.logger import configure_logger, log_execution_time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langfuse import Langfuse
from shared.src.evaluation import run_deepeval_retrieval
import threading
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from flashrank import Ranker, RerankRequest
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class CustomEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        all_docs = []
        seen = set()
        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            for d in docs:
                # Deduplicate by page_content
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    all_docs.append(d)
        return all_docs

logger = configure_logger("vector_agent")

class FlashrankRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    ranker: Ranker = Field(default_factory=lambda: Ranker(model_name="ms-marco-TinyBERT-L-2-v2"))
    score_threshold: float = 0.50

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
            
        passages = [{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(docs)]
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerankrequest)
        
        final_docs = []
        for r in results:
            # Drop any chunk that falls below our threshold, eliminating all noise
            if r.get("score", 0.0) >= self.score_threshold:
                idx = r["id"]
                doc = docs[idx]
                doc.metadata["flashrank_score"] = r.get("score", 0.0)
                final_docs.append(doc)
                
        return final_docs

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
        
        # User requested: Base retrieval K directly from settings.yaml (10), rerank dynamic
        base_retriever = self.db.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
        
        # Build BM25 index over all current ChromaDB documents
        all_data = self.db.get()
        if all_data and all_data.get("documents"):
            print(f"Building BM25 index over {len(all_data['documents'])} chunks...")
            bm25_docs = [Document(page_content=txt, metadata=meta or {}) for txt, meta in zip(all_data['documents'], all_data['metadatas'])]
            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = settings.VECTOR_SEARCH_K
            
            ensemble_retriever = CustomEnsembleRetriever(
                retrievers=[bm25_retriever, base_retriever]
            )
        else:
            # Fallback if DB is completely empty at startup
            ensemble_retriever = base_retriever

        # Note: FlashrankRetriever now evaluates the combined ensemble
        self.retriever = FlashrankRetriever(base_retriever=ensemble_retriever)
        
        self.llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=settings.LLM_TEMPERATURE)
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever, return_source_documents=True)

    def query(self, user_query: str, trace_id: str = None, parent_observation_id: str = None) -> str:
        with log_execution_time(logger, "vector_search_execution"):
            logger.info("received_query", query=user_query, trace_id=trace_id)
            
            # Start Langfuse Generation Span
            generation = None
            if trace_id:
                generation = langfuse.generation(
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    name="Vector Agent RetrievalQA",
                    model=settings.LLM_MODEL_NAME,
                    model_parameters={"temperature": settings.LLM_TEMPERATURE},
                    input={"query": user_query},
                )
                
            try:
                result = self.qa_chain.invoke({"query": user_query})
                output = result["result"]
                source_documents = result.get("source_documents", [])
                contexts = [doc.page_content for doc in source_documents]
                
                retrieved_chunks_meta = []
                for doc in source_documents:
                    retrieved_chunks_meta.append({
                        "text": doc.page_content,
                        "score": doc.metadata.get("flashrank_score", "N/A"),
                        "source": doc.metadata.get("source", "Unknown")
                    })
                
                logger.info("query_success", output_preview=output[:100])
                
                # Trigger async DeepEval retrieval evaluation via a separate thread
                try:
                    # Use threading to ensure we don't block the FastAPI event loop
                    eval_thread = threading.Thread(
                        target=run_deepeval_retrieval,
                        args=(user_query, output, contexts, trace_id)
                    )
                    eval_thread.start()
                except Exception as eval_e:
                    logger.error("failed_to_trigger_eval", error=str(eval_e))
                
                if generation:
                    generation.update(
                        output=output,
                        metadata={"retrieved_chunks": retrieved_chunks_meta}
                    )
                
                return output
            except Exception as e:
                logger.error("query_failed", error=str(e))
                if generation:
                    generation.update(level="ERROR", status_message=str(e))
                return f"Error executing Vector search: {str(e)}"

def get_vector_agent(persist_dir: str = None):
    final_dir = persist_dir or settings.VECTOR_STORE_DIR
    return VectorAgentService(final_dir)
