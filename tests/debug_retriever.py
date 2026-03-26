import sys
import os

sys.path.append(os.getcwd())

from services.vector_agent.src.agent import get_vector_agent
from flashrank import Ranker, RerankRequest

def main():
    agent = get_vector_agent()
    query = "Give me the medical history and details about Jack Barton"
    
    print(f"Testing Query: {query}")
    
    # Run the raw Chroma retrieval
    docs = agent.retriever.base_retriever.invoke(query)
    print(f"\n[Raw Chroma Retrieved {len(docs)} docs]")
    
    # Run FlashRank manually to see the exact scores
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    passages = [{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(docs)]
    rerankrequest = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerankrequest)
    
    print("\n[FlashRank Scores]")
    for r in results:
        print(f"ID: {r['id']} | Score: {r.get('score', 0):.4f} | Preview: {r['text'][:50]}...")

if __name__ == "__main__":
    main()
