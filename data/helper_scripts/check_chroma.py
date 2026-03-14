import os
import sys
from collections import Counter

# Add project root to sys.path so we can import shared
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from shared.src.config import settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Resolve relative paths against project root so script works from any directory
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, settings.VECTOR_STORE_DIR) if not os.path.isabs(settings.VECTOR_STORE_DIR) else settings.VECTOR_STORE_DIR
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME

def check_chroma():
    print(f"Checking ChromaDB at: {VECTOR_STORE_DIR}")
    
    if not os.path.exists(VECTOR_STORE_DIR):
        print("Error: Vector store directory not found.")
        return

    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    try:
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_function)
        collection = db._collection
        
        total_count = collection.count()
        print(f"\nTotal Embeddings: {total_count}")
        
        if total_count == 0:
            return

        # Fetch all metadata to aggregate counts
        # Warning: For very large DBs, this might be slow, but fine for this scale.
        all_data = collection.get(include=['metadatas'])
        metadatas = all_data['metadatas']
        
        # Count by 'source'
        source_counts = Counter()
        for meta in metadatas:
            if meta and 'source' in meta:
                # Extract basename for cleaner output
                source_name = os.path.basename(meta['source'])
                source_counts[source_name] += 1
            else:
                source_counts["Unknown Source"] += 1
        
        print("\n--- Embeddings per Patient Report ---")
        for source, count in source_counts.most_common():
            print(f"- {source}: {count} chunks")
            
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")

if __name__ == "__main__":
    check_chroma()
