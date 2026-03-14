import os
import sys

# Add project root to sys.path so we can import packages from 'services' if needed
# But now we rely on installed 'shared' lib or pythonpath setting in Docker
sys.path.append(os.getcwd())

import glob
import shutil
from shared.src.config import settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def load_documents(source_path: str):
    documents = []
    
    # Check if source_path is a directory or file
    if os.path.isdir(source_path):
        print(f"Scanning directory: {source_path}")
        # Load all .txt files
        for file_path in glob.glob(os.path.join(source_path, "**/*.txt"), recursive=True):
             print(f"Loading TEXT: {file_path}")
             try:
                 loader = TextLoader(file_path)
                 documents.extend(loader.load())
             except Exception as e:
                 print(f"Error loading {file_path}: {e}")

        # Load all .pdf files
        for file_path in glob.glob(os.path.join(source_path, "**/*.pdf"), recursive=True):
             print(f"Loading PDF: {file_path}")
             try:
                 loader = PyPDFLoader(file_path)
                 documents.extend(loader.load())
             except Exception as e:
                 print(f"Error loading {file_path}: {e}")
                 
    elif os.path.isfile(source_path):
        if source_path.endswith(".pdf"):
             print(f"Loading PDF: {source_path}")
             loader = PyPDFLoader(source_path)
             documents.extend(loader.load())
        else:
             print(f"Loading TEXT: {source_path}")
             loader = TextLoader(source_path)
             documents.extend(loader.load())
             
    return documents

def ingest_documents(source_path: str, persist_dir: str):
    # Always clean clean the existing vector store first to recreate from scratch
    if os.path.exists(persist_dir):
        print(f"Removing existing vector store at {persist_dir} to recreate from scratch...")
        shutil.rmtree(persist_dir)
        
    docs = load_documents(source_path)
    
    if not docs:
        print("No documents found to ingest.")
        return

    print(f"Loaded {len(docs)} documents.")
    
    # Use Recursive for better chunking of mixed content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    
    print(f"Split into {len(split_docs)} chunks.")
    
    print(f"Using embedding model: {settings.EMBEDDING_MODEL_NAME}")
    embedding_function = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
    
    print(f"Creating vector store in {persist_dir}...")
    db = Chroma.from_documents(split_docs, embedding_function, persist_directory=persist_dir)
    print("Ingestion complete.")


if __name__ == "__main__":
    # Use path from config which handles local vs docker resolution
    # SOURCE_PATH and PERSIST_DIR are now centrally managed in config.py
    # We still allow env var override but default to config
    source = settings.VECTOR_SOURCE_PATH
    persist = settings.VECTOR_STORE_DIR
    ingest_documents(source, persist)
