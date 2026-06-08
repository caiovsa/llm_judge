import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

with open("config/rag_config.json") as f:
    RAG_CFG = json.load(f)

CHROMA_DIR = RAG_CFG["chroma_persist_dir"]
COLLECTION_NAME = RAG_CFG["chroma_collection"]
TOP_K = RAG_CFG["top_k"]
EMBEDDING_MODEL = RAG_CFG["embedding_model"]

_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME not in existing:
        _collection = None
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )

    _collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    return _collection


def retrieve_context(question: str, options: str = "", top_k: int = None) -> tuple[str, list[dict]]:
    col = get_collection()
    if col is None:
        return "(Nenhum documento carregado na base vetorial.)", []

    query = question
    if options:
        query = f"{question}\n\nAlternativas:\n{options}"

    k = top_k if top_k is not None else TOP_K
    results = col.query(query_texts=[query], n_results=k)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": results["distances"][0][i] if results.get("distances") else None,
        })

    formatted = "\n\n---\n\n".join(
        f"[Fonte: {c['source']}]\n{c['content']}" for c in chunks
    )

    return formatted, chunks


def retrieve_context_for_mcq(question: str, options_dict: dict, top_k: int = None) -> tuple[str, list[dict]]:
    options_str = "\n".join(f"{k}: {v}" for k, v in sorted(options_dict.items()))
    return retrieve_context(question, options_str, top_k=top_k)


if __name__ == "__main__":
    print("=== RAG Retriever (teste) ===")
    ctx, chunks = retrieve_context("What is the treatment for hypertension?")
    print(f"{len(chunks)} chunks recuperados")
    print(ctx[:500])