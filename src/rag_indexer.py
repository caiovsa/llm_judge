import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

with open("config/rag_config.json") as f:
    RAG_CFG = json.load(f)

RAG_DIR = Path("docs/rag")
CHROMA_DIR = RAG_CFG["chroma_persist_dir"]
COLLECTION_NAME = RAG_CFG["chroma_collection"]
CHUNK_SIZE = RAG_CFG["chunk_size"]
CHUNK_OVERLAP = RAG_CFG["chunk_overlap"]
EMBEDDING_MODEL = RAG_CFG["embedding_model"]


def extract_text_from_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        if end < text_len:
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start:
                end = boundary + 1

        content = text[start:end].strip()
        if content:
            chunks.append({
                "content": content,
                "metadata": {"source": source, "chunk_index": len(chunks), "start_char": start},
            })

        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        new_start = start + step
        if new_start <= start:
            break
        start = new_start

    return chunks


def get_embedding_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)


def build_chroma_collection(client: OpenAI):
    os.makedirs(CHROMA_DIR, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=client.api_key,
        model_name=EMBEDDING_MODEL,
    )

    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return chroma_client, collection


def index_pdfs():
    pdf_files = sorted(RAG_DIR.glob("*.pdf")) + sorted(RAG_DIR.glob("*.PDF"))
    if not pdf_files:
        print("[RAG INDEXER] Nenhum PDF encontrado em docs/rag/. Pulando indexação.")
        return

    client = get_embedding_client()
    _, collection = build_chroma_collection(client)

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  Lendo: {pdf_path.name}")
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  [ERRO] Falha ao extrair texto de {pdf_path.name}: {e}")
            continue

        chunks = chunk_text(text, pdf_path.name)
        print(f"    {len(chunks)} chunks gerados")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[RAG INDEXER] Nenhum chunk gerado. Abortando.")
        return

    ids = [str(i) for i in range(len(all_chunks))]
    documents = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    for i in range(0, len(documents), RAG_CFG["batch_size"]):
        batch_end = min(i + RAG_CFG["batch_size"], len(documents))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"    Indexados {batch_end}/{len(documents)} chunks")

    print(f"\n[RAG INDEXER] Concluído. {len(all_chunks)} chunks indexados em '{COLLECTION_NAME}'")


if __name__ == "__main__":
    print("=== RAG Indexer ===")
    index_pdfs()