import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag_retriever import retrieve_context

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
RAG_MODEL = os.getenv("RAG_MODEL")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME")
DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.5"))

if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_KEY not set in .env")
if not RAG_MODEL:
    raise ValueError("RAG_MODEL not set in .env. Example: google/gemma-3-4b-it")
if not RAG_MODEL_NAME:
    raise ValueError("RAG_MODEL_NAME not set in .env. Example: Gemma-3-4B")

QUESTIONS_FILE = "respostas_atividade_open_questions.json"
OUTPUT_FILE = "respostas_rag.json"

PROMPT_WITH_CONTEXT = """You are a helpful medical assistant. Provide accurate and concise answers.

[CONTEXT - Medical Reference Documents]
{context}

[QUESTION]
{question}

[INSTRUCTIONS]
Use the context above if it is relevant to answer the question. If the context is irrelevant or does not address the question, ignore it completely and answer using your own medical knowledge.

Be direct, accurate, and specific."""

PROMPT_NO_CONTEXT = """You are a helpful medical assistant. Provide accurate and concise answers.

[QUESTION]
{question}

[INSTRUCTIONS]
Answer the question using your medical knowledge. Be direct, accurate, and specific."""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)


def load_existing_output() -> list[dict]:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def call_model(prompt: str) -> str:
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=RAG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [ERRO] Attempt {attempt + 1}/3: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return "[FAILED AFTER 3 RETRIES]"


def filter_chunks_by_threshold(chunks: list[dict], threshold: float) -> list[dict]:
    return [c for c in chunks if c.get("score", 1.0) < threshold]


def main():
    print(f"=== RAG Generation (Simple) ===")
    print(f"OpenRouter model: {RAG_MODEL}")
    print(f"JSON model name: {RAG_MODEL_NAME}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}")
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    target_questions = [q for q in all_questions if q.get("model_name ", "").strip() == RAG_MODEL_NAME]
    print(f"\nFound {len(target_questions)} questions for model '{RAG_MODEL_NAME}'")

    if not target_questions:
        print(f"[ERRO] No questions found for model '{RAG_MODEL_NAME}'. Check RAG_MODEL_NAME in .env")
        return

    existing = load_existing_output()
    existing_pairs = {(item["question"], item["model_name"]) for item in existing}

    results = list(existing)
    processed = 0

    for i, item in enumerate(target_questions):
        question = item.get("question ", "").strip()
        golden_answer = item.get("golden_answer ", "").strip()

        if not question:
            continue

        if (question, RAG_MODEL) in existing_pairs:
            print(f"[{i+1}/{len(target_questions)}] SKIP (already answered): {question[:60]}...")
            continue

        processed += 1
        print(f"\n[{i+1}/{len(target_questions)}] Processing: {question[:60]}...")

        print(f"  Retrieving context...", end=" ", flush=True)
        _, all_chunks = retrieve_context(question)
        print(f"OK ({len(all_chunks)} chunks retrieved)")

        filtered_chunks = filter_chunks_by_threshold(all_chunks, DISTANCE_THRESHOLD)
        print(f"  Chunks passed threshold: {len(filtered_chunks)}/{len(all_chunks)}")

        if filtered_chunks:
            context = "\n\n---\n\n".join(
                f"[Fonte: {c['source']}]\n{c['content']}" for c in filtered_chunks
            )
            prompt = PROMPT_WITH_CONTEXT.format(context=context, question=question)
            context_used = True
        else:
            prompt = PROMPT_NO_CONTEXT.format(question=question)
            context = ""
            context_used = False

        print(f"  Calling model...", end=" ", flush=True)
        answer = call_model(prompt)
        print(f"OK")

        top_chunk_distance = filtered_chunks[0].get("score") if filtered_chunks else (all_chunks[0].get("score") if all_chunks else None)

        result = {
            "question": question,
            "golden_answer": golden_answer,
            "model_name": RAG_MODEL,
            "model_answer": answer,
            "context_used": context_used,
            "context_summary": {
                "chunks_passed_threshold": len(filtered_chunks),
                "top_chunk_distance": top_chunk_distance,
                "threshold": DISTANCE_THRESHOLD,
                "sources": [c["source"] for c in filtered_chunks] if filtered_chunks else [],
            },
            "retrieved_chunks": [
                {
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "distance": chunk.get("score"),
                    "rank": idx + 1,
                }
                for idx, chunk in enumerate(all_chunks)
            ],
        }

        results.append(result)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    print(f"\n=== RAG Generation Complete ===")
    print(f"Questions processed: {processed}")
    print(f"Total answers saved: {len(results)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
