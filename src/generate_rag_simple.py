import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag_retriever import retrieve_context

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.5"))

if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_KEY not set in .env")

MODEL_MAPPING_FILE = "config/model_mapping.json"
QUESTIONS_FILE = "respostas_atividade_open_questions.json"
OUTPUT_FILE = "respostas_rag_simple.json"

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


def load_model_mapping() -> dict[str, str]:
    with open(MODEL_MAPPING_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if v.strip():
            mapping[k] = v.strip()
    return mapping


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_output() -> list[dict]:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def call_model(model_id: str, prompt: str) -> str:
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model_id,
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
    print(f"Distance threshold: {DISTANCE_THRESHOLD}")
    print(f"Model mapping: {MODEL_MAPPING_FILE}")
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    mapping = load_model_mapping()
    if not mapping:
        print("[ERRO] Nenhum modelo mapeado. Preencha config/model_mapping.json primeiro.")
        return

    print(f"\nModelos mapeados: {len(mapping)}")
    for name, or_id in mapping.items():
        print(f"  {name} -> {or_id}")

    all_questions = load_questions()
    existing = load_existing_output()
    existing_pairs = {(item["question"], item["model_name"]) for item in existing}

    results = list(existing)
    total_processed = 0

    for model_name, openrouter_id in mapping.items():
        target_questions = [q for q in all_questions if q.get("model_name ", "").strip() == model_name]
        print(f"\n=== Modelo: {model_name} ({openrouter_id}) ===")
        print(f"  Perguntas no dataset: {len(target_questions)}")

        if not target_questions:
            print(f"  [AVISO] Nenhuma pergunta encontrada para '{model_name}'")
            continue

        model_processed = 0
        for i, item in enumerate(target_questions):
            question = item.get("question ", "").strip()
            golden_answer = item.get("golden_answer ", "").strip()

            if not question:
                continue

            if (question, model_name) in existing_pairs:
                print(f"  [{i+1}/{len(target_questions)}] SKIP: {question[:60]}...")
                continue

            model_processed += 1
            total_processed += 1
            print(f"\n  [{i+1}/{len(target_questions)}] Processing: {question[:60]}...")

            print(f"    Retrieving context...", end=" ", flush=True)
            _, all_chunks = retrieve_context(question)
            print(f"OK ({len(all_chunks)} chunks retrieved)")

            filtered_chunks = filter_chunks_by_threshold(all_chunks, DISTANCE_THRESHOLD)
            print(f"    Chunks passed threshold: {len(filtered_chunks)}/{len(all_chunks)}")

            if filtered_chunks:
                context = "\n\n---\n\n".join(
                    f"[Fonte: {c['source']}]\n{c['content']}" for c in filtered_chunks
                )
                prompt = PROMPT_WITH_CONTEXT.format(context=context, question=question)
            else:
                prompt = PROMPT_NO_CONTEXT.format(question=question)

            print(f"    Calling model...", end=" ", flush=True)
            answer = call_model(openrouter_id, prompt)
            print(f"OK")

            result = {
                "question": question,
                "golden_answer": golden_answer,
                "model_name": model_name,
                "model_answer": answer,
            }

            results.append(result)
            existing_pairs.add((question, model_name))

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            time.sleep(0.5)

        print(f"\n  Modelo {model_name}: {model_processed} perguntas processadas")

    print(f"\n=== RAG Generation Complete ===")
    print(f"Total perguntas processadas: {total_processed}")
    print(f"Total respostas salvas: {len(results)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()