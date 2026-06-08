import json
import os
import re
import sys
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from openai import OpenAI

from rag_retriever import retrieve_context, retrieve_context_for_mcq

load_dotenv()

with open("config/model_mapping.json") as f:
    MODEL_MAP: dict[str, str] = json.load(f)

with open("config/rag_config.json") as f:
    RAG_CFG = json.load(f)

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "llm_judge"),
    user=os.getenv("DB_USER", "llm_user"),
    password=os.getenv("DB_PASSWORD", "llm_pass"),
)

client = OpenAI()

PROMPT_ABERTA_TEMPLATE = """[CONTEXTO MÉDICO]
{contexto}

[PERGUNTA]
{question}

[INSTRUÇÃO]
Responda à pergunta com base no seu conhecimento médico e no contexto fornecido acima.
Seja direto, preciso e específico na sua resposta."""

PROMPT_MCQ_TEMPLATE = """[CONTEXTO MÉDICO]
{contexto}

[PERGUNTA]
{question}

[ALTERNATIVAS]
{options}

[INSTRUÇÃO]
Com base no contexto médico fornecido, analise as alternativas e responda APENAS com a letra da alternativa correta (A, B, C, D ou E)."""


def ensure_rag_columns(cur):
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'respostas_atividade_1' AND column_name = 'rag'
    """)
    if not cur.fetchone():
        cur.execute("ALTER TABLE respostas_atividade_1 ADD COLUMN rag BOOLEAN NOT NULL DEFAULT FALSE")
        cur.execute("ALTER TABLE respostas_atividade_1 ADD COLUMN contexto_rag TEXT")
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'avaliacoes_juiz' AND column_name = 'rag'
    """)
    if not cur.fetchone():
        cur.execute("ALTER TABLE avaliacoes_juiz ADD COLUMN rag BOOLEAN NOT NULL DEFAULT FALSE")


def get_model_mapping() -> dict:
    missing = [k for k, v in MODEL_MAP.items() if not v]
    if missing:
        print(f"[AVISO] Modelos sem mapeamento: {missing}")
        print("Preencha config/model_mapping.json antes de rodar o generate_rag.")
    return {k: v for k, v in MODEL_MAP.items() if v}


def parse_model_name(raw: str) -> tuple[str, str]:
    m = re.match(r'^([A-Za-z][A-Za-z]*)[ ](.+)$', raw)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^([A-Za-z][A-Za-z]*)-(.+)$', raw)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^([A-Za-z]+)[_](.+)$', raw)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^([A-Za-z]+)(\d.*)$', raw)
    if m:
        return m.group(1), m.group(2)
    return raw, "unknown"


def get_candidate_models(cur) -> list[dict]:
    cur.execute("""
        SELECT id_modelo, nome_modelo, versao,
               nome_modelo || ' ' || versao AS full_name
        FROM modelos
        WHERE tipo = 'candidato'
        ORDER BY nome_modelo
    """)
    return cur.fetchall()


def get_open_questions(cur) -> list[dict]:
    cur.execute("""
        SELECT p.id_pergunta, p.enunciado, p.resposta_ouro
        FROM perguntas p
        JOIN datasets d ON d.id_dataset = p.id_dataset
        WHERE d.nome_dataset = 'Itaymanes K-QA'
        ORDER BY p.id_pergunta
    """)
    return cur.fetchall()


def get_mcq_questions(cur) -> list[dict]:
    cur.execute("""
        SELECT p.id_pergunta, p.enunciado, p.resposta_ouro
        FROM perguntas p
        JOIN datasets d ON d.id_dataset = p.id_dataset
        WHERE d.nome_dataset = 'USMLE'
        ORDER BY p.id_pergunta
    """)
    return cur.fetchall()


def load_mcq_options() -> dict[str, dict]:
    with open("respostas_atividade_mcq.json", encoding="utf-8") as f:
        raw = json.load(f)
    options_map = {}
    for item in raw:
        q = item.get("question", "").strip() if isinstance(item.get("question"), str) else ""
        opts = item.get("options", {})
        if isinstance(opts, str):
            opts = {}
        if q:
            options_map[q] = opts
    return options_map


def build_question_id_map(cur) -> dict[str, int]:
    cur.execute("SELECT id_pergunta, enunciado FROM perguntas")
    return {row["enunciado"]: row["id_pergunta"] for row in cur.fetchall()}


def call_model(model_id: str, prompt: str) -> str:
    completion = client.responses.create(
        model=model_id,
        input=prompt,
    )
    return completion.output_text.strip()


def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        ensure_rag_columns(cur)
        conn.commit()

        mapping = get_model_mapping()
        if not mapping:
            print("[ERRO] Nenhum modelo mapeado. Abortando.")
            sys.exit(1)

        models = get_candidate_models(cur)
        questions_open = get_open_questions(cur)
        mcq_options = load_mcq_options()

        print(f"Modelos candidatos: {len(models)}")
        print(f"Perguntas abertas (K-QA): {len(questions_open)}")

        for modelo in models:
            full_name = modelo["full_name"]
            model_id = mapping.get(full_name)
            if not model_id:
                print(f"  Pulando {full_name}: sem mapeamento")
                continue

            print(f"\n=== Modelo: {full_name} -> {model_id} ===")

            for questions, dataset_type in [(questions_open, "open")]:
                for q in questions:
                    cur.execute(
                        """
                        SELECT id_resposta FROM respostas_atividade_1
                        WHERE id_pergunta = %s AND id_modelo = %s AND rag = TRUE
                        """,
                        (q["id_pergunta"], modelo["id_modelo"]),
                    )
                    if cur.fetchone():
                        continue

                    if dataset_type == "open":
                        ctx, _ = retrieve_context(q["enunciado"])
                        prompt = PROMPT_ABERTA_TEMPLATE.format(contexto=ctx, question=q["enunciado"])
                    else:
                        opts = mcq_options.get(q["enunciado"], {})
                        opts_str = "\n".join(f"{k}: {v}" for k, v in sorted(opts.items())) if opts else ""
                        ctx, _ = retrieve_context_for_mcq(q["enunciado"], opts) if opts else retrieve_context(q["enunciado"])
                        prompt = PROMPT_MCQ_TEMPLATE.format(
                            contexto=ctx, question=q["enunciado"], options=opts_str,
                        )

                    try:
                        print(f"  [{full_name}] Pergunta {q['id_pergunta']}...", end=" ", flush=True)
                        resposta = call_model(model_id, prompt)
                        print("OK")
                    except Exception as e:
                        print(f"[ERRO] {e}")
                        conn.rollback()
                        continue

                    try:
                        cur.execute(
                            """INSERT INTO respostas_atividade_1
                               (id_pergunta, id_modelo, texto_resposta, rag, contexto_rag)
                               VALUES (%s, %s, %s, TRUE, %s)""",
                            (q["id_pergunta"], modelo["id_modelo"], resposta, ctx),
                        )
                        conn.commit()
                    except Exception as e:
                        print(f"  [ERRO DB] {e}")
                        conn.rollback()

                    time.sleep(0.5)

        print("\n=== Geração RAG concluída ===")

    except Exception as e:
        conn.rollback()
        print(f"[ERRO FATAL] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    print("=== Geração de Respostas com RAG ===")
    main()