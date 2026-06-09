import json
import os
import re
import sys

import psycopg2
import psycopg2.extras

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "llm_judge"),
    user=os.getenv("DB_USER", "llm_user"),
    password=os.getenv("DB_PASSWORD", "llm_pass"),
)

INPUT_FILE = "respostas_rag_simple.json"


def parse_model(name: str) -> tuple[str, str]:
    m = re.match(r'^([A-Za-z][A-Za-z]*)[ ](.+)$', name)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^([A-Za-z][A-Za-z]*)-(.+)$', name)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^([A-Za-z]+)[_](.+)$', name)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r'^([A-Za-z]+)(\d.*)$', name)
    if m:
        return m.group(1), m.group(2)
    return name, "unknown"


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


def build_question_map(cur) -> dict[str, int]:
    cur.execute("SELECT id_pergunta, enunciado FROM perguntas")
    return {row[1]: row[0] for row in cur.fetchall()}


def build_model_map(cur) -> dict[tuple[str, str], int]:
    cur.execute("SELECT id_modelo, nome_modelo, versao FROM modelos WHERE tipo = 'candidato'")
    return {(row[1], row[2]): row[0] for row in cur.fetchall()}


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERRO] Arquivo '{INPUT_FILE}' não encontrado. Execute generate_rag_simple.py primeiro.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Lendo {len(data)} respostas de {INPUT_FILE}")

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    try:
        ensure_rag_columns(cur)

        question_map = build_question_map(cur)
        model_map = build_model_map(cur)

        inserted = 0
        skipped = 0
        errors = 0

        for item in data:
            question = item.get("question", "").strip()
            model_name = item.get("model_name", "").strip()
            answer = item.get("model_answer", "").strip()

            if not question or not model_name:
                continue

            id_pergunta = question_map.get(question)
            if id_pergunta is None:
                print(f"  [ERRO] Pergunta não encontrada: {question[:60]}...")
                errors += 1
                continue

            nome, versao = parse_model(model_name)
            id_modelo = model_map.get((nome, versao))
            if id_modelo is None:
                print(f"  [ERRO] Modelo não encontrado: {model_name} -> ({nome}, {versao})")
                errors += 1
                continue

            cur.execute(
                """
                SELECT id_resposta FROM respostas_atividade_1
                WHERE id_pergunta = %s AND id_modelo = %s AND rag = TRUE
                """,
                (id_pergunta, id_modelo),
            )
            if cur.fetchone():
                skipped += 1
                continue

            cur.execute(
                """INSERT INTO respostas_atividade_1
                   (id_pergunta, id_modelo, texto_resposta, rag)
                   VALUES (%s, %s, %s, TRUE)""",
                (id_pergunta, id_modelo, answer),
            )
            inserted += 1

        conn.commit()
        print(f"\nConcluído: {inserted} inseridas, {skipped} já existentes, {errors} erros")

    except Exception as e:
        conn.rollback()
        print(f"[ERRO FATAL] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()