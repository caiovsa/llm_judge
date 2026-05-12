import json
import os
import re
import sys
import psycopg2


DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "llm_judge"),
    user=os.getenv("DB_USER", "llm_user"),
    password=os.getenv("DB_PASSWORD", "llm_pass"),
)

GABARITO_FILE = "gabarito_humano_abertas.json"


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


def parse_gabarito(json_path: str) -> list[tuple[str, str, int]]:
    """Returns list of (question_text, model_name_raw, score)."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for item in raw:
        q_text = item.get("question")
        if not q_text:
            continue
        for k, v in item.items():
            if k == "question" or "Quest" in k:
                continue
            records.append((q_text, k, v))
    return records


def main():
    records = parse_gabarito(GABARITO_FILE)
    print(f"Gabarito: {len(records)} anotações para {len(set(r[0] for r in records))} questões")

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS avaliacoes_humanas (
                id_avaliacao_humana SERIAL PRIMARY KEY,
                id_resposta INT NOT NULL REFERENCES respostas_atividade_1(id_resposta) ON DELETE CASCADE,
                nota SMALLINT NOT NULL CHECK (nota BETWEEN 1 AND 5),
                UNIQUE (id_resposta)
            )
        """)
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM avaliacoes_humanas")
        if cur.fetchone()[0] > 0:
            print("Gabarito já carregado. Nada a fazer.")
            return

        inserted = 0
        skipped = 0
        for q_text, model_raw, score in records:
            nome, versao = parse_model(model_raw)

            cur.execute(
                """
                SELECT r.id_resposta
                FROM respostas_atividade_1 r
                JOIN perguntas p ON p.id_pergunta = r.id_pergunta
                JOIN modelos   m ON m.id_modelo   = r.id_modelo
                JOIN datasets  d ON d.id_dataset  = p.id_dataset
                WHERE d.nome_dataset = 'Itaymanes K-QA'
                  AND p.enunciado    = %s
                  AND m.nome_modelo  = %s
                  AND m.versao       = %s
                """,
                (q_text, nome, versao),
            )
            row = cur.fetchone()
            if not row:
                print(f"  [SKIP] {model_raw} ({nome} {versao}) — Q não encontrada: {q_text[:60]}")
                skipped += 1
                continue

            cur.execute(
                "INSERT INTO avaliacoes_humanas (id_resposta, nota) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (row[0], score),
            )
            inserted += 1

        conn.commit()
        print(f"Gabarito carregado: {inserted} inseridas, {skipped} ignoradas.")

    except Exception as e:
        conn.rollback()
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
