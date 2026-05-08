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

DATASET_NAME = "Itaymanes K-QA"
DATASET_DESC = "Dataset de perguntas e respostas médicas com 201 questões"

# Maps full model name to (api_endpoint)
API_ENDPOINTS = {
    "Claude":    "https://api.anthropic.com/v1",
    "GPT":       "https://api.openai.com/v1",
    "gpt":       "https://api.openai.com/v1",
    "Gemini":    "https://generativelanguage.googleapis.com/v1",
    "Gemma":     "https://generativelanguage.googleapis.com/v1",
    "Llama":     "https://api.meta.com/v1",
    "llama":     "https://api.meta.com/v1",
    "DeepSeek":  "https://api.deepseek.com/v1",
    "deepseek":  "https://api.deepseek.com/v1",
    "Kimi":      "https://api.moonshot.cn/v1",
    "Nova":      "https://api.nova.ai/v1",
    "phi":       "https://api.azure.com/openai/v1",
    "qwen":      "https://dashscope.aliyuncs.com/api/v1",
    "deep":      "https://api.deepseek.com/v1",
}


def parse_model(name: str) -> tuple[str, str]:
    """Split a raw model name into (nome_modelo, versao)."""
    # "Claude 4.6 Sonnet", "Gemini 3.0"
    m = re.match(r'^([A-Za-z][A-Za-z]*)[ ](.+)$', name)
    if m:
        return m.group(1), m.group(2)
    # "GPT-5.4 Thinking", "Gemma-3-4B", "Llama-3-8B", "DeepSeek-V3-0324", "Nova-Lite", "Kimi-K2.5", "gpt-5-mini"
    m = re.match(r'^([A-Za-z][A-Za-z]*)-(.+)$', name)
    if m:
        return m.group(1), m.group(2)
    # "deepseek_r1_1dot5b", "phi4_mini"
    m = re.match(r'^([A-Za-z]+)[_](.+)$', name)
    if m:
        return m.group(1), m.group(2)
    # "llama3", "phi4"
    m = re.match(r'^([A-Za-z]+)(\d.*)$', name)
    if m:
        return m.group(1), m.group(2)
    return name, "unknown"


def get_endpoint(nome: str) -> str:
    return API_ENDPOINTS.get(nome, f"https://api.{nome.lower()}.com/v1")


def main():
    with open("respostas_atividade_1.json", encoding="utf-8") as f:
        raw = json.load(f)

    # Strip trailing spaces from keys
    data = [{k.strip(): v.strip() if isinstance(v, str) else v for k, v in item.items()} for item in raw]

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    try:
        # 1. Dataset
        cur.execute(
            "INSERT INTO datasets (nome_dataset, descricao) VALUES (%s, %s) "
            "ON CONFLICT (nome_dataset) DO UPDATE SET descricao = EXCLUDED.descricao "
            "RETURNING id_dataset",
            (DATASET_NAME, DATASET_DESC),
        )
        id_dataset = cur.fetchone()[0]
        print(f"Dataset '{DATASET_NAME}' -> id {id_dataset}")

        # 2. Modelos candidatos
        model_id: dict[str, int] = {}
        for full_name in sorted({item["model_name"] for item in data}):
            nome, versao = parse_model(full_name)
            endpoint = get_endpoint(nome)
            cur.execute(
                "INSERT INTO modelos (nome_modelo, versao, tipo, api_endpoint) "
                "VALUES (%s, %s, 'candidato', %s) "
                "ON CONFLICT (nome_modelo, versao) DO NOTHING "
                "RETURNING id_modelo",
                (nome, versao, endpoint),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    "SELECT id_modelo FROM modelos WHERE nome_modelo = %s AND versao = %s",
                    (nome, versao),
                )
                row = cur.fetchone()
            model_id[full_name] = row[0]
            print(f"  Modelo '{full_name}' -> ({nome}, {versao}) id {row[0]}")

        # 3. Perguntas (deduplicadas por enunciado)
        question_id: dict[str, int] = {}
        for item in data:
            q = item["question"]
            if q in question_id:
                continue
            cur.execute(
                "INSERT INTO perguntas (id_dataset, enunciado, resposta_ouro) "
                "VALUES (%s, %s, %s) RETURNING id_pergunta",
                (id_dataset, q, item["golden_answer"]),
            )
            question_id[q] = cur.fetchone()[0]
        print(f"{len(question_id)} perguntas inseridas")

        # 4. Respostas
        for item in data:
            cur.execute(
                "INSERT INTO respostas_atividade_1 (id_pergunta, id_modelo, texto_resposta) "
                "VALUES (%s, %s, %s)",
                (question_id[item["question"]], model_id[item["model_name"]], item["model_answer"]),
            )
        print(f"{len(data)} respostas inseridas")

        conn.commit()
        print("Carga concluída com sucesso.")

    except Exception as e:
        conn.rollback()
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
