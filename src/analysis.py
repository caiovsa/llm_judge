import os
import psycopg2
import pandas as pd
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "llm_judge"),
    user=os.getenv("DB_USER", "llm_user"),
    password=os.getenv("DB_PASSWORD", "llm_pass"),
)

QUERY_MEDIA_POR_MODELO = """
SELECT
    d.nome_dataset,
    m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
    m_juiz.nome_modelo || ' ' || m_juiz.versao  AS juiz,
    ROUND(AVG(a.nota)::numeric, 2)              AS media_nota,
    COUNT(*)                                    AS total_avaliado
FROM avaliacoes_juiz a
JOIN respostas_atividade_1 r ON r.id_resposta   = a.id_resposta
JOIN perguntas             p ON p.id_pergunta   = r.id_pergunta
JOIN datasets              d ON d.id_dataset    = p.id_dataset
JOIN modelos         m_cand ON m_cand.id_modelo = r.id_modelo
JOIN modelos         m_juiz ON m_juiz.id_modelo = a.id_modelo_juiz
GROUP BY d.nome_dataset, candidato, juiz
ORDER BY d.nome_dataset, media_nota DESC
"""

QUERY_MEDIA_CONSENSO = """
SELECT
    d.nome_dataset,
    m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
    ROUND(AVG(a.nota)::numeric, 2)             AS media_consenso,
    COUNT(DISTINCT a.id_modelo_juiz)           AS num_judges,
    COUNT(*)                                   AS total_avaliacoes
FROM avaliacoes_juiz a
JOIN respostas_atividade_1 r ON r.id_resposta   = a.id_resposta
JOIN perguntas             p ON p.id_pergunta   = r.id_pergunta
JOIN datasets              d ON d.id_dataset    = p.id_dataset
JOIN modelos         m_cand ON m_cand.id_modelo = r.id_modelo
GROUP BY d.nome_dataset, candidato
ORDER BY d.nome_dataset, media_consenso DESC
"""

QUERY_INTER_JUDGE_KQA = """
SELECT
    a.id_resposta,
    m_juiz.nome_modelo || ' ' || m_juiz.versao AS juiz,
    a.nota
FROM avaliacoes_juiz a
JOIN respostas_atividade_1 r ON r.id_resposta  = a.id_resposta
JOIN perguntas             p ON p.id_pergunta  = r.id_pergunta
JOIN datasets              d ON d.id_dataset   = p.id_dataset
JOIN modelos         m_juiz ON m_juiz.id_modelo = a.id_modelo_juiz
WHERE d.nome_dataset = 'Itaymanes K-QA'
"""

REFERENCE_JUDGE = "gpt 5.4-mini"


def analisar_reference_judge(df_inter: pd.DataFrame):
    pivot = df_inter.pivot_table(index="id_resposta", columns="juiz", values="nota")

    if REFERENCE_JUDGE not in pivot.columns:
        print(f"  Judge de referência '{REFERENCE_JUDGE}' não encontrado nas avaliações.")
        return

    print(f"  Judge de referência: {REFERENCE_JUDGE}\n")
    for juiz in pivot.columns:
        if juiz == REFERENCE_JUDGE:
            continue
        subset = pivot[[REFERENCE_JUDGE, juiz]].dropna()
        if len(subset) < 3:
            print(f"  vs {juiz}: dados insuficientes")
            continue
        rho, pval = spearmanr(subset[REFERENCE_JUDGE], subset[juiz])
        print(f"  vs {juiz:42s} →  ρ = {rho:.4f}  (p = {pval:.4f},  n = {len(subset)})")


def main():
    conn = psycopg2.connect(**DB)

    df_media = pd.read_sql(QUERY_MEDIA_POR_MODELO, conn)
    if df_media.empty:
        print("Nenhuma avaliação encontrada. Execute o judge.py primeiro.")
        conn.close()
        return

    print("=== Média por modelo candidato (por judge) ===\n")
    print(df_media.to_string(index=False))

    print("\n\n=== Média de consenso dos judges por modelo candidato ===\n")
    df_consenso = pd.read_sql(QUERY_MEDIA_CONSENSO, conn)
    print(df_consenso.to_string(index=False))

    print("\n\n=== K-QA: Reference Judge (concordância entre judges) ===\n")
    df_inter = pd.read_sql(QUERY_INTER_JUDGE_KQA, conn)
    if df_inter.empty:
        print("  Sem avaliações K-QA para análise de concordância.")
    else:
        analisar_reference_judge(df_inter)

    conn.close()


if __name__ == "__main__":
    main()
