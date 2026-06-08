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

QUERY_COMPARATIVO = """
SELECT
    p.id_pergunta,
    d.nome_dataset,
    m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
    m_juiz.nome_modelo || ' ' || m_juiz.versao  AS juiz,
    r_sem.texto_resposta  AS resposta_sem_rag,
    r_com.texto_resposta  AS resposta_com_rag,
    a_sem.nota            AS nota_sem_rag,
    a_com.nota            AS nota_com_rag,
    a_com.justificativa   AS justificativa_rag
FROM avaliacoes_juiz a_sem
JOIN avaliacoes_juiz a_com
    ON a_com.id_resposta    != a_sem.id_resposta
   AND a_com.id_modelo_juiz  = a_sem.id_modelo_juiz
   AND a_com.rag             = TRUE
JOIN respostas_atividade_1 r_sem ON r_sem.id_resposta  = a_sem.id_resposta  AND r_sem.rag = FALSE
JOIN respostas_atividade_1 r_com ON r_com.id_pergunta   = r_sem.id_pergunta
   AND r_com.id_modelo    = r_sem.id_modelo
   AND r_com.rag          = TRUE
JOIN perguntas p ON p.id_pergunta = r_sem.id_pergunta
JOIN datasets d ON d.id_dataset = p.id_dataset
JOIN modelos m_cand ON m_cand.id_modelo = r_sem.id_modelo
JOIN modelos m_juiz ON m_juiz.id_modelo = a_sem.id_modelo_juiz
WHERE a_sem.rag = FALSE
ORDER BY d.nome_dataset, candidato, juiz
"""

QUERY_MEDIA_RAG = """
SELECT
    d.nome_dataset,
    m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
    m_juiz.nome_modelo || ' ' || m_juiz.versao AS juiz,
    ROUND(AVG(a.nota)::numeric, 2)             AS media_nota,
    COUNT(*)                                   AS total
FROM avaliacoes_juiz a
JOIN respostas_atividade_1 r ON r.id_resposta  = a.id_resposta
JOIN perguntas             p ON p.id_pergunta  = r.id_pergunta
JOIN datasets              d ON d.id_dataset   = p.id_dataset
JOIN modelos         m_cand ON m_cand.id_modelo = r.id_modelo
JOIN modelos         m_juiz ON m_juiz.id_modelo = a.id_modelo_juiz
WHERE a.rag = TRUE
GROUP BY d.nome_dataset, candidato, juiz
ORDER BY d.nome_dataset, media_nota DESC
"""

QUERY_MEDIA_SEM_RAG = """
SELECT
    d.nome_dataset,
    m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
    ROUND(AVG(a.nota)::numeric, 2)             AS media_sem_rag,
    COUNT(*)                                   AS total
FROM avaliacoes_juiz a
JOIN respostas_atividade_1 r ON r.id_resposta  = a.id_resposta
JOIN perguntas             p ON p.id_pergunta  = r.id_pergunta
JOIN datasets              d ON d.id_dataset   = p.id_dataset
JOIN modelos         m_cand ON m_cand.id_modelo = r.id_modelo
WHERE a.rag = FALSE
GROUP BY d.nome_dataset, candidato
"""

QUERY_CONSENSO_RAG = """
SELECT
    d.nome_dataset,
    m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
    ROUND(AVG(a.nota)::numeric, 2)             AS media_consenso,
    COUNT(DISTINCT a.id_modelo_juiz)           AS num_judges,
    COUNT(*)                                   AS total_avaliacoes
FROM avaliacoes_juiz a
JOIN respostas_atividade_1 r ON r.id_resposta  = a.id_resposta
JOIN perguntas             p ON p.id_pergunta  = r.id_pergunta
JOIN datasets              d ON d.id_dataset   = p.id_dataset
JOIN modelos         m_cand ON m_cand.id_modelo = r.id_modelo
WHERE a.rag = TRUE
GROUP BY d.nome_dataset, candidato
ORDER BY d.nome_dataset, media_consenso DESC
"""

QUERY_HUMANO_VS_RAG = """
SELECT
    ah.id_resposta,
    ah.nota                                       AS nota_humana,
    r_sem.id_modelo                               AS id_modelo,
    m_juiz.nome_modelo || ' ' || m_juiz.versao    AS juiz,
    aj.nota                                       AS nota_juiz_rag
FROM avaliacoes_humanas ah
JOIN respostas_atividade_1 r_sem ON r_sem.id_resposta = ah.id_resposta AND r_sem.rag = FALSE
JOIN respostas_atividade_1 r_rag ON r_rag.id_pergunta = r_sem.id_pergunta
   AND r_rag.id_modelo  = r_sem.id_modelo
   AND r_rag.rag        = TRUE
JOIN avaliacoes_juiz aj ON aj.id_resposta = r_rag.id_resposta AND aj.rag = TRUE
JOIN modelos m_juiz ON m_juiz.id_modelo = aj.id_modelo_juiz
"""


def main():
    conn = psycopg2.connect(**DB)

    print("=== ANÁLISE COMPARATIVA: RAG vs SEM RAG ===\n")

    df = pd.read_sql(QUERY_COMPARATIVO, conn)
    if df.empty:
        print("Nenhum dado comparativo encontrado. Execute generate_rag.py e judge_rag.py primeiro.")
        conn.close()
        return

    print("--- Média por modelo: SEM RAG vs COM RAG (consenso entre judges) ---\n")
    df_media_sem = pd.read_sql(QUERY_MEDIA_SEM_RAG, conn)
    df_media_rag = pd.read_sql(QUERY_CONSENSO_RAG, conn)

    merged = pd.merge(
        df_media_sem[["nome_dataset", "candidato", "media_sem_rag"]],
        df_media_rag[["nome_dataset", "candidato", "media_consenso"]],
        on=["nome_dataset", "candidato"],
        how="outer",
        suffixes=("_sem", "_rag"),
    )
    merged["delta"] = merged["media_consenso"] - merged["media_sem_rag"]
    merged = merged.sort_values("nome_dataset")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(merged.to_string(index=False))

    print("\n\n--- Correlação de Spearman: nota SEM RAG vs nota COM RAG ---\n")
    for dataset in df["nome_dataset"].unique():
        sub = df[df["nome_dataset"] == dataset]
        for candidato in sorted(sub["candidato"].unique()):
            cdf = sub[sub["candidato"] == candidato][["nota_sem_rag", "nota_com_rag"]].dropna()
            if len(cdf) < 3:
                print(f"  {dataset:20s} | {candidato:30s} | dados insuficientes")
                continue
            rho, pval = spearmanr(cdf["nota_sem_rag"], cdf["nota_com_rag"])
            print(f"  {dataset:20s} | {candidato:30s} | ρ = {rho:.4f}  (p = {pval:.4f},  n = {len(cdf)})")

    print("\n\n--- Impacto do RAG: quantas respostas melhoraram / pioraram ---\n")
    df["delta_nota"] = df["nota_com_rag"] - df["nota_sem_rag"]
    for dataset in df["nome_dataset"].unique():
        sub = df[df["nome_dataset"] == dataset]
        melhorou = (sub["delta_nota"] > 0).sum()
        piorou = (sub["delta_nota"] < 0).sum()
        manteve = (sub["delta_nota"] == 0).sum()
        media_delta = sub["delta_nota"].mean()
        print(f"  {dataset:20s} | Melhorou: {melhorou:>4} | Piorou: {piorou:>4} | Manteve: {manteve:>4} | Δ médio: {media_delta:+.3f}")

    print("\n\n--- Gabarito Humano vs LLM Judge (RAG) ---\n")
    try:
        df_humano = pd.read_sql(QUERY_HUMANO_VS_RAG, conn)
        if not df_humano.empty:
            for juiz in sorted(df_humano["juiz"].unique()):
                sub = df_humano[df_humano["juiz"] == juiz][["nota_humana", "nota_juiz_rag"]].dropna()
                if len(sub) < 3:
                    print(f"  vs {juiz}: dados insuficientes")
                    continue
                rho, pval = spearmanr(sub["nota_humana"], sub["nota_juiz_rag"])
                print(f"  vs {juiz:42s} →  ρ = {rho:.4f}  (p = {pval:.4f},  n = {len(sub)})")
        else:
            print("  Sem dados de gabarito humano para RAG.")
    except Exception as e:
        print(f"  Gabarito humano não disponível: {e}")

    conn.close()


if __name__ == "__main__":
    main()