import os
import psycopg2
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LLM Judge Dashboard", layout="wide")

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "llm_judge"),
    user=os.getenv("DB_USER", "llm_user"),
    password=os.getenv("DB_PASSWORD", "llm_pass"),
)

REFERENCE_JUDGE = "gpt 5.4-mini"


@st.cache_data(ttl=60)
def load_media_por_modelo():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
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
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_media_consenso():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
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
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_humano_vs_juiz():
    conn = psycopg2.connect(**DB)
    try:
        df = pd.read_sql("""
            SELECT
                ah.id_resposta,
                ah.nota                                       AS nota_humana,
                m_juiz.nome_modelo || ' ' || m_juiz.versao    AS juiz,
                aj.nota                                       AS nota_juiz
            FROM avaliacoes_humanas ah
            JOIN avaliacoes_juiz aj     ON aj.id_resposta    = ah.id_resposta
            JOIN modelos        m_juiz  ON m_juiz.id_modelo  = aj.id_modelo_juiz
        """, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_avaliacoes_kqa():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
        SELECT
            a.id_resposta,
            m_cand.nome_modelo || ' ' || m_cand.versao AS candidato,
            m_juiz.nome_modelo || ' ' || m_juiz.versao AS juiz,
            a.nota
        FROM avaliacoes_juiz a
        JOIN respostas_atividade_1 r ON r.id_resposta  = a.id_resposta
        JOIN perguntas             p ON p.id_pergunta  = r.id_pergunta
        JOIN datasets              d ON d.id_dataset   = p.id_dataset
        JOIN modelos         m_cand ON m_cand.id_modelo = r.id_modelo
        JOIN modelos         m_juiz ON m_juiz.id_modelo = a.id_modelo_juiz
        WHERE d.nome_dataset = 'Itaymanes K-QA'
    """, conn)
    conn.close()
    return df


# Layout 

st.title("LLM Judge — Dashboard de Avaliação")

try:
    df_media = load_media_por_modelo()
    df_consenso = load_media_consenso()
    df_kqa = load_avaliacoes_kqa()
    df_humano = load_humano_vs_juiz()
except Exception as e:
    st.error(f"Erro ao conectar ao banco: {e}")
    st.stop()

if df_media.empty:
    st.warning("Nenhuma avaliação encontrada. Execute o judge.py primeiro.")
    st.stop()

datasets = df_consenso["nome_dataset"].unique().tolist()
dataset_sel = st.sidebar.selectbox("Dataset", datasets)

df_media_f = df_media[df_media["nome_dataset"] == dataset_sel]
df_consenso_f = df_consenso[df_consenso["nome_dataset"] == dataset_sel]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ranking", "Por Judge", "Distribuição", "Reference Judge", "Humano vs LLM"])

# Ranking (consenso) 
with tab1:
    st.subheader("Ranking dos modelos — Média de consenso dos judges")
    fig = px.bar(
        df_consenso_f.sort_values("media_consenso"),
        x="media_consenso",
        y="candidato",
        orientation="h",
        text="media_consenso",
        color="media_consenso",
        color_continuous_scale="Blues",
        range_x=[1, 5],
        labels={"media_consenso": "Média (1–5)", "candidato": "Modelo"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=max(400, len(df_consenso_f) * 40))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_consenso_f.sort_values("media_consenso", ascending=False), use_container_width=True)

# Heatmap (candidato × judge) 
with tab2:
    st.subheader("Heatmap — Média por modelo candidato × judge")
    pivot = df_media_f.pivot_table(index="candidato", columns="juiz", values="media_nota")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        zmin=1, zmax=5,
        text=pivot.values.round(2),
        texttemplate="%{text}",
        hovertemplate="Candidato: %{y}<br>Judge: %{x}<br>Média: %{z}<extra></extra>",
    ))
    fig.update_layout(height=max(400, len(pivot) * 40), xaxis_title="Judge", yaxis_title="Modelo candidato")
    st.plotly_chart(fig, use_container_width=True)

# Distribuição (box plot) 
with tab3:
    st.subheader("Distribuição de scores por modelo candidato")
    df_box = df_kqa if dataset_sel == "Itaymanes K-QA" else pd.DataFrame()
    if df_box.empty:
        st.info("Distribuição detalhada disponível apenas para K-QA.")
    else:
        fig = px.box(
            df_box,
            x="candidato",
            y="nota",
            color="juiz",
            points="all",
            range_y=[0.5, 5.5],
            labels={"nota": "Score (1–5)", "candidato": "Modelo", "juiz": "Judge"},
        )
        fig.update_layout(height=500, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

# Reference Judge 
with tab4:
    st.subheader(f"Reference Judge — concordância com `{REFERENCE_JUDGE}`")

    if df_kqa.empty:
        st.info("Dados K-QA não disponíveis.")
    else:
        pivot_rj = df_kqa.pivot_table(index="id_resposta", columns="juiz", values="nota")

        if REFERENCE_JUDGE not in pivot_rj.columns:
            st.warning(f"Judge de referência '{REFERENCE_JUDGE}' não encontrado.")
        else:
            outros = [c for c in pivot_rj.columns if c != REFERENCE_JUDGE]
            rows = []
            for juiz in outros:
                subset = pivot_rj[[REFERENCE_JUDGE, juiz]].dropna()
                if len(subset) < 3:
                    continue
                rho, pval = spearmanr(subset[REFERENCE_JUDGE], subset[juiz])
                rows.append({"Judge": juiz, "ρ (Spearman)": round(rho, 4), "p-value": round(pval, 4), "n": len(subset)})

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            cols = st.columns(len(outros))
            for col, juiz in zip(cols, outros):
                subset = pivot_rj[[REFERENCE_JUDGE, juiz]].dropna()
                if len(subset) < 3:
                    continue
                fig = px.scatter(
                    subset,
                    x=REFERENCE_JUDGE,
                    y=juiz,
                    trendline="ols",
                    range_x=[0.5, 5.5],
                    range_y=[0.5, 5.5],
                    labels={REFERENCE_JUDGE: f"Ref ({REFERENCE_JUDGE})", juiz: juiz},
                    title=f"{REFERENCE_JUDGE} vs {juiz}",
                )
                fig.update_layout(height=350)
                col.plotly_chart(fig, use_container_width=True)

# Humano vs LLM Judges
with tab5:
    st.subheader("Gabarito Humano vs LLM Judges — Correlação de Spearman")

    if df_humano.empty:
        st.info("Gabarito humano não disponível. Execute load_gabarito.py primeiro.")
    else:
        juizes = sorted(df_humano["juiz"].unique())
        rows = []
        for juiz in juizes:
            subset = df_humano[df_humano["juiz"] == juiz][["nota_humana", "nota_juiz"]].dropna()
            if len(subset) < 3:
                continue
            rho, pval = spearmanr(subset["nota_humana"], subset["nota_juiz"])
            rows.append({"Judge": juiz, "ρ (Spearman)": round(rho, 4), "p-value": round(pval, 4), "n": len(subset)})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        cols = st.columns(max(len(juizes), 1))
        for col, juiz in zip(cols, juizes):
            subset = df_humano[df_humano["juiz"] == juiz][["nota_humana", "nota_juiz"]].dropna()
            if len(subset) < 3:
                continue
            fig = px.scatter(
                subset,
                x="nota_humana",
                y="nota_juiz",
                trendline="ols",
                range_x=[0.5, 5.5],
                range_y=[0.5, 5.5],
                labels={"nota_humana": "Nota Humana", "nota_juiz": "Nota LLM Judge"},
                title=juiz,
            )
            fig.update_layout(height=350)
            col.plotly_chart(fig, use_container_width=True)
