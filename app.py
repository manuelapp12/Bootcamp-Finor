#!/usr/bin/env python3
"""
FINOR | Index Tracking IBOV — Dashboard de Storytelling
Bootcamp de Introdução a Data Science · 2025

Execução:
    cd /caminho/para/Bootcamp-Finor
    streamlit run app.py
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cvxpy as cp

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FINOR | Index Tracking IBOV",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CORES
# ══════════════════════════════════════════════════════════════════════════════
VERDE  = "#00a859"
AZUL   = "#0057b7"
OURO   = "#FFD700"
BRANCO = "#e6edf3"
CINZA  = "#8b949e"
BG     = "#0d1117"
BG2    = "#161b22"
BORDA  = "#30363d"

# Versões RGBA para uso em propriedades Plotly (não aceita hex 8 dígitos)
VERDE_A10 = "rgba(0, 168, 89, 0.10)"   # VERDE com 10% de opacidade
VERDE_A07 = "rgba(0, 168, 89, 0.07)"   # VERDE com 7% de opacidade

PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(family="Inter, system-ui, sans-serif", color=BRANCO, size=12),
    xaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDA),
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; }}
section[data-testid="stSidebar"] {{
    background: {BG2};
    border-right: 1px solid {BORDA};
}}
.card {{
    background: {BG2};
    border: 1px solid {BORDA};
    border-radius: 10px;
    padding: 20px;
    margin: 8px 0;
}}
.card-green {{
    background: {BG2};
    border: 2px solid {VERDE};
    border-radius: 10px;
    padding: 20px;
    margin: 8px 0;
}}
.chapter-num {{
    font-size: 5.5rem;
    font-weight: 900;
    color: {VERDE}12;
    line-height: 1;
    font-family: 'Courier New', monospace;
    margin-bottom: -16px;
}}
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}}
.b-green {{ background: {VERDE}18; color: {VERDE}; border: 1px solid {VERDE}50; }}
.b-gold  {{ background: {OURO}18;  color: {OURO};  border: 1px solid {OURO}50; }}
.b-blue  {{ background: {AZUL}40;  color: #58a6ff; border: 1px solid #58a6ff50; }}
.kpi {{ text-align: center; padding: 14px 8px; }}
.kpi-val {{ font-size: 1.9rem; font-weight: 700; color: {VERDE}; }}
.kpi-lbl {{ font-size: 0.75rem; color: {CINZA}; margin-top: 3px; }}
hr {{ border-color: {BORDA}; margin: 24px 0; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DADOS (cached)
# ══════════════════════════════════════════════════════════════════════════════
BASE = "data/processed"

@st.cache_data
def load_main():
    ret  = pd.read_csv(f"{BASE}/ibov_acoes_selecionadas.csv", index_col=0, parse_dates=True)
    idx  = pd.read_csv(f"{BASE}/ibov_indice_retornos.csv",    index_col=0, parse_dates=True)
    dados = ret.join(idx["Variação_Diária_%"], how="inner").dropna()
    dados.rename(columns={"Variação_Diária_%": "IBOV"}, inplace=True)
    dados["IBOV"] /= 100.0
    return dados, idx

@st.cache_data
def load_results():
    pesos_qp = pd.read_csv(f"{BASE}/index_tracking_pesos_qp.csv",        index_col=0)
    pesos_en = pd.read_csv(f"{BASE}/index_tracking_pesos_elasticnet.csv", index_col=0)
    backtest = pd.read_csv(f"{BASE}/index_tracking_backtest.csv")
    card     = pd.read_csv(f"{BASE}/tradeoff_cardinalidade_expandido.csv")
    ml_res   = pd.read_csv(f"{BASE}/ml_rebalanceamento_resumo.csv")
    ml_imp   = pd.read_csv(f"{BASE}/ml_feature_importances.csv")
    with open(f"{BASE}/index_tracking_resumo.json") as f:
        resumo = json.load(f)
    return pesos_qp, pesos_en, backtest, card, ml_res, ml_imp, resumo

@st.cache_data
def calcular_pesos_k(K: int) -> pd.Series:
    """Otimiza carteira QP para K ações. Resultado cacheado por K."""
    dados, _ = load_main()
    treino = dados.loc[:"2024-01-31"]
    acoes  = [c for c in dados.columns if c != "IBOV"]
    X_tr   = treino[acoes].values
    y_tr   = treino["IBOV"].values
    # Ranking via universo completo
    n      = len(acoes)
    w_all  = cp.Variable(n)
    cp.Problem(
        cp.Minimize(cp.sum_squares(X_tr @ w_all - y_tr)),
        [cp.sum(w_all) == 1, w_all >= 0, w_all <= 0.15]
    ).solve(solver=cp.OSQP)
    ranking = pd.Series(np.maximum(w_all.value, 0), index=acoes).sort_values(ascending=False)
    top    = ranking.head(K).index.tolist()
    cap_k  = float(np.clip(1.5 / K, 0.15, 0.40))
    X_k    = treino[top].values
    w_k    = cp.Variable(K)
    cp.Problem(
        cp.Minimize(cp.sum_squares(X_k @ w_k - y_tr)),
        [cp.sum(w_k) == 1, w_k >= 0, w_k <= cap_k]
    ).solve(solver=cp.OSQP)
    pesos = pd.Series(np.maximum(w_k.value, 0), index=top)
    return (pesos / pesos.sum()).sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
JANELAS = [
    ("2024-02-01","2024-04-30"), ("2024-05-01","2024-07-31"),
    ("2024-08-01","2024-10-31"), ("2024-11-01","2025-01-31"),
    ("2025-02-01","2025-04-30"),
]
SECOES = [
    "🏠  Capa",
    "1 —  O Problema",
    "2 —  Os Dados",
    "3 —  O Modelo (QP)",
    "4 —  Backtest",
    "5 —  Trade-off de Cardinalidade",
    "6 —  ML de Rebalanceamento",
    "7 —  Recomendação Final",
]

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:14px 0 10px 0;'>
        <div style='font-size:2.2rem'>📈</div>
        <div style='font-weight:700; font-size:1.05rem; color:{BRANCO}'>FINOR</div>
        <div style='font-size:0.72rem; color:{CINZA}'>Index Tracking · Bootcamp 2025</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    secao = st.radio("", SECOES, label_visibility="hidden")
    idx_s = SECOES.index(secao)
    pct   = int(idx_s / (len(SECOES) - 1) * 100) if idx_s > 0 else 0
    st.markdown(
        f"<div style='height:3px; width:{pct}%; background:linear-gradient(90deg,{VERDE},#00d4aa);"
        f"border-radius:2px; margin:4px 0 6px 0'></div>"
        f"<small style='color:{CINZA}'>Capítulo {idx_s}/{len(SECOES)-1}</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        f"<small style='color:{CINZA}'>🗓 2018–2025 · B3 / Yahoo Finance<br>"
        f"🛠 Python · cvxpy · scikit-learn<br>📊 Plotly · Streamlit</small>",
        unsafe_allow_html=True,
    )

# Carregar dados
dados, ibov_raw = load_main()
pesos_qp, pesos_en, backtest, card, ml_res, ml_imp, resumo = load_results()
acoes  = [c for c in dados.columns if c != "IBOV"]
testes = [dados.loc[i:f] for i, f in JANELAS]
treino = dados.loc[:"2024-01-31"]


# ══════════════════════════════════════════════════════════════════════════════
# 0 — CAPA
# ══════════════════════════════════════════════════════════════════════════════
if "Capa" in secao:
    st.markdown(f"""
    <div style='padding:56px 0 28px 0; text-align:center;'>
        <div style='font-size:0.85rem; color:{CINZA}; letter-spacing:4px;
                    text-transform:uppercase; margin-bottom:14px;'>
            Bootcamp de Introdução a Data Science · 2025
        </div>
        <h1 style='font-size:3rem; font-weight:900; margin:0 0 10px 0;
                   background:linear-gradient(135deg,{VERDE},{OURO});
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Index Tracking do IBOV
        </h1>
        <div style='font-size:1.1rem; color:{CINZA}; max-width:640px;
                    margin:0 auto; line-height:1.6;'>
            Como replicar o comportamento da bolsa com menos ações,
            menor custo e maior inteligência — do zero ao modelo com ML.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        ["54", "7 anos", "97.3%", "0.234%", "5"],
        ["Ações analisadas", "Dados históricos", "Correlação c/ IBOV",
         "Tracking Error", "Janelas de teste"],
    ):
        col.markdown(
            f"<div class='kpi'>"
            f"<div class='kpi-val'>{val}</div>"
            f"<div class='kpi-lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; margin-bottom:20px;'>
        <span class='badge b-green'>Programação Quadrática (QP)</span>
        <span class='badge b-gold'>Random Forest</span>
        <span class='badge b-blue'>Elastic Net (benchmark)</span>
        <span class='badge b-green'>cvxpy · OSQP</span>
        <span class='badge b-blue'>scikit-learn</span>
        <span class='badge b-gold'>Plotly · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Estrutura da apresentação")
    cols = st.columns(7)
    etapas = [
        ("📋", "O Problema", "Desafio do cliente"),
        ("📊", "Os Dados", "IBOV + 54 ações · 7a"),
        ("🔧", "O Modelo", "Prog. Quadrática"),
        ("📈", "Backtest", "5 janelas OOS"),
        ("⚖️", "Trade-off", "Slider interativo"),
        ("🤖", "ML", "Random Forest"),
        ("🎯", "Recomendação", "Decisão estratégica"),
    ]
    for col, (icon, nome, desc) in zip(cols, etapas):
        col.markdown(
            f"<div class='card' style='text-align:center; padding:14px 6px;'>"
            f"<div style='font-size:1.7rem; margin-bottom:6px'>{icon}</div>"
            f"<div style='font-weight:700; font-size:0.82rem; color:{BRANCO}'>{nome}</div>"
            f"<div style='font-size:0.68rem; color:{CINZA}; margin-top:4px'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 1 — O PROBLEMA
# ══════════════════════════════════════════════════════════════════════════════
elif "Problema" in secao:
    st.markdown("<div class='chapter-num'>01</div>", unsafe_allow_html=True)
    st.title("O Problema do Cliente")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
        #### O que é um índice de mercado?
        O **IBOVESPA (IBOV)** representa o desempenho médio ponderado das principais
        ações da B3 — a Bolsa de Valores do Brasil. É o principal termômetro do mercado.

        #### O desafio dos fundos passivos
        Fundos de investimento passivo querem entregar o retorno *"da bolsa"* aos seus
        cotistas. Mas o IBOV tem **~85 ações** de diferentes setores.

        Manter esse portfólio completo tem custo elevado:
        - Taxas de corretagem em cada ativo
        - Complexidade de rebalanceamento periódico
        - Capital mínimo fragmentado entre muitas posições

        #### A pergunta-problema
        > *"É possível replicar o IBOV com muito **menos ações**, sem perda relevante
        de precisão e com menor custo?"*
        """)

    with col2:
        ibov_acum = (1 + dados["IBOV"]).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ibov_acum.index, y=ibov_acum.values,
            fill="tozeroy", fillcolor=VERDE_A10,
            line=dict(color=VERDE, width=2), name="IBOV",
        ))
        fig.update_layout(
            **PLOTLY_BASE, title="IBOV — Retorno acumulado (2018–2025)",
            height=330, xaxis_title="", yaxis_title="Base 1.0", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success("📐 **Modelo matemático** que seleciona as ações e calcula pesos ótimos automaticamente")
    c2.success("📊 **Backtest rigoroso** em 5 períodos que o modelo nunca viu — validação honesta")
    c3.success("🤖 **ML de rebalanceamento** que decide quando atualizar a carteira com inteligência")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — OS DADOS
# ══════════════════════════════════════════════════════════════════════════════
elif "Dados" in secao:
    st.markdown("<div class='chapter-num'>02</div>", unsafe_allow_html=True)
    st.title("Os Dados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Início", "Jan 2018")
    c2.metric("Fim", "Abr 2025")
    c3.metric("Dias úteis", f"{len(dados):,}")
    c4.metric("Ações candidatas", f"{len(acoes)}")

    tab1, tab2, tab3 = st.tabs(["📈 IBOV", "🏦 Universo de ações", "🔗 Correlações"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            preco = ibov_raw["^BVSP"].dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preco.index, y=preco.values,
                fill="tozeroy", fillcolor=VERDE_A10,
                line=dict(color=VERDE, width=1.5),
            ))
            fig.update_layout(
                **PLOTLY_BASE, title="Índice IBOV — Pontos", height=300,
                xaxis_title="", yaxis_title="Pontos", showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=dados["IBOV"] * 100, nbinsx=80,
                marker_color=VERDE, opacity=0.8,
            ))
            fig2.update_layout(
                **PLOTLY_BASE, title="Distribuição dos retornos diários (%)",
                height=300, xaxis_title="Retorno diário (%)",
                yaxis_title="Frequência", showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        ibov_pct = dados["IBOV"] * 100
        stats = {
            "Retorno médio diário": f"{ibov_pct.mean():.4f}%",
            "Volatilidade diária":  f"{ibov_pct.std():.4f}%",
            "Retorno anualizado":   f"{ibov_pct.mean() * 252:.2f}%",
            "Pior dia":             f"{ibov_pct.min():.2f}%",
            "Melhor dia":           f"{ibov_pct.max():.2f}%",
        }
        st.dataframe(
            pd.DataFrame(stats, index=["Valor"]).T.reset_index().rename(columns={"index":"Métrica"}),
            use_container_width=True, hide_index=True,
        )

    with tab2:
        ret_anual = dados[acoes].mean() * 252 * 100
        vol_anual = dados[acoes].std() * np.sqrt(252) * 100
        scatter_df = pd.DataFrame({
            "Retorno anualizado (%)":    ret_anual,
            "Volatilidade anualizada (%)": vol_anual,
        })
        scatter_df.index = scatter_df.index.str.replace(".SA", "")
        fig3 = px.scatter(
            scatter_df.reset_index(),
            x="Volatilidade anualizada (%)", y="Retorno anualizado (%)",
            text="index", color_discrete_sequence=[VERDE],
            title="Risco × Retorno das 54 ações candidatas",
        )
        fig3.update_traces(textposition="top center", textfont_size=9, marker_size=7)
        fig3.update_layout(**PLOTLY_BASE, height=450)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        sample = acoes[:20]
        corr   = dados[sample].corr()
        corr.index = corr.columns = [c.replace(".SA", "") for c in sample]
        fig4 = px.imshow(
            corr, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
            title="Matriz de correlação — primeiras 20 ações",
        )
        fig4.update_layout(**PLOTLY_BASE, height=500)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — O MODELO QP
# ══════════════════════════════════════════════════════════════════════════════
elif "Modelo" in secao:
    st.markdown("<div class='chapter-num'>03</div>", unsafe_allow_html=True)
    st.title("O Modelo — Programação Quadrática (QP)")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(f"""
        #### Em linguagem simples

        O modelo responde:
        > *"Quanto investir em cada ação para que a carteira
        imite o IBOV o melhor possível?"*

        **Passos:**
        1. Testa combinações de pesos para as ações
        2. Calcula a diferença diária entre carteira e IBOV
        3. Escolhe os pesos que **minimizam** essa diferença ao quadrado
        4. Garante: soma = 100%, sem vender a descoberto, sem concentrar demais

        #### Por que "ao quadrado"?
        Penaliza erros grandes muito mais que erros pequenos.
        Um desvio de 10% pesa **25×** mais que um desvio de 2%.
        Isso força o modelo a evitar dias de grande distanciamento do índice.

        #### Solver
        Usamos `OSQP` via `cvxpy` — open-source, sem custo de licença,
        resolve o problema em frações de segundo.
        """)

    with col2:
        st.markdown("#### Formulação matemática")
        st.latex(r"""
        \min_{w} \; \sum_{t=1}^{T}
        \left(
            \underbrace{\sum_{i=1}^{N} w_i \cdot r_{i,t}}_{\text{retorno da carteira}} -
            \underbrace{r_{\text{IBOV},t}}_{\text{retorno do índice}}
        \right)^2
        """)
        st.markdown("**Sujeito às restrições de portfólio:**")
        st.latex(r"""
        \underbrace{\textstyle\sum_{i} w_i = 1}_{\text{100\% investido}} \qquad
        \underbrace{w_i \geq 0}_{\text{long-only}} \qquad
        \underbrace{w_i \leq \text{cap}}_{\text{limite por ativo}}
        """)
        st.info("O que diferencia QP de uma simples regressão: as restrições de portfólio são incorporadas diretamente no problema de otimização.")

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success("**Pré-seleção:** Correlação + RFECV reduzem ~85 → 54 ações candidatas")
    c2.success("**Ranking:** QP no universo completo gera importância de cada ação por peso ótimo")
    c3.success("**Cardinalidade:** K=5 a 54 testados — escolhemos o menor K dentro de 10% do melhor TE")

    st.markdown("#### Carteira final otimizada — K = 30 ações")
    pq = pesos_qp.copy()
    pq.index = pq.index.str.replace(".SA", "")
    cores_bar = [OURO if v > pq["peso"].mean() else VERDE for v in pq["peso"].values]
    fig = go.Figure(go.Bar(
        x=pq["peso"].values * 100, y=pq.index,
        orientation="h", marker_color=cores_bar,
        text=[f"{v:.1f}%" for v in pq["peso"].values * 100],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=640,
        xaxis_title="Peso na carteira (%)",
        yaxis=dict(**PLOTLY_BASE["yaxis"], autorange="reversed"),
        title="Alocação ótima — carteira QP com K=30",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif "Backtest" in secao:
    st.markdown("<div class='chapter-num'>04</div>", unsafe_allow_html=True)
    st.title("Backtest — Validação Fora da Amostra")

    st.markdown("""
    Os pesos foram otimizados **somente com dados de 2018 a jan/2024** e depois
    aplicados em **5 trimestres consecutivos que o modelo nunca viu**.
    Sem data snooping — a validação mais honesta possível.
    """)

    qp_bt = backtest[backtest["modelo"] == "QP_tracking"]
    en_bt = backtest[backtest["modelo"] == "ElasticNet"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TE médio · QP",           f"{qp_bt['tracking_error'].mean()*100:.4f}%")
    c2.metric("Correlação média · QP",    f"{qp_bt['correlacao'].mean():.2%}")
    c3.metric("TE médio · Elastic Net",   f"{en_bt['tracking_error'].mean()*100:.4f}%")
    c4.metric("Correlação média · E.Net", f"{en_bt['correlacao'].mean():.2%}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Retorno acumulado — 5 janelas de teste")

    fig2 = make_subplots(
        rows=1, cols=5,
        subplot_titles=[f"Janela {k+1} · {j[0][5:]}" for k, j in enumerate(JANELAS)],
        shared_yaxes=False,
    )
    for k, df_t in enumerate(testes):
        if len(df_t) == 0:
            continue
        try:
            rc_qp = df_t[pesos_qp.index].values @ pesos_qp["peso"].values
            rc_en = df_t[pesos_en.index].values @ pesos_en["peso"].values
            ri    = df_t["IBOV"].values
        except Exception:
            continue
        fig2.add_trace(go.Scatter(
            x=df_t.index, y=(1 + rc_qp).cumprod(),
            line=dict(color=VERDE, width=2.5),
            name="Carteira QP", showlegend=(k == 0),
        ), row=1, col=k + 1)
        fig2.add_trace(go.Scatter(
            x=df_t.index, y=(1 + ri).cumprod(),
            line=dict(color=BRANCO, width=1.5, dash="dot"),
            name="IBOV", showlegend=(k == 0),
        ), row=1, col=k + 1)
        fig2.add_trace(go.Scatter(
            x=df_t.index, y=(1 + rc_en).cumprod(),
            line=dict(color=OURO, width=1.5, dash="dash"),
            name="Elastic Net", showlegend=(k == 0),
        ), row=1, col=k + 1)

    fig2.update_layout(
        **PLOTLY_BASE, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.5, xanchor="center"),
    )
    for i in range(1, 6):
        fig2.update_xaxes(tickformat="%b\n%Y", row=1, col=i)
    st.plotly_chart(fig2, use_container_width=True)

    tab_qp, tab_en, tab_comp = st.tabs(["Carteira QP", "Elastic Net", "Comparação"])

    def fmt_bt(df_raw):
        df = df_raw[["janela", "periodo", "tracking_error", "correlacao",
                      "retorno_carteira_acum", "retorno_ibov_acum"]].copy()
        df["tracking_error"]       = df["tracking_error"].map("{:.5f}".format)
        df["correlacao"]           = df["correlacao"].map("{:.4f}".format)
        df["retorno_carteira_acum"] = df["retorno_carteira_acum"].map("{:.2%}".format)
        df["retorno_ibov_acum"]     = df["retorno_ibov_acum"].map("{:.2%}".format)
        df.columns = ["Janela", "Período", "Tracking Error", "Correlação",
                      "Retorno Carteira", "Retorno IBOV"]
        return df

    with tab_qp:
        st.dataframe(fmt_bt(qp_bt), use_container_width=True, hide_index=True)
    with tab_en:
        st.dataframe(fmt_bt(en_bt), use_container_width=True, hide_index=True)
    with tab_comp:
        comp = backtest.groupby("modelo")[["tracking_error", "correlacao"]].mean().reset_index()
        comp.columns = ["Modelo", "TE médio (fração)", "Correlação média"]
        comp["TE médio (%)"]     = (comp["TE médio (fração)"] * 100).map("{:.4f}%".format)
        comp["Correlação média"] = comp["Correlação média"].map("{:.2%}".format)
        st.dataframe(
            comp[["Modelo", "TE médio (%)", "Correlação média"]],
            use_container_width=True, hide_index=True,
        )
        vant = (en_bt["tracking_error"].mean() / qp_bt["tracking_error"].mean() - 1) * 100
        st.success(f"✅ O modelo QP tem Tracking Error **{vant:.0f}% menor** que o Elastic Net.")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — TRADE-OFF DE CARDINALIDADE
# ══════════════════════════════════════════════════════════════════════════════
elif "Trade-off" in secao:
    st.markdown("<div class='chapter-num'>05</div>", unsafe_allow_html=True)
    st.title("Trade-off — Quantas Ações Usar?")

    st.markdown("""
    **Você, o cliente, escolhe** quantas ações quer na carteira.
    O gráfico e a carteira abaixo atualizam em tempo real com a sua escolha.
    """)

    Ks_disp = card["K"].tolist()
    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        K_sel = st.select_slider(
            "Número de ações (K):", options=Ks_disp, value=20,
            help="Mova para ver o impacto na precisão de replicação do IBOV",
        )
        row_k     = card[card["K"] == K_sel].iloc[0]
        te_val    = float(row_k["TE_medio_oos"])
        custo_val = float(row_k["custo_reducao_%TE"])

        if custo_val <= 10:
            cor_c, label_c = VERDE,    "✅ Excelente"
        elif custo_val <= 22:
            cor_c, label_c = OURO,     "⚠️ Aceitável"
        else:
            cor_c, label_c = "#ff4b4b","❌ Custo elevado"

        st.markdown(f"""
        <div class='card' style='text-align:center; margin-top:12px;'>
            <div style='font-size:2rem; font-weight:800; color:{VERDE}'>{te_val:.3f}%</div>
            <div style='font-size:0.75rem; color:{CINZA}; margin-top:4px'>
                Tracking Error médio diário
            </div>
        </div>
        <div class='card' style='text-align:center; margin-top:8px;'>
            <div style='font-size:2rem; font-weight:800; color:{cor_c}'>+{custo_val:.1f}%</div>
            <div style='font-size:0.75rem; color:{CINZA}; margin-top:4px'>
                Custo vs carteira completa (K=54)
            </div>
        </div>
        <div style='text-align:center; margin-top:10px;
                    font-weight:700; font-size:1rem; color:{cor_c}'>{label_c}</div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Curva de eficiência
        fig = go.Figure()
        fig.add_vrect(
            x0=24, x1=32, fillcolor=VERDE, opacity=0.07, line_width=0,
            annotation_text="⭐ Zona recomendada", annotation_position="top left",
            annotation_font_color=VERDE, annotation_font_size=10,
        )
        marker_colors = [OURO if k == K_sel else VERDE for k in card["K"]]
        marker_sizes  = [14   if k == K_sel else 8    for k in card["K"]]
        marker_syms   = ["star" if k == K_sel else "circle" for k in card["K"]]
        fig.add_trace(go.Scatter(
            x=card["K"], y=card["TE_medio_oos"],
            mode="lines+markers",
            line=dict(color=VERDE, width=2.5),
            marker=dict(color=marker_colors, size=marker_sizes, symbol=marker_syms,
                        line=dict(color=BG, width=1.5)),
            hovertemplate="<b>K=%{x} ações</b><br>TE=%{y:.4f}%<extra></extra>",
            name="Tracking Error",
        ))
        fig.add_annotation(
            x=K_sel, y=te_val,
            text=f"  K={K_sel}: {te_val:.3f}%",
            showarrow=False, xanchor="left",
            font=dict(color=OURO, size=11, family="monospace"),
        )
        fig.update_layout(
            **PLOTLY_BASE, height=360,
            title=f"Curva de Eficiência — K={K_sel} destacado (estrela dourada)",
            xaxis_title="Número de ações (K)",
            yaxis_title="Tracking Error médio OOS (%)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Composição dinâmica
    st.markdown(f"#### Composição ótima com **K = {K_sel} ações**")
    with st.spinner(f"Calculando pesos QP para K={K_sel}..."):
        pesos_k = calcular_pesos_k(K_sel)

    pk = pesos_k.copy()
    pk.index = pk.index.str.replace(".SA", "")

    c_bar, c_top = st.columns([3, 1])
    with c_bar:
        fig2 = go.Figure(go.Bar(
            x=pk.index, y=pk.values * 100,
            marker_color=[OURO if v > pk.mean() else VERDE for v in pk.values],
            text=[f"{v:.1f}%" for v in pk.values * 100],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig2.update_layout(
            **PLOTLY_BASE, height=320,
            xaxis_title="Ação", yaxis_title="Peso (%)",
            title=f"Alocação ótima com {K_sel} ações",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c_top:
        st.markdown("**Top 5 posições:**")
        for acao, peso in pk.head(5).items():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:7px 0;"
                f"border-bottom:1px solid {BORDA};'>"
                f"<span style='font-weight:600; color:{BRANCO}'>{acao}</span>"
                f"<span style='color:{VERDE}; font-weight:700'>{peso*100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6 — ML DE REBALANCEAMENTO
# ══════════════════════════════════════════════════════════════════════════════
elif "ML" in secao:
    st.markdown("<div class='chapter-num'>06</div>", unsafe_allow_html=True)
    st.title("ML de Rebalanceamento — Random Forest")

    st.markdown("""
    A carteira é otimizada com dados históricos. Mas o mercado muda — e os pesos
    ficam defasados com o tempo. **Quando vale a pena rebalancear?**

    Treinamos um **Random Forest** para prever, com base no comportamento recente da
    carteira, se o tracking error vai piorar nos próximos 20 dias.
    Se a previsão for "vai piorar" → dispara o rebalanceamento.
    """)

    col1, col2 = st.columns(2)

    with col1:
        feat_labels = {
            "TE_med_60d":          "TE médio (60 dias)",
            "TE_vol_20d":          "Volatilidade do TE (20d)",
            "Dias_acima_mediana":  "Dias com TE acima da mediana",
            "TE_med_20d":          "TE médio (20 dias)",
            "IBOV_ret_20d":        "Retorno do IBOV (20d)",
            "IBOV_vol_20d":        "Volatilidade do IBOV (20d)",
            "Corr_20d":            "Correlação carteira-IBOV (20d)",
        }
        fi = ml_imp.copy()
        fi.columns = ["Feature", "Importância"]
        fi["Feature"] = fi["Feature"].map(lambda x: feat_labels.get(x, x))
        fi = fi.sort_values("Importância")

        fig = go.Figure(go.Bar(
            x=fi["Importância"], y=fi["Feature"],
            orientation="h",
            marker=dict(
                color=fi["Importância"],
                colorscale=[[0, AZUL], [1, VERDE]],
                showscale=False,
            ),
            text=[f"{v:.1%}" for v in fi["Importância"]],
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_BASE, height=320,
            title="Quais sinais o modelo observa?",
            xaxis_title="Importância (Gini)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 3 estratégias comparadas")
        cores_strat = {
            "Estática":          CINZA,
            "Sempre-Rebalanceia": "#58a6ff",
            "ML-Guiada":         VERDE,
        }
        for _, row in ml_res.iterrows():
            nome  = row["Estratégia"]
            cor   = cores_strat.get(nome, CINZA)
            borda = f"border: 2px solid {VERDE};" if "ML" in nome else ""
            n_reb = int(row["Nº_rebalanceamentos"])
            st.markdown(
                f"<div class='card' style='{borda} margin-bottom:8px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='font-weight:700; color:{cor}; font-size:1.05rem'>{nome}</span>"
                f"<span style='color:{CINZA}; font-size:0.8rem'>{n_reb} rebalanceamento(s)</span>"
                f"</div>"
                f"<div style='display:flex; gap:22px; margin-top:10px;'>"
                f"<div><div style='font-size:1.45rem; font-weight:700; color:{cor}'>"
                f"{row['TE_médio']:.4f}%</div>"
                f"<div style='font-size:0.7rem; color:{CINZA}'>Tracking Error</div></div>"
                f"<div><div style='font-size:1.45rem; font-weight:700; color:{BRANCO}'>"
                f"{row['Correlação_média']:.4f}</div>"
                f"<div style='font-size:0.7rem; color:{CINZA}'>Correlação IBOV</div></div>"
                f"<div><div style='font-size:1.45rem; font-weight:700; color:{BRANCO}'>"
                f"{row['Retorno_carteira_%']:.2f}%</div>"
                f"<div style='font-size:0.7rem; color:{CINZA}'>Retorno acum.</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    te_est  = ml_res[ml_res["Estratégia"] == "Estática"]["TE_médio"].values[0]
    te_ml   = ml_res[ml_res["Estratégia"] == "ML-Guiada"]["TE_médio"].values[0]
    te_semp = ml_res[ml_res["Estratégia"] == "Sempre-Rebalanceia"]["TE_médio"].values[0]
    red     = (te_est - te_ml) / te_est * 100
    dif_ml  = abs(te_ml - te_semp) / te_semp * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Redução de TE (ML vs Estática)",      f"{red:.1f}%",  "↓ melhora")
    c2.metric("Diferença ML vs Sempre-Rebalanceia",  f"{dif_ml:.2f}%","≈ equivalente")
    c3.metric("Rebalanceamentos poupados",            "1",            "ML vs Sempre-Rebalanceia")

    st.info(
        "💡 **Descoberta principal:** qualquer rebalanceamento reduz o tracking error em ~18%. "
        "A estratégia **ML-Guiada** alcança o mesmo resultado que 'Sempre-Rebalanceia' "
        "com **1 evento a menos** — menor custo operacional e transacional."
    )

    try:
        st.image(
            "reports/figures/ml_rebalanceamento_backtest.png",
            caption="Comparação das 3 estratégias nas 5 janelas de teste",
            use_container_width=True,
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 7 — RECOMENDAÇÃO FINAL
# ══════════════════════════════════════════════════════════════════════════════
elif "Recomendação" in secao:
    st.markdown("<div class='chapter-num'>07</div>", unsafe_allow_html=True)
    st.title("Recomendação Final")

    st.markdown("""
    Com base nos resultados de backtest, análise de trade-off e ML,
    recomendamos a seguinte matriz de decisão estratificada por perfil de uso.
    """)

    c1, c2, c3 = st.columns(3)
    perfis = [
        (CINZA,   "K = 15–20",   "ETF simplificado · Carteira piloto", False,
         ["✅ Mais simples de gerir",
          "⚡ Custo operacional mínimo",
          "⚠️ TE +19–22% acima do ideal",
          "📉 Correlação ~95–96% com IBOV"]),
        (OURO,    "K = 25–30 ⭐","Fundo varejo — equilíbrio ideal", True,
         ["✅ Melhor custo-benefício comprovado",
          "✅ TE apenas +7–12% acima do ideal",
          "✅ Correlação ~97% com IBOV",
          "✅ Gestão ainda acessível"]),
        (AZUL,    "K = 40–54",   "Fundo institucional — máxima precisão", False,
         ["✅ Mínimo tracking error possível",
          "✅ Correlação 97.7% com IBOV",
          "⚠️ Maior custo de transação",
          "⚠️ Mais ações para gerir"]),
    ]
    for col, (cor, titulo, sub, dest, itens) in zip([c1, c2, c3], perfis):
        borda = f"border: 2px solid {cor};" if dest else f"border: 1px solid {BORDA};"
        col.markdown(
            f"<div class='card' style='{borda}'>"
            f"<div style='color:{cor}; font-size:1.3rem; font-weight:800; margin-bottom:4px'>{titulo}</div>"
            f"<div style='color:{CINZA}; font-size:0.8rem; margin-bottom:12px'>{sub}</div>"
            + "".join(
                f"<div style='font-size:0.84rem; margin:5px 0; color:{BRANCO}'>{it}</div>"
                for it in itens
            ) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ O que foi entregue")
        entregues = [
            ("🔧 Modelo QP",           "Mínimo tracking error, solver open-source (OSQP), sem licença"),
            ("📊 Backtest rigoroso",    "5 janelas trimestrais 100% fora da amostra"),
            ("⚖️ Análise de trade-off", "11 tamanhos de carteira (K=5 a 54) testados"),
            ("🤖 ML de rebalanceamento","Random Forest reduz TE em ~18% com intervenções mínimas"),
            ("📉 Benchmark comparativo","Elastic Net superado em todos os critérios pelo modelo QP"),
        ]
        for nome, desc in entregues:
            st.markdown(
                f"<div style='display:flex; gap:10px; margin:9px 0; align-items:flex-start;'>"
                f"<div style='font-weight:700; color:{VERDE}; min-width:190px; flex-shrink:0'>{nome}</div>"
                f"<div style='color:{CINZA}; font-size:0.87rem'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 🔭 Próximos passos sugeridos")
        proximos = [
            "Rebalanceamento dinâmico de K por regime de mercado",
            "MIQP com cardinalidade explícita (variáveis binárias, Gurobi/CPLEX)",
            "Incorporar custos de transação e spread na função objetivo",
            "Rolling Window contínuo — reotimizar pesos mensalmente",
            "Extensão para outros índices: S&P100, MSCI EM",
        ]
        for item in proximos:
            st.markdown(
                f"<div style='padding:8px 0; border-bottom:1px solid {BORDA};"
                f"color:{CINZA}; font-size:0.87rem'>▸ {item}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Números finais do projeto")

    c1, c2, c3, c4, c5 = st.columns(5)
    te_qp   = resumo["tracking_error_medio_qp"]
    te_en   = resumo["tracking_error_medio_enet"]
    cor_qp  = resumo["correlacao_media_qp"]
    K_esc   = resumo["K_escolhido"]
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [str(K_esc), f"{cor_qp:.1%}", f"{te_qp*100:.4f}%",
         f"{te_en/te_qp:.1f}×", "~18%"],
        ["K escolhido", "Correlação QP", "TE médio QP",
         "Superioridade vs E.Net", "Melhoria c/ ML"],
    ):
        col.markdown(
            f"<div class='kpi'>"
            f"<div class='kpi-val'>{val}</div>"
            f"<div class='kpi-lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center; padding:22px; background:{BG2}; border-radius:12px;"
        f"border:1px solid {VERDE}40;'>"
        f"<div style='font-size:1.05rem; font-weight:600; color:{VERDE}'>"
        f"🎓 Projeto desenvolvido no Bootcamp de Introdução a Data Science — FINOR 2025</div>"
        f"<div style='font-size:0.82rem; color:{CINZA}; margin-top:6px'>"
        f"Dados: B3 / Yahoo Finance · 2018–2025 · Python · cvxpy · scikit-learn · Streamlit · Plotly"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 📚 Referências")
    st.markdown(
        f"<div style='color:{CINZA}; font-size:0.83rem; line-height:2.1;'>"
        "Cornuejols, G.; Tütüncü, R. <em>Optimization Methods in Finance</em>. Cambridge University Press, 2006.<br>"
        "Santanna, L.; Filomena, T. P.; Borenstein, D. "
        "<em>Index Tracking com Controle do Número de Ativos</em>. "
        "Revista Brasileira de Finanças, v. 12, 2014.<br>"
        "Santanna, L. et al. "
        "<em>Index tracking with controlled number of assets using a hybrid heuristic</em>. "
        "Annals of Operations Research, v. 258, 2017."
        "</div>",
        unsafe_allow_html=True,
    )
