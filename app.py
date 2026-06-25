#!/usr/bin/env python3
"""
FINOR | Index Tracking IBOV — Plataforma e Dashboard de Storytelling

Execução:
    cd /caminho/para/Bootcamp-Finor
    streamlit run app.py
"""
import json
import io
import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cvxpy as cp
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FINOR | Index Tracking IBOV",
    page_icon="assets/finor_mark.png" if os.path.exists("assets/finor_mark.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CORES — identidade visual Finor (extraída de finor.tech)
# ══════════════════════════════════════════════════════════════════════════════
VERDE  = "#EE7F25"   # laranja primário da marca Finor (nomes de variável mantidos)
AZUL   = "#6E7B8B"   # cinza-azulado neutro, usado em séries secundárias de gráficos
OURO   = "#F2953A"   # laranja claro (gradiente da marca), usado para destaque
BRANCO = "#F2F2F2"
CINZA  = "#9A9A9A"
BG     = "#0d0d0d"
BG2    = "#1a1a1a"
BORDA  = "#2a2a2a"

# Versões RGBA para uso em propriedades Plotly (não aceita hex 8 dígitos)
VERDE_A10 = "rgba(238, 127, 37, 0.10)"
VERDE_A07 = "rgba(238, 127, 37, 0.07)"
VERDE_A20 = "rgba(238, 127, 37, 0.20)"

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
.b-blue  {{ background: #6E7B8B40; color: #aab4bf; border: 1px solid #6E7B8B50; }}
.kpi {{ text-align: center; padding: 14px 8px; }}
.kpi-val {{ font-size: 1.9rem; font-weight: 700; color: {VERDE}; }}
.kpi-lbl {{ font-size: 0.75rem; color: {CINZA}; margin-top: 3px; }}
.ajuda, span.ajuda {{
    cursor: help !important;
    pointer-events: auto !important;
    color: {CINZA};
    font-size: 0.72rem;
    border: 1px solid {CINZA};
    border-radius: 50%;
    width: 15px;
    height: 15px;
    min-width: 15px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-left: 6px;
    vertical-align: middle;
    user-select: none;
}}
.ajuda:hover {{
    cursor: help !important;
    border-color: {VERDE};
    color: {VERDE};
}}
hr {{ border-color: {BORDA}; margin: 24px 0; }}
</style>
""", unsafe_allow_html=True)


_TAMANHOS_TITULO = {
    "####": ("1.05rem", "600"),
    "###":  ("1.3rem",  "700"),
    "##":   ("1.6rem",  "800"),
    "":     ("0.95rem", "600"),
}

def titulo_ajuda(texto, tooltip, nivel="####"):
    """Renderiza um título de seção/gráfico com um ícone de ajuda (?) ao lado,
    que mostra uma explicação ao passar o mouse — sem depender de emojis.

    Importante: isto NÃO usa sintaxe de cabeçalho markdown (#, ##, ...) porque
    o Streamlit injeta automaticamente um link de âncora em cabeçalhos, o que
    faz o ícone de ajuda (e o título) parecer "não clicável" ao passar o mouse.
    Em vez disso, o título é simulado com um <div> estilizado.
    """
    tam, peso = _TAMANHOS_TITULO.get(nivel, _TAMANHOS_TITULO["####"])
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:0; margin:0.3em 0 0.5em 0;'>"
        f"<span style='font-size:{tam}; font-weight:{peso}; color:{BRANCO};'>{texto}</span>"
        f"<span class='ajuda' title='{tooltip}'>?</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
# FUNÇÕES DA PLATAFORMA — recomendação de K, simulação, ML, persistência,
# relatório PDF, projeções de Monte Carlo e monitoramento de regime/anomalias
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def recomendar_k(universo, treino_ini, treino_fim, cap_max):
    """Recomenda o menor K cuja perda de precisão fique a até 10% do melhor
    Tracking Error possível. Validado com um split interno 80/20 dentro do
    próprio período de treino — nunca usa o período de teste do cliente,
    para evitar viés (data leakage)."""
    dados, _ = load_main()
    janela = dados.loc[treino_ini:treino_fim]
    universo = [a for a in universo if a in janela.columns]
    n = len(universo)
    if len(janela) < 40 or n < 5:
        return min(30, n) if n else 10

    corte = int(len(janela) * 0.8)
    treino_int = janela.iloc[:corte]
    valid_int  = janela.iloc[corte:]
    if len(valid_int) < 10:
        return min(30, n)

    X_tr, y_tr = treino_int[universo].values, treino_int["IBOV"].values
    w_all = cp.Variable(n)
    cp.Problem(
        cp.Minimize(cp.sum_squares(X_tr @ w_all - y_tr)),
        [cp.sum(w_all) == 1, w_all >= 0, w_all <= cap_max],
    ).solve(solver=cp.OSQP)
    ranking = pd.Series(np.maximum(w_all.value, 0), index=universo).sort_values(ascending=False)

    candidatos = sorted(set([k for k in [10, 15, 20, 25, 30, 40] if k < n] + [n]))
    resultados = {}
    for K in candidatos:
        top = ranking.head(K).index.tolist()
        cap_efetivo = max(cap_max, 1.0 / K)
        w_k = cp.Variable(K)
        cp.Problem(
            cp.Minimize(cp.sum_squares(treino_int[top].values @ w_k - y_tr)),
            [cp.sum(w_k) == 1, w_k >= 0, w_k <= cap_efetivo],
        ).solve(solver=cp.OSQP)
        if w_k.value is None:
            continue
        pesos_k = pd.Series(np.maximum(w_k.value, 0), index=top)
        soma = pesos_k.sum()
        if soma <= 0:
            continue
        pesos_k = pesos_k / soma
        rc = valid_int[top].values @ pesos_k.values
        ri = valid_int["IBOV"].values
        resultados[K] = float(np.std(rc - ri))

    if not resultados:
        return min(30, n)
    melhor_te = min(resultados.values())
    candidatos_ok = [K for K, te in resultados.items() if te <= melhor_te * 1.10]
    return min(candidatos_ok)


@st.cache_data(show_spinner=False)
def rodar_simulacao(universo, K, cap_max, treino_ini, treino_fim, teste_ini, teste_fim):
    """Roda uma simulação completa: ranking por QP no universo escolhido,
    seleção das K ações de maior peso, novo QP com restrição de concentração,
    e avaliação fora da amostra no período de teste."""
    dados, _ = load_main()
    universo = [a for a in universo if a in dados.columns]
    treino_j = dados.loc[treino_ini:treino_fim]
    teste_j  = dados.loc[teste_ini:teste_fim]

    if len(teste_j) == 0 or len(treino_j) < 30 or len(universo) < 5:
        return {"teste_vazio": True}

    n = len(universo)
    X_tr, y_tr = treino_j[universo].values, treino_j["IBOV"].values
    w_all = cp.Variable(n)
    cp.Problem(
        cp.Minimize(cp.sum_squares(X_tr @ w_all - y_tr)),
        [cp.sum(w_all) == 1, w_all >= 0, w_all <= cap_max],
    ).solve(solver=cp.OSQP)
    ranking = pd.Series(np.maximum(w_all.value, 0), index=universo).sort_values(ascending=False)

    K = int(np.clip(K, 1, n))
    top = ranking.head(K).index.tolist()
    cap_efetivo = max(cap_max, 1.0 / K)
    w_k = cp.Variable(K)
    cp.Problem(
        cp.Minimize(cp.sum_squares(treino_j[top].values @ w_k - y_tr)),
        [cp.sum(w_k) == 1, w_k >= 0, w_k <= cap_efetivo],
    ).solve(solver=cp.OSQP)
    if w_k.value is None:
        return {"teste_vazio": True}
    pesos = pd.Series(np.maximum(w_k.value, 0), index=top)
    pesos = (pesos / pesos.sum()).sort_values(ascending=False)

    rc = teste_j[top].values @ pesos.values
    ri = teste_j["IBOV"].values
    te = float(np.std(rc - ri))
    corr = float(np.corrcoef(rc, ri)[0, 1]) if len(rc) > 1 else float("nan")
    serie_carteira = pd.Series((1 + rc).cumprod(), index=teste_j.index)
    serie_ibov     = pd.Series((1 + ri).cumprod(), index=teste_j.index)

    return {
        "teste_vazio": False,
        "pesos": pesos,
        "tracking_error": te,
        "correlacao": corr,
        "retorno_carteira_acum": float(serie_carteira.iloc[-1] - 1),
        "retorno_ibov_acum": float(serie_ibov.iloc[-1] - 1),
        "serie_carteira": serie_carteira,
        "serie_ibov": serie_ibov,
    }


def gerar_sinal_ml(pesos, teste_ini, teste_fim):
    """Treina um Random Forest com a mesma arquitetura validada offline
    (LOOK=60, STEP=20, HORIZ=20, 7 features) usando apenas dados anteriores
    ao início do teste, e aplica o sinal a cada dia do período de teste.
    Retorna None se não houver histórico suficiente para treinar com segurança."""
    dados, _ = load_main()
    LOOK, STEP, HORIZ = 60, 20, 20
    ativos = [a for a in pesos.index if a in dados.columns]
    hist = dados.loc[:teste_ini].iloc[:-1]
    if len(hist) < LOOK + HORIZ + STEP * 10:
        return None

    rc_h = hist[ativos].values @ pesos.loc[ativos].values
    ri_h = hist["IBOV"].values
    te_h = np.abs(rc_h - ri_h)

    def feats_em(i, mediana_ref):
        janela = te_h[i - LOOK:i]
        return [
            float(np.mean(janela[-20:])),
            float(np.mean(janela)),
            float(np.std(janela[-20:])),
            float(np.std(ri_h[i - 20:i])),
            float(np.mean(ri_h[i - 20:i])),
            float(np.corrcoef(rc_h[i - 20:i], ri_h[i - 20:i])[0, 1]),
            float(np.sum(janela > mediana_ref)),
        ]

    X_ml, y_ml = [], []
    for i in range(LOOK, len(te_h) - HORIZ, STEP):
        mediana_ref = np.median(te_h[:i])
        X_ml.append(feats_em(i, mediana_ref))
        y_ml.append(1 if np.mean(te_h[i:i + HORIZ]) > mediana_ref else 0)

    if len(X_ml) < 20:
        return None
    X_ml, y_ml = np.array(X_ml), np.array(y_ml)
    rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
    rf.fit(X_ml, y_ml)
    mediana_geral = float(np.median(te_h))

    teste_j = dados.loc[teste_ini:teste_fim]
    sinais = []
    for data_corte in teste_j.index:
        janela_hist = dados.loc[:data_corte].iloc[-LOOK - 1:-1]
        if len(janela_hist) < LOOK:
            continue
        rc_w = janela_hist[ativos].values @ pesos.loc[ativos].values
        ri_w = janela_hist["IBOV"].values
        te_w = np.abs(rc_w - ri_w)
        feats_live = np.array([[
            float(np.mean(te_w[-20:])), float(np.mean(te_w)),
            float(np.std(te_w[-20:])), float(np.std(ri_w[-20:])),
            float(np.mean(ri_w[-20:])),
            float(np.corrcoef(rc_w[-20:], ri_w[-20:])[0, 1]),
            float(np.sum(te_w > mediana_geral)),
        ]])
        sinais.append({"data": data_corte, "sinal": int(rf.predict(feats_live)[0])})

    return {"sinais": sinais, "n_treino": len(X_ml)}


HIST_FILE = "data/processed/historico_simulacoes_cliente.json"

def carregar_historico_persistente():
    if not os.path.exists(HIST_FILE):
        return []
    try:
        with open(HIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

def registrar_simulacao(resultado, lote_id=None):
    historico = carregar_historico_persistente()
    pesos = resultado["pesos"]
    entrada = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().isoformat(),
        "lote_id": lote_id,
        "K": resultado.get("K"),
        "tracking_error": resultado["tracking_error"],
        "correlacao": resultado["correlacao"],
        "retorno_carteira_acum": resultado["retorno_carteira_acum"],
        "retorno_ibov_acum": resultado["retorno_ibov_acum"],
        "params": resultado.get("params", {}),
        "pesos": {k: float(v) for k, v in pesos.items()},
    }
    historico.append(entrada)
    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2)


def _pdf_safe(texto):
    substituicoes = {
        "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "•": "-",
    }
    for k, v in substituicoes.items():
        texto = texto.replace(k, v)
    return texto.encode("latin-1", errors="replace").decode("latin-1")

def gerar_pdf_relatorio(resultado):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(238, 127, 37)
    pdf.cell(0, 12, _pdf_safe("FINOR - Relatorio de Simulacao de Index Tracking"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, _pdf_safe(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe("Parametros da simulacao"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = resultado.get("params", {})
    for linha in [
        f"Numero de acoes (K): {resultado.get('K', '-')}",
        f"Universo: {params.get('universo_n', '-')} acoes candidatas",
        f"Concentracao maxima por ativo: {params.get('cap', 0):.0%}",
        f"Janela de treino: {params.get('janela_anos', '-')} ano(s)",
        f"Frequencia de rebalanceamento: {params.get('freq_reb', '-')}",
        f"Periodo de treino: {params.get('treino', '-')}",
        f"Periodo de teste: {params.get('teste', '-')}",
    ]:
        pdf.cell(0, 6, _pdf_safe(linha), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe("Resultados"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for linha in [
        f"Tracking Error: {resultado['tracking_error']*100:.4f}%",
        f"Correlacao com IBOV: {resultado['correlacao']:.2%}",
        f"Retorno da carteira no periodo: {resultado['retorno_carteira_acum']:.2%}",
        f"Retorno do IBOV no periodo: {resultado['retorno_ibov_acum']:.2%}",
    ]:
        pdf.cell(0, 6, _pdf_safe(linha), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _pdf_safe("Composicao da carteira"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for ativo, peso in resultado["pesos"].sort_values(ascending=False).items():
        pdf.cell(0, 5.5, _pdf_safe(f"{ativo.replace('.SA','')}: {peso*100:.2f}%"), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def montecarlo_projecao(pesos, horizonte, n_sim, bloco, seed=42):
    """Block bootstrap: reamostra blocos contíguos do histórico real de
    divergência (carteira - IBOV) para projetar cenários futuros plausíveis,
    preservando parte da autocorrelação real do mercado."""
    dados, _ = load_main()
    ativos = [a for a in pesos.index if a in dados.columns]
    rc_hist = dados[ativos].values @ pesos.loc[ativos].values
    ri_hist = dados["IBOV"].values
    diff_hist = rc_hist - ri_hist

    rng = np.random.default_rng(seed)
    n = len(diff_hist)
    caminhos = np.zeros((n_sim, horizonte))
    for s in range(n_sim):
        dias = []
        while len(dias) < horizonte:
            inicio = rng.integers(0, max(1, n - bloco))
            dias.extend(diff_hist[inicio:inicio + bloco])
        caminhos[s] = np.cumsum(dias[:horizonte])

    percentis_diff = {p: np.percentile(caminhos, p, axis=0) for p in [5, 25, 50, 75, 95]}
    te_hist = float(np.std(diff_hist))
    limite = 2 * te_hist * np.sqrt(horizonte)
    prob_divergencia = float(np.mean(np.abs(caminhos[:, -1]) > limite))

    return {
        "horizonte": horizonte, "n_sim": n_sim, "bloco": bloco,
        "percentis_diff": percentis_diff, "prob_divergencia": prob_divergencia,
    }


def detectar_regime_anomalias(pesos):
    """Classifica o regime de volatilidade atual do IBOV (limiares sobre a
    volatilidade móvel) e detecta anomalias de tracking error via z-score
    (|z| > 3) na série de erro absoluto diário da carteira."""
    dados, _ = load_main()
    ativos = [a for a in pesos.index if a in dados.columns]
    rc = dados[ativos].values @ pesos.loc[ativos].values
    ri = dados["IBOV"].values
    te_abs = pd.Series(np.abs(rc - ri), index=dados.index)

    vol_ibov_20d = (dados["IBOV"].rolling(20).std() * np.sqrt(252) * 100).dropna()
    vol_mediana = float(vol_ibov_20d.median()) if len(vol_ibov_20d) else None
    vol_atual = float(vol_ibov_20d.iloc[-1]) if len(vol_ibov_20d) else None

    if vol_atual is None or vol_mediana is None:
        regime_atual = "Indefinido"
    elif vol_atual < vol_mediana * 0.75:
        regime_atual = "Calmo"
    elif vol_atual > vol_mediana * 1.35:
        regime_atual = "Volátil"
    else:
        regime_atual = "Normal"

    media_te = te_abs.rolling(60, min_periods=20).mean()
    std_te = te_abs.rolling(60, min_periods=20).std().replace(0, np.nan)
    z = (te_abs - media_te) / std_te
    anomalias = te_abs[z.abs() > 3].dropna()

    return {
        "regime_atual": regime_atual, "vol_atual": vol_atual, "vol_mediana": vol_mediana,
        "vol_ibov_20d": vol_ibov_20d, "te_abs": te_abs, "anomalias": anomalias,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
JANELAS = [
    ("2024-02-01","2024-04-30"), ("2024-05-01","2024-07-31"),
    ("2024-08-01","2024-10-31"), ("2024-11-01","2025-01-31"),
    ("2025-02-01","2025-04-30"),
]
SECOES = [
    "Capa",
    "1 — O Problema",
    "2 — Os Dados",
    "3 — O Modelo (QP)",
    "4 — Backtest",
    "5 — Trade-off de Cardinalidade",
    "6 — ML de Rebalanceamento",
    "7 — Recomendação Final",
]

with st.sidebar:
    if os.path.exists("assets/finor_mark.png"):
        c_logo1, c_logo2, c_logo3 = st.columns([1, 1, 1])
        with c_logo2:
            st.image("assets/finor_mark.png", width=36)
    st.markdown(f"""
    <div style='text-align:center; padding:2px 0 10px 0;'>
        <div style='font-weight:700; font-size:1.05rem; letter-spacing:1px; color:{BRANCO}'>FINOR</div>
        <div style='font-size:0.72rem; color:{CINZA}'>Index Tracking · Plataforma</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    area = st.radio(
        "Área", ["Sobre o Projeto", "Plataforma"],
        label_visibility="hidden",
        help="\"Sobre o Projeto\" apresenta a metodologia em formato de história. "
             "\"Plataforma\" é a ferramenta interativa para simular carteiras.",
    )
    st.markdown("---")

    if area == "Sobre o Projeto":
        secao = st.radio("", SECOES, label_visibility="hidden")
        idx_s = SECOES.index(secao)
        pct   = int(idx_s / (len(SECOES) - 1) * 100) if idx_s > 0 else 0
        st.markdown(
            f"<div style='height:3px; width:{pct}%; background:linear-gradient(90deg,{VERDE},{OURO});"
            f"border-radius:2px; margin:4px 0 6px 0'></div>"
            f"<small style='color:{CINZA}'>Seção {idx_s}/{len(SECOES)-1}</small>",
            unsafe_allow_html=True,
        )
    else:
        secao = ""
        st.markdown(
            f"<small style='color:{CINZA}'>Navegue pelas abas no topo da tela</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"<small style='color:{CINZA}'>2018–2025 · B3 / Yahoo Finance<br>"
        f"Python · cvxpy · scikit-learn · Plotly · Streamlit</small>",
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
if area == "Sobre o Projeto" and "Capa" in secao:
    st.markdown(f"""
    <div style='padding:56px 0 28px 0; text-align:center;'>
        <div style='font-size:0.85rem; color:{CINZA}; letter-spacing:4px;
                    text-transform:uppercase; margin-bottom:14px;'>
            FINOR · Index Tracking
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
        ("O Problema", "Desafio do cliente"),
        ("Os Dados", "IBOV + 54 ações · 7a"),
        ("O Modelo", "Prog. Quadrática"),
        ("Backtest", "5 janelas OOS"),
        ("Trade-off", "Slider interativo"),
        ("ML", "Random Forest"),
        ("Recomendação", "Decisão estratégica"),
    ]
    for col, (nome, desc) in zip(cols, etapas):
        col.markdown(
            f"<div class='card' style='text-align:center; padding:14px 6px;'>"
            f"<div style='font-weight:700; font-size:0.82rem; color:{BRANCO}'>{nome}</div>"
            f"<div style='font-size:0.68rem; color:{CINZA}; margin-top:4px'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 1 — O PROBLEMA
# ══════════════════════════════════════════════════════════════════════════════
elif area == "Sobre o Projeto" and "Problema" in secao:
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
        titulo_ajuda(
            "Retorno acumulado do IBOV",
            "Mostra como R$1 investido no índice cresceria ao longo do tempo, "
            "multiplicando (1 + retorno diário) dia após dia desde 2018.",
        )
        ibov_acum = (1 + dados["IBOV"]).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ibov_acum.index, y=ibov_acum.values,
            fill="tozeroy", fillcolor=VERDE_A10,
            line=dict(color=VERDE, width=2), name="IBOV",
        ))
        fig.update_layout(
            **PLOTLY_BASE, height=330, xaxis_title="", yaxis_title="Base 1.0", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success("**Modelo matemático** que seleciona as ações e calcula pesos ótimos automaticamente")
    c2.success("**Backtest rigoroso** em 5 períodos que o modelo nunca viu — validação honesta")
    c3.success("**ML de rebalanceamento** que decide quando atualizar a carteira com inteligência")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — OS DADOS
# ══════════════════════════════════════════════════════════════════════════════
elif area == "Sobre o Projeto" and "Dados" in secao:
    st.markdown("<div class='chapter-num'>02</div>", unsafe_allow_html=True)
    st.title("Os Dados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Início", "Jan 2018")
    c2.metric("Fim", "Abr 2025")
    c3.metric("Dias úteis", f"{len(dados):,}")
    c4.metric("Ações candidatas", f"{len(acoes)}")

    tab1, tab2, tab3 = st.tabs(["IBOV", "Universo de ações", "Correlações"])

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
elif area == "Sobre o Projeto" and "Modelo" in secao:
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

    titulo_ajuda(
        "Carteira final otimizada — K = 30 ações",
        "Peso de cada ação na carteira resolvida pelo QP. Barras em laranja claro "
        "indicam posições acima do peso médio da carteira.",
    )
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
    _base_sem_yaxis = {k: v for k, v in PLOTLY_BASE.items() if k != "yaxis"}
    fig.update_layout(
        **_base_sem_yaxis, height=640,
        xaxis_title="Peso na carteira (%)",
        yaxis=dict(**PLOTLY_BASE["yaxis"], autorange="reversed"),
        title="Alocação ótima — carteira QP com K=30",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif area == "Sobre o Projeto" and "Backtest" in secao:
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
    titulo_ajuda(
        "Retorno acumulado — 5 janelas de teste",
        "Cada painel é um trimestre fora da amostra. Compara o crescimento de "
        "R$1 na carteira QP, no IBOV e no Elastic Net dentro daquele período.",
    )

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
            line=dict(color=AZUL, width=1.5, dash="dash"),
            name="Elastic Net", showlegend=(k == 0),
        ), row=1, col=k + 1)

    _base_sem_legend = {k: v for k, v in PLOTLY_BASE.items() if k != "legend"}
    fig2.update_layout(
        **_base_sem_legend, height=380,
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
        st.success(f"O modelo QP tem Tracking Error **{vant:.0f}% menor** que o Elastic Net.")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — TRADE-OFF DE CARDINALIDADE
# ══════════════════════════════════════════════════════════════════════════════
elif area == "Sobre o Projeto" and "Trade-off" in secao:
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
            help="Quantas ações a carteira terá. Menos ações = mais simples e "
                 "barato de operar, porém maior erro de réplica (Tracking Error) "
                 "em relação ao IBOV.",
        )
        row_k     = card[card["K"] == K_sel].iloc[0]
        te_val    = float(row_k["TE_medio_oos"])
        custo_val = float(row_k["custo_reducao_%TE"])

        if custo_val <= 10:
            cor_c, label_c = VERDE,    "Excelente"
        elif custo_val <= 22:
            cor_c, label_c = OURO,     "Aceitável"
        else:
            cor_c, label_c = "#ff4b4b","Custo elevado"

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
        titulo_ajuda(
            "Curva de eficiência",
            "Cada ponto é um K testado em backtest fora da amostra. A estrela "
            "marca o K escolhido no controle ao lado. Quanto mais à esquerda e "
            "mais baixo, melhor o equilíbrio entre simplicidade e precisão.",
        )
        fig = go.Figure()
        fig.add_vrect(
            x0=24, x1=32, fillcolor=VERDE, opacity=0.07, line_width=0,
            annotation_text="Zona recomendada", annotation_position="top left",
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
            title=f"Curva de Eficiência — K={K_sel} destacado",
            xaxis_title="Número de ações (K)",
            yaxis_title="Tracking Error médio OOS (%)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

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
elif area == "Sobre o Projeto" and "ML" in secao:
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

        titulo_ajuda(
            "Quais sinais o modelo observa?",
            "Importância (Gini) de cada variável usada pelo Random Forest para "
            "decidir se vale a pena rebalancear a carteira.",
        )
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
            xaxis_title="Importância (Gini)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 3 estratégias comparadas")
        cores_strat = {
            "Estática":          CINZA,
            "Sempre-Rebalanceia": AZUL,
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
    c1.metric("Redução de TE (ML vs Estática)",      f"{red:.1f}%",  "melhora")
    c2.metric("Diferença ML vs Sempre-Rebalanceia",  f"{dif_ml:.2f}%","equivalente")
    c3.metric("Rebalanceamentos poupados",            "1",            "ML vs Sempre-Rebalanceia")

    st.info(
        "**Descoberta principal:** qualquer rebalanceamento reduz o tracking error em ~18%. "
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
elif area == "Sobre o Projeto" and "Recomendação" in secao:
    st.markdown("<div class='chapter-num'>07</div>", unsafe_allow_html=True)
    st.title("Recomendação Final")

    st.markdown("""
    Com base nos resultados de backtest, análise de trade-off e ML,
    recomendamos a seguinte matriz de decisão estratificada por perfil de uso.
    """)

    c1, c2, c3 = st.columns(3)
    perfis = [
        (CINZA,   "K = 15–20",   "ETF simplificado · Carteira piloto", False,
         ["Mais simples de gerir",
          "Custo operacional mínimo",
          "TE +19–22% acima do ideal",
          "Correlação ~95–96% com IBOV"]),
        (OURO,    "K = 25–30",   "Fundo varejo — equilíbrio ideal", True,
         ["Melhor custo-benefício comprovado",
          "TE apenas +7–12% acima do ideal",
          "Correlação ~97% com IBOV",
          "Gestão ainda acessível"]),
        (AZUL,    "K = 40–54",   "Fundo institucional — máxima precisão", False,
         ["Mínimo tracking error possível",
          "Correlação 97.7% com IBOV",
          "Maior custo de transação",
          "Mais ações para gerir"]),
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
        st.markdown("#### O que foi entregue")
        entregues = [
            ("Modelo QP",            "Mínimo tracking error, solver open-source (OSQP), sem licença"),
            ("Backtest rigoroso",    "5 janelas trimestrais 100% fora da amostra"),
            ("Análise de trade-off", "11 tamanhos de carteira (K=5 a 54) testados"),
            ("ML de rebalanceamento","Random Forest reduz TE em ~18% com intervenções mínimas"),
            ("Benchmark comparativo","Elastic Net superado em todos os critérios pelo modelo QP"),
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
        st.markdown("#### Próximos passos sugeridos")
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
        f"FINOR — Index Tracking do IBOV</div>"
        f"<div style='font-size:0.82rem; color:{CINZA}; margin-top:6px'>"
        f"Dados: B3 / Yahoo Finance · 2018–2025 · Python · cvxpy · scikit-learn · Streamlit · Plotly"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Referências")
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


# ══════════════════════════════════════════════════════════════════════════════
# PLATAFORMA — experiência reativa em abas
# ══════════════════════════════════════════════════════════════════════════════
if area == "Plataforma":
    st.title("Plataforma de Index Tracking")
    st.caption(
        "Ajuste os parâmetros à esquerda e veja os resultados mudarem ao vivo, na "
        "mesma tela — sem precisar clicar em \"rodar\" ou trocar de página."
    )

    tab_sim, tab_comp, tab_proj, tab_monit = st.tabs([
        "Simulador", "Comparar Cenários", "Projeções Futuras", "Monitoramento",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — SIMULADOR (Configurador + Resultados mesclados, 100% reativo)
    # ══════════════════════════════════════════════════════════════════════
    with tab_sim:
        ctrl, resu = st.columns([1, 2], gap="large")

        with ctrl:
            st.markdown("##### Parâmetros")

            modo_universo = st.segmented_control(
                "Universo de ações", ["Todas as 54", "Selecionar"],
                default="Todas as 54", key="sim_modo_universo",
                help="Define se a otimização considera todas as 54 ações candidatas "
                     "ou apenas um subconjunto escolhido por você.",
            ) or "Todas as 54"
            if modo_universo == "Selecionar":
                sel_curto = st.multiselect(
                    "Ações candidatas:",
                    options=[a.replace(".SA", "") for a in acoes],
                    default=[a.replace(".SA", "") for a in acoes[:30]],
                    key="sim_universo_manual",
                    help="Apenas as ações marcadas aqui entram no ranking e na "
                         "otimização da carteira.",
                )
                universo_sel = [a + ".SA" for a in sel_curto]
            else:
                universo_sel = list(acoes)
            st.caption(f"{len(universo_sel)} ações no universo.")

            n_max_k = max(5, len(universo_sel))
            K_sel = st.slider(
                "Número de ações (K)", 5, n_max_k,
                value=min(int(st.session_state.get("sim_K", 30)), n_max_k),
                key="sim_K",
                help="Quantas ações a carteira final terá. Menos ações = mais "
                     "simples e barato de operar; mais ações = tende a reduzir "
                     "o Tracking Error.",
            )

            perfil_risco = st.segmented_control(
                "Perfil de risco", ["Conservador", "Moderado", "Agressivo"],
                default="Moderado", key="sim_perfil",
                help="Controla a concentração máxima permitida por ativo na carteira: "
                     "Conservador = até 15%, Moderado = até 25%, Agressivo = até 40%.",
            ) or "Moderado"
            cap_sugerido = {"Conservador": 0.15, "Moderado": 0.25, "Agressivo": 0.40}[perfil_risco]
            st.caption(f"Concentração máxima por ativo: {cap_sugerido:.0%}")

            data_min, data_max = dados.index.min().date(), dados.index.max().date()
            periodo = st.date_input(
                "Período de teste (fora da amostra)",
                value=(pd.Timestamp("2024-11-01").date(), data_max),
                min_value=data_min, max_value=data_max, key="sim_periodo",
                help="Intervalo em que a carteira é avaliada fora da amostra. O "
                     "treino usa sempre dados anteriores a esta data inicial — "
                     "nunca há vazamento de informação do futuro.",
            )

            with st.popover("Configurações avançadas", use_container_width=True):
                janela_anos = st.selectbox(
                    "Janela de treino (anos)", [1, 3, 5, 7], index=2, key="sim_janela",
                    help="Quantos anos de histórico, imediatamente antes do período "
                         "de teste, são usados para treinar (otimizar) a carteira.",
                )
                freq_reb = st.segmented_control(
                    "Frequência de rebalanceamento",
                    ["Mensal", "Trimestral", "Guiado por ML"],
                    default="Trimestral", key="sim_freq",
                    help="\"Guiado por ML\" ativa o sinal de Random Forest, que decide "
                         "dia a dia se vale a pena rebalancear a carteira.",
                ) or "Trimestral"
                st.caption(
                    "Objetivo de otimização ativo: **Mínimo Tracking Error**. "
                    "Demais objetivos (retorno ajustado, mínimo turnover) entram na próxima fase."
                )

            periodo_valido = isinstance(periodo, tuple) and len(periodo) == 2
            universo_valido = len(universo_sel) >= 5

        with resu:
            if not universo_valido:
                st.warning("Selecione ao menos 5 ações no universo para simular.")
            elif not periodo_valido:
                st.info("Selecione um período de teste completo (data inicial e final).")
            else:
                teste_ini, teste_fim = periodo
                treino_fim = pd.Timestamp(teste_ini) - pd.Timedelta(days=1)
                treino_ini = max(treino_fim - pd.DateOffset(years=int(janela_anos)), dados.index.min())

                if treino_fim <= treino_ini:
                    st.error("Não há dados suficientes antes do período escolhido para treinar.")
                else:
                    with st.spinner("Otimizando carteira..."):
                        resultado = rodar_simulacao(
                            universo_sel, K_sel, cap_sugerido,
                            str(treino_ini.date()), str(treino_fim.date()),
                            str(teste_ini), str(teste_fim),
                        )

                    if resultado.get("teste_vazio"):
                        st.error("O período de teste escolhido não contém dias de pregão. Ajuste as datas.")
                    else:
                        resultado = dict(resultado)
                        resultado["K"] = K_sel
                        resultado["params"] = {
                            "universo_n": len(universo_sel), "cap": cap_sugerido,
                            "janela_anos": janela_anos, "freq_reb": freq_reb,
                            "treino": f"{treino_ini.date()} a {treino_fim.date()}",
                            "teste": f"{teste_ini} a {teste_fim}",
                        }

                        if freq_reb == "Guiado por ML":
                            with st.spinner("Treinando sinal de ML..."):
                                resultado["ml_sinal"] = gerar_sinal_ml(
                                    resultado["pesos"], str(teste_ini), str(teste_fim),
                                )
                        else:
                            resultado["ml_sinal"] = None

                        assinatura = (K_sel, len(universo_sel), round(cap_sugerido, 3),
                                      str(treino_ini.date()), str(teste_ini), str(teste_fim), freq_reb)
                        if st.session_state.get("sim_ultima_assinatura") != assinatura:
                            st.session_state["sim_ultima_assinatura"] = assinatura
                            hist = st.session_state.get("historico_simulacoes", [])
                            hist.append(resultado)
                            st.session_state["historico_simulacoes"] = hist[-10:]
                            registrar_simulacao(resultado)
                        st.session_state["ultima_simulacao"] = resultado

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Tracking Error", f"{resultado['tracking_error']*100:.4f}%",
                                   help="Desvio padrão da diferença diária entre o retorno da "
                                        "carteira e o retorno do IBOV no período de teste. "
                                        "Quanto menor, mais fiel a réplica.")
                        c2.metric("Correlação c/ IBOV", f"{resultado['correlacao']:.2%}",
                                   help="Correlação entre os retornos diários da carteira e do "
                                        "IBOV no período de teste.")
                        c3.metric("Retorno da carteira", f"{resultado['retorno_carteira_acum']:.2%}",
                                   help="Retorno acumulado da carteira simulada durante todo o "
                                        "período de teste escolhido.")
                        c4.metric("Retorno do IBOV", f"{resultado['retorno_ibov_acum']:.2%}",
                                   help="Retorno acumulado do IBOV no mesmo período, para comparação.")

                        pesos = resultado["pesos"]
                        pk = pesos.copy()
                        pk.index = pk.index.str.replace(".SA", "")

                        cg1, cg2 = st.columns(2)
                        with cg1:
                            titulo_ajuda(
                                "Composição da carteira",
                                "Peso de cada ação na carteira otimizada para o K e o perfil "
                                "de risco escolhidos ao lado.",
                                nivel="",
                            )
                            fig = go.Figure(go.Bar(
                                x=pk.index, y=pk.values * 100,
                                marker_color=[OURO if v > pk.mean() else VERDE for v in pk.values],
                                text=[f"{v:.1f}%" for v in pk.values * 100], textposition="outside",
                            ))
                            fig.update_layout(**PLOTLY_BASE, height=320, xaxis_title="", yaxis_title="Peso (%)",
                                               title="Composição da carteira", showlegend=False)
                            st.plotly_chart(fig, use_container_width=True, key="sim_fig_pesos")
                        with cg2:
                            titulo_ajuda(
                                "Retorno acumulado",
                                "Crescimento de R$1 investido na carteira simulada e no IBOV "
                                "durante o período de teste. Triângulos marcam dias em que o "
                                "sinal de ML indicou rebalanceamento.",
                                nivel="",
                            )
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=resultado["serie_carteira"].index, y=resultado["serie_carteira"].values,
                                line=dict(color=VERDE, width=2.5), name="Carteira",
                            ))
                            fig2.add_trace(go.Scatter(
                                x=resultado["serie_ibov"].index, y=resultado["serie_ibov"].values,
                                line=dict(color=BRANCO, width=1.5, dash="dot"), name="IBOV",
                            ))
                            ml_sinal = resultado.get("ml_sinal")
                            if ml_sinal:
                                serie_c = resultado["serie_carteira"]
                                datas_sinal = [s["data"] for s in ml_sinal["sinais"]
                                               if s["sinal"] == 1 and s["data"] in serie_c.index]
                                if datas_sinal:
                                    fig2.add_trace(go.Scatter(
                                        x=datas_sinal, y=[serie_c.loc[d] for d in datas_sinal],
                                        mode="markers",
                                        marker=dict(color=OURO, size=11, symbol="triangle-up"),
                                        name="Sinal ML",
                                    ))
                            _base_sem_legend = {k: v for k, v in PLOTLY_BASE.items() if k != "legend"}
                            fig2.update_layout(
                                **_base_sem_legend, height=320, title="Retorno acumulado",
                                yaxis_title="Base 1.0",
                                legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.5, xanchor="center"),
                            )
                            st.plotly_chart(fig2, use_container_width=True, key="sim_fig_retorno")

                        if resultado.get("ml_sinal") is not None:
                            n_sinais = sum(1 for s in resultado["ml_sinal"]["sinais"] if s["sinal"] == 1)
                            st.caption(
                                f"{n_sinais} sinal(is) de rebalanceamento detectado(s) no período "
                                f"(Random Forest treinado com {resultado['ml_sinal']['n_treino']} amostras)."
                            )
                        elif freq_reb == "Guiado por ML":
                            st.caption("Histórico insuficiente antes do período de teste para treinar o sinal de ML.")

                        with st.expander("Custo de ajustar a carteira e exportar resultados"):
                            st.markdown(
                                "Toda vez que você muda os parâmetros e a carteira é reotimizada, "
                                "algumas ações entram e outras saem — essa movimentação gera custos de "
                                "corretagem. Esta seção estima quanto custaria executar esse ajuste e "
                                "permite exportar os resultados da simulação."
                            )
                            hist = st.session_state.get("historico_simulacoes", [])
                            turnover = None
                            if len(hist) >= 2:
                                pesos_ant = hist[-2]["pesos"]
                                todos = set(pesos.index) | set(pesos_ant.index)
                                turnover = 0.5 * sum(
                                    abs(float(pesos.get(t, 0.0)) - float(pesos_ant.get(t, 0.0))) for t in todos
                                )
                            ce1, ce2, ce3 = st.columns(3)
                            custo_bps = ce1.slider(
                                "Custo por operação (bps)", 1, 50, 10, key="sim_custo_bps",
                                help="Um bps (ponto-base) equivale a 0,01% do valor operado. "
                                     "Corretoras cobram tipicamente entre 5 e 30 bps. "
                                     "Arraste para simular diferentes patamares de custo.",
                            )
                            if turnover is not None:
                                ce2.metric(
                                    "Troca de composição",
                                    f"{turnover:.1%}",
                                    help="Percentual da carteira que mudou entre a configuração anterior e a atual. "
                                         "100% = trocou tudo; 10% = ajuste pequeno.",
                                )
                                ce3.metric(
                                    "Custo total estimado",
                                    f"{turnover*custo_bps/100:.3f}%",
                                    help="Troca de composição multiplicada pelo custo por operação. "
                                         "Representa o impacto da corretagem ao ajustar a carteira.",
                                )
                            else:
                                ce2.info("Disponível a partir do segundo ajuste de parâmetros nesta sessão.")

                            st.markdown("---")
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                                pesos.to_frame("peso").to_excel(writer, sheet_name="Carteira")
                                pd.DataFrame([{
                                    "K": resultado["K"], "Tracking Error": resultado["tracking_error"],
                                    "Correlacao": resultado["correlacao"],
                                    "Retorno_Carteira": resultado["retorno_carteira_acum"],
                                    "Retorno_IBOV": resultado["retorno_ibov_acum"],
                                    **resultado["params"],
                                }]).to_excel(writer, sheet_name="Resumo", index=False)
                            cdl1, cdl2 = st.columns(2)
                            cdl1.download_button(
                                "Exportar Excel", data=buffer.getvalue(),
                                file_name="simulacao_index_tracking.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True, key="sim_dl_xlsx",
                            )
                            cdl2.download_button(
                                "Exportar PDF", data=gerar_pdf_relatorio(resultado),
                                file_name="relatorio_index_tracking.pdf",
                                mime="application/pdf",
                                use_container_width=True, key="sim_dl_pdf",
                            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — COMPARAR CENÁRIOS
    # ══════════════════════════════════════════════════════════════════════
    with tab_comp:
        st.markdown("##### Rodar um novo lote de comparação")
        st.caption("Usa o mesmo universo, perfil de risco e período configurados no Simulador.")

        resultado_atual = st.session_state.get("ultima_simulacao")
        if resultado_atual is None:
            st.info("Ajuste parâmetros na aba **Simulador** primeiro — o lote reaproveita esse contexto.")
        else:
            p = resultado_atual["params"]
            universo_lote = list(acoes) if p["universo_n"] == len(acoes) else None
            k_max_disp = min(54, len(acoes))
            k_range = st.slider(
                "Intervalo de K a testar",
                min_value=5, max_value=k_max_disp,
                value=(10, min(30, k_max_disp)),
                step=5, key="comp_k_range",
                help="Arraste as duas extremidades para definir o menor e o maior número "
                     "de ações que você quer comparar. A comparação roda um cenário para "
                     "cada múltiplo de 5 dentro do intervalo escolhido.",
            )
            Ks_comparar = sorted(set(
                [k_range[0], k_range[1]]
                + [k for k in range(10, k_range[1] + 1, 5) if k_range[0] <= k <= k_range[1]]
            ))
            Ks_comparar = [k for k in Ks_comparar if k <= len(acoes)]
            if Ks_comparar:
                st.caption(f"Valores de K que serao testados: {', '.join(str(k) for k in Ks_comparar)}")
            if st.button("Rodar comparação", type="primary", disabled=not Ks_comparar):
                treino_ini_str, treino_fim_str = p["treino"].split(" a ")
                teste_ini_str, teste_fim_str = p["teste"].split(" a ")
                universo_lote = universo_lote or list(acoes)
                lote_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                resultados_lote, barra = [], st.progress(0.0)
                for i, K in enumerate(sorted(Ks_comparar)):
                    r = rodar_simulacao(universo_lote, K, p["cap"], treino_ini_str, treino_fim_str,
                                         teste_ini_str, teste_fim_str)
                    if not r.get("teste_vazio"):
                        r = dict(r)
                        r["K"] = K
                        r["params"] = {**p}
                        registrar_simulacao(r, lote_id=lote_id)
                        resultados_lote.append(r)
                    barra.progress((i + 1) / len(Ks_comparar))
                barra.empty()
                if resultados_lote:
                    st.success(f"{len(resultados_lote)} cenários rodados (lote {lote_id}) — veja a comparação abaixo.")
                else:
                    st.error("Nenhum cenário pôde ser avaliado no período escolhido.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("##### Comparar simulações salvas")
        st.markdown(
            "Cada vez que você roda uma simulação no **Simulador**, ela fica salva automaticamente. "
            "Aqui você pode colocar duas ou mais simulações lado a lado para ver qual configuração "
            "entregou o melhor resultado — qual K rastreou o IBOV com mais precisão, qual teve "
            "menor erro, qual gerou mais retorno."
        )

        historico = carregar_historico_persistente()
        if not historico:
            st.info("Nenhuma simulacao registrada ainda. Rode ao menos duas simulacoes no Simulador para comparar.")
        else:
            opcoes = {}
            for h in reversed(historico):
                rotulo = (
                    f"K={h['K']} · Erro={h['tracking_error']*100:.4f}% · "
                    f"Retorno={h['retorno_carteira_acum']*100:.1f}% · "
                    f"{h['timestamp'][:16].replace('T',' ')}"
                )
                opcoes[rotulo] = h

            selecionados = st.multiselect(
                "Escolha as simulacoes que quer comparar (minimo 2):",
                options=list(opcoes.keys()), default=list(opcoes.keys())[:min(3, len(opcoes))],
                max_selections=6, key="comp_selecionados",
                help="Cada linha representa uma simulacao ja realizada. O 'Erro' e o Tracking Error — "
                     "quanto menor, melhor a replica do IBOV. 'Retorno' e o resultado acumulado da carteira.",
            )
            if len(selecionados) < 2:
                st.warning("Selecione ao menos 2 simulações para comparar.")
            else:
                entradas = [opcoes[s] for s in selecionados]
                df_comp = pd.DataFrame([{
                    "Cenário": f"K={e['K']} · #{e['id'][-4:]}", "K": e["K"],
                    "Tracking Error (%)": e["tracking_error"] * 100,
                    "Correlação (%)": e["correlacao"] * 100,
                    "Retorno Carteira (%)": e["retorno_carteira_acum"] * 100,
                    "Retorno IBOV (%)": e["retorno_ibov_acum"] * 100,
                    "Treino": e["params"].get("treino", ""), "Teste": e["params"].get("teste", ""),
                } for e in entradas])

                col1, col2 = st.columns(2)
                _base_comp = {k: v for k, v in PLOTLY_BASE.items() if k != "yaxis"}
                with col1:
                    titulo_ajuda("Tracking Error por cenário",
                                  "Quanto menor a barra, melhor: significa que a carteira seguiu "
                                  "o IBOV com mais precisao. A barra destacada tem o menor erro.", nivel="")
                    fig = go.Figure(go.Bar(
                        x=df_comp["Cenário"], y=df_comp["Tracking Error (%)"],
                        marker_color=[OURO if v == df_comp["Tracking Error (%)"].min() else VERDE
                                      for v in df_comp["Tracking Error (%)"]],
                        text=[f"{v:.4f}%" for v in df_comp["Tracking Error (%)"]], textposition="outside",
                    ))
                    _te_max = df_comp["Tracking Error (%)"].max()
                    fig.update_layout(
                        **_base_comp, height=360, title="Tracking Error por cenario",
                        showlegend=False, xaxis_title="", yaxis_title="Tracking Error (%)",
                        yaxis=dict(**PLOTLY_BASE["yaxis"], range=[0, _te_max * 1.30]),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="comp_fig_te")
                with col2:
                    titulo_ajuda("Correlacao com IBOV por cenario",
                                  "Quanto maior a barra, melhor: significa que a carteira andou "
                                  "junto com o IBOV. A barra destacada tem a maior correlacao.", nivel="")
                    fig2 = go.Figure(go.Bar(
                        x=df_comp["Cenário"], y=df_comp["Correlação (%)"],
                        marker_color=[OURO if v == df_comp["Correlação (%)"].max() else AZUL
                                      for v in df_comp["Correlação (%)"]],
                        text=[f"{v:.2f}%" for v in df_comp["Correlação (%)"]], textposition="outside",
                    ))
                    _corr_max = df_comp["Correlação (%)"].max()
                    _corr_min = df_comp["Correlação (%)"].min()
                    fig2.update_layout(
                        **_base_comp, height=360, title="Correlacao com IBOV por cenario",
                        showlegend=False, xaxis_title="", yaxis_title="Correlacao (%)",
                        yaxis=dict(**PLOTLY_BASE["yaxis"],
                                   range=[max(0, _corr_min - (_corr_max - _corr_min) * 0.5),
                                          min(105, _corr_max * 1.05)]),
                    )
                    st.plotly_chart(fig2, use_container_width=True, key="comp_fig_corr")

                df_show = df_comp.drop(columns=["K"]).copy()
                for c in ["Tracking Error (%)", "Correlação (%)", "Retorno Carteira (%)", "Retorno IBOV (%)"]:
                    df_show[c] = df_show[c].map("{:.4f}".format)
                st.dataframe(df_show, use_container_width=True, hide_index=True)

                melhor = df_comp.loc[df_comp["Tracking Error (%)"].idxmin()]
                st.info(
                    f"Melhor resultado: **{melhor['Cenário']}** — Tracking Error de {melhor['Tracking Error (%)']:.4f}%, "
                    f"correlacao de {melhor['Correlação (%)']:.2f}% com o IBOV."
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — PROJEÇÕES FUTURAS (MONTE CARLO)
    # ══════════════════════════════════════════════════════════════════════
    with tab_proj:
        st.markdown(
            "Aqui você visualiza como a sua carteira **pode se comportar nos próximos meses**, "
            "com base no que o mercado já fez no passado. Nao é uma previsao do futuro — "
            "é uma análise de 'e se o futuro se parecer com o passado?', mostrando o cenário "
            "mais provável e os extremos possíveis."
        )
        resultado = st.session_state.get("ultima_simulacao")
        if resultado is None:
            st.info("Configure e rode uma simulacao na aba **Simulador** primeiro — a projecao parte da carteira otimizada.")
        else:
            pesos = resultado["pesos"]
            c1, c2, c3 = st.columns(3)
            with c1:
                horizonte = st.select_slider(
                    "Por quanto tempo projetar?",
                    options=[20, 40, 60, 90, 120],
                    value=60, key="proj_horizonte",
                    help="Quantos dias de mercado (dias uteis) voce quer projetar. "
                         "1 mes = ~20 dias · 3 meses = ~60 dias · 6 meses = ~120 dias. "
                         "Horizontes mais longos trazem mais incerteza — as faixas ficam mais largas.",
                    format_func=lambda x: {20: "1 mes (~20d)", 40: "2 meses (~40d)",
                                           60: "3 meses (~60d)", 90: "4,5 meses (~90d)",
                                           120: "6 meses (~120d)"}[x],
                )
            with c2:
                n_sim = st.slider(
                    "Quantos cenarios gerar?", 200, 2000, 500, step=100, key="proj_nsim",
                    help="Cada cenario e um 'futuro alternativo' construido com fragmentos do historico real. "
                         "Mais cenarios = resultado mais preciso e confiavel, mas demora um pouco mais para calcular. "
                         "500 ja e suficiente para a maioria dos casos.",
                )
            with c3:
                bloco = st.slider(
                    "Tamanho dos blocos de historico (dias)", 5, 40, 20, key="proj_bloco",
                    help="O modelo monta cada cenario futuro colando pedacos do historico real. "
                         "Este valor define o tamanho desses pedacos. "
                         "Valor menor = mais variacao entre cenarios; "
                         "valor maior = preserva melhor o ritmo e a volatilidade do mercado. "
                         "Recomendado: entre 15 e 25 dias.",
                )

            st.caption(
                f"Carteira ativa: {resultado['K']} acoes · "
                f"Precisao historica (Tracking Error) = {resultado['tracking_error']*100:.4f}%"
            )

            if st.button("Gerar projecao", type="primary", use_container_width=True, key="proj_rodar"):
                with st.spinner(f"Gerando {n_sim} cenarios para os proximos {horizonte} dias uteis..."):
                    st.session_state["ultima_projecao"] = montecarlo_projecao(pesos, horizonte, n_sim, bloco)

            proj = st.session_state.get("ultima_projecao")
            if proj:
                titulo_ajuda(
                    "Diferenca projetada entre a carteira e o IBOV",
                    "Mostra o quanto a carteira pode se afastar do IBOV ao longo do tempo. "
                    "Linha central = cenario mais provavel. "
                    "Faixa interna (mais escura) = onde ficam 50% dos cenarios. "
                    "Faixa externa = onde ficam 90% dos cenarios. "
                    "Acima de zero = carteira superou o IBOV; abaixo = IBOV foi melhor.",
                    nivel="",
                )
                dias_x = list(range(1, proj["horizonte"] + 1))
                pdiff = proj["percentis_diff"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dias_x + dias_x[::-1], y=list(pdiff[95]) + list(pdiff[5])[::-1],
                                          fill="toself", fillcolor=VERDE_A10, line=dict(width=0),
                                          name="Faixa 90% dos cenarios"))
                fig.add_trace(go.Scatter(x=dias_x + dias_x[::-1], y=list(pdiff[75]) + list(pdiff[25])[::-1],
                                          fill="toself", fillcolor=VERDE_A20, line=dict(width=0),
                                          name="Faixa 50% dos cenarios"))
                fig.add_trace(go.Scatter(x=dias_x, y=pdiff[50], line=dict(color=VERDE, width=2.5),
                                          name="Cenario mais provavel"))
                fig.add_hline(y=0, line_dash="dot", line_color=BRANCO, opacity=0.4)
                _meses = horizonte // 20
                fig.update_layout(
                    **PLOTLY_BASE, height=420,
                    title=f"Diferenca projetada (carteira vs IBOV) — {proj['n_sim']} cenarios, {_meses} mes(es)",
                    xaxis_title="Dias uteis a frente", yaxis_title="Diferenca acumulada",
                    yaxis_tickformat=".1%",
                )
                st.plotly_chart(fig, use_container_width=True, key="proj_fig")

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Resultado mais provavel ao final",
                    f"{pdiff[50][-1]:+.2%}",
                    help="Na metade dos cenarios simulados, a carteira termina o periodo com essa diferenca "
                         "em relacao ao IBOV. Valor positivo = carteira na frente; negativo = IBOV na frente.",
                )
                c2.metric(
                    "Intervalo de 90% dos cenarios",
                    f"[{pdiff[5][-1]:+.2%} , {pdiff[95][-1]:+.2%}]",
                    help="Em 90% dos cenarios simulados, a diferenca final ficou dentro deste intervalo. "
                         "Quanto mais estreito, mais previsivel e o comportamento da carteira.",
                )
                c3.metric(
                    "Risco de desvio relevante",
                    f"{proj['prob_divergencia']:.1%}",
                    help="Probabilidade de a carteira se afastar mais do que o dobro do seu erro historico. "
                         "Acima de 25% e sinal de atencao — vale revisar o numero de acoes (K) ou "
                         "a frequencia de ajuste da carteira.",
                )

                if proj["prob_divergencia"] > 0.25:
                    st.warning(
                        "Ha uma chance relevante de desvio significativo neste horizonte. "
                        "Considere aumentar K ou usar rebalanceamento mais frequente no Simulador."
                    )
                else:
                    st.success(
                        "Risco de desvio dentro do esperado para este horizonte — "
                        "a carteira tende a acompanhar o IBOV com boa consistencia."
                    )

                with st.expander("Como interpretar este grafico?"):
                    st.markdown(
                        "**Linha central (cor laranja):** o cenario mais provavel — o meio da distribuicao "
                        "de todos os futuros simulados.\n\n"
                        "**Faixa interna:** metade dos cenarios caiu dentro dessa faixa. "
                        "Quanto mais estreita, mais previsivel a carteira.\n\n"
                        "**Faixa externa:** 90% dos cenarios ficaram aqui. "
                        "Os extremos do grafico representam situacoes raras mas possiveis.\n\n"
                        "**Linha pontilhada em zero:** linha de referencia. "
                        "Acima = carteira superou o IBOV; abaixo = IBOV foi melhor.\n\n"
                        "**Faixas mais largas com o tempo** sao normais — incerteza cresce quanto mais longe "
                        "voce projeta. Isso nao e um defeito do modelo, e a realidade do mercado."
                    )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — MONITORAMENTO E REOTIMIZAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    with tab_monit:
        st.markdown(
            "**O que e o Monitoramento?** Depois que uma carteira e montada, ela precisa ser acompanhada "
            "ao longo do tempo — o mercado muda, e uma carteira que funcionava bem pode comecara desvia "
            "do IBOV. Esta aba detecta automaticamente sinais de que isso esta acontecendo e avisa quando "
            "vale a pena revisar a composicao."
        )
        st.markdown(
            "**Usa dados em tempo real?** Nao — esta versao usa uma base historica fixa (ate abril de 2025). "
            "Em producao real, os dados seriam atualizados diariamente. O que voce ve aqui e a mesma logica "
            "que seria aplicada com dados ao vivo: regime de mercado, anomalias e drift da carteira."
        )
        st.markdown("**Como usar:** rode uma simulacao no **Simulador** e volte aqui para ver o diagnostico.")
        st.markdown("---")
        resultado = st.session_state.get("ultima_simulacao")
        if resultado is None:
            st.info("Ajuste parametros na aba **Simulador** primeiro.")
        else:
            pesos = resultado["pesos"]
            with st.spinner("Analisando regime e anomalias..."):
                diag = detectar_regime_anomalias(pesos)

            cor_regime = {"Calmo": VERDE, "Normal": AZUL, "Volátil": "#ff4b4b", "Indefinido": CINZA}
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"<div class='card' style='text-align:center'>"
                f"<div style='font-size:1.6rem; font-weight:800; color:{cor_regime.get(diag['regime_atual'], CINZA)}'>"
                f"{diag['regime_atual']}</div>"
                f"<div style='font-size:0.75rem; color:{CINZA}; margin-top:4px'>Regime atual do IBOV</div></div>",
                unsafe_allow_html=True,
            )
            c2.metric(
                "Volatilidade atual do mercado",
                f"{diag['vol_atual']:.1f}%" if diag['vol_atual'] is not None else "N/D",
                help="Volatilidade anualizada do IBOV nos ultimos 20 dias. "
                     "Alta volatilidade significa que o mercado esta oscilando muito — "
                     "nesses momentos, e mais dificil a carteira acompanhar o IBOV com precisao.",
            )
            c3.metric(
                "Dias com comportamento anomalo",
                len(diag["anomalias"]),
                help="Numero de dias em que o erro da carteira em relacao ao IBOV foi "
                     "estatisticamente fora do padrao — muito acima da media historica. "
                     "Alguns dias anomalos sao normais; muitos seguidos podem indicar necessidade de ajuste.",
            )

            titulo_ajuda(
                "Nivel de oscilacao do mercado ao longo do tempo",
                "Mostra o quanto o IBOV oscilou em cada periodo. "
                "A linha pontilhada e a oscilacao media historica — acima dela = mercado agitado, "
                "abaixo = mercado calmo. O regime atual e classificado com base nessa comparacao.",
                nivel="",
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=diag["vol_ibov_20d"].index, y=diag["vol_ibov_20d"].values,
                                      line=dict(color=AZUL, width=1.3), name="Vol. IBOV (20d)"))
            fig.add_hline(y=diag["vol_mediana"], line_dash="dot", line_color=BRANCO, opacity=0.5,
                          annotation_text="mediana histórica")
            fig.update_layout(**PLOTLY_BASE, height=300, title="Volatilidade do IBOV ao longo do tempo",
                               yaxis_title="Vol. anualizada (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="monit_fig_vol")

            titulo_ajuda(
                "Erro diario da carteira e dias anomalos",
                "Mostra o quanto a carteira se desviou do IBOV a cada dia. "
                "Os marcadores em X sao os dias em que esse desvio foi excepcionalmente alto — "
                "fora do padrao estatistico normal. Alguns sao esperados; muitos concentrados "
                "num curto periodo podem sinalizar que a carteira precisa ser reajustada.",
                nivel="",
            )
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=diag["te_abs"].index, y=diag["te_abs"].values * 100,
                                       line=dict(color=VERDE, width=1.1), name="TE diário absoluto"))
            if len(diag["anomalias"]) > 0:
                fig2.add_trace(go.Scatter(
                    x=diag["anomalias"].index, y=diag["te_abs"].loc[diag["anomalias"].index].values * 100,
                    mode="markers", marker=dict(color="#ff4b4b", size=9, symbol="x"), name="Anomalia",
                ))
            _base_sem_legend = {k: v for k, v in PLOTLY_BASE.items() if k != "legend"}
            fig2.update_layout(**_base_sem_legend, height=300, title="Tracking error diário e anomalias (|z| > 3)",
                                yaxis_title="TE absoluto (%)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"))
            st.plotly_chart(fig2, use_container_width=True, key="monit_fig_te")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("##### A carteira ainda funciona bem?")
            st.markdown(
                "Abaixo comparamos o erro da carteira no periodo de teste original com o erro "
                "observado nos dados mais recentes disponíveis. Se o erro piorou muito, "
                "e hora de reotimizar."
            )

            teste_fim_str = resultado["params"]["teste"].split(" a ")[-1]
            try:
                teste_fim_ts = pd.Timestamp(teste_fim_str)
                pos_data = dados.loc[dados.index > teste_fim_ts]
            except Exception:
                pos_data = pd.DataFrame()

            if len(pos_data) < 10:
                st.info(
                    "Não há dados suficientes após o período de teste desta simulação para medir "
                    "o drift real (este protótipo usa uma base histórica fixa, até 2025-04-30). "
                    "Em produção, esta seção compararia o desempenho real dos dias mais recentes "
                    "contra o Tracking Error esperado, dia após dia."
                )
            else:
                try:
                    ret_c = pos_data[pesos.index].values @ pesos.values
                    ret_i = pos_data["IBOV"].values
                    te_atual = float(np.std(ret_c - ret_i))
                    drift_pct = (te_atual - resultado["tracking_error"]) / resultado["tracking_error"] * 100

                    cd1, cd2, cd3 = st.columns(3)
                    cd1.metric(
                        "Erro original (no teste)",
                        f"{resultado['tracking_error']*100:.4f}%",
                        help="Tracking Error medido durante o backtest — a precisao da carteira "
                             "no periodo em que foi avaliada inicialmente.",
                    )
                    cd2.metric(
                        "Erro atual (dados recentes)",
                        f"{te_atual*100:.4f}%",
                        help="Tracking Error observado nos dados disponíveis apos o periodo de teste. "
                             "Se estiver muito maior que o original, a carteira perdeu qualidade.",
                    )
                    cd3.metric(
                        "Variacao do erro",
                        f"{drift_pct:+.1f}%",
                        help="Quanto o erro atual e maior ou menor que o original. "
                             "Ate +25% e aceitavel — acima disso, e recomendavel reotimizar a carteira.",
                    )

                    if drift_pct > 25:
                        st.warning(
                            "O erro da carteira aumentou mais de 25% em relacao ao esperado — "
                            "isso indica que a composicao atual perdeu precisao. "
                            "Clique abaixo para reotimizar com os dados mais recentes disponíveis."
                        )
                        if st.button("Reotimizar a carteira agora", type="primary", key="monit_reotim"):
                            with st.spinner("Reotimizando..."):
                                novo_treino_fim = dados.index.max()
                                novo_treino_ini = max(
                                    novo_treino_fim - pd.DateOffset(years=int(resultado["params"]["janela_anos"])),
                                    dados.index.min(),
                                )
                                pos_corte = max(0, len(dados.loc[:novo_treino_fim]) - 20)
                                checagem_ini = dados.loc[:novo_treino_fim].index[pos_corte]
                                K_novo = recomendar_k(
                                    pesos.index.tolist(), str(novo_treino_ini.date()), str(novo_treino_fim.date()),
                                    resultado["params"]["cap"],
                                )
                                resultado_novo = rodar_simulacao(
                                    pesos.index.tolist(), K_novo, resultado["params"]["cap"],
                                    str(novo_treino_ini.date()), str(novo_treino_fim.date()),
                                    str(checagem_ini.date()), str(novo_treino_fim.date()),
                                )
                                resultado_novo = dict(resultado_novo)
                                resultado_novo["K"] = K_novo
                                resultado_novo["params"] = {
                                    **resultado["params"],
                                    "treino": f"{novo_treino_ini.date()} a {novo_treino_fim.date()}",
                                    "teste": f"snapshot in-sample {checagem_ini.date()} a {novo_treino_fim.date()} (nao e backtest OOS)",
                                }
                                resultado_novo["ml_sinal"] = None
                                registrar_simulacao(resultado_novo)
                                st.session_state["ultima_simulacao"] = resultado_novo
                                st.success(
                                    f"Carteira reotimizada com dados até {novo_treino_fim.date()} — "
                                    f"novo K={K_novo}. As métricas refletem um snapshot in-sample, não "
                                    f"um novo backtest fora da amostra. Veja na aba **Simulador**."
                                )
                    else:
                        st.success(
                            "O erro da carteira esta dentro do esperado — nenhuma acao necessaria por enquanto."
                        )
                except KeyError:
                    st.info("Algumas ações da carteira não têm dados no período mais recente para medir o drift.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                "**Sobre automação em produção:** esta página demonstra a lógica de monitoramento e "
                "reotimização — em produção real, essa verificação seria executada automaticamente "
                "todos os dias por um job agendado (cron, Airflow, ou equivalente), com alertas via "
                "e-mail/Slack quando o drift superasse o limite. Nesta versão local, o botão acima "
                "reproduz manualmente o mesmo efeito."
            )
