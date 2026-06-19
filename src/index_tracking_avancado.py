import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CARREGAR DADOS (mesmo pipeline do modelo principal)
# ============================================================
BASE = "data/processed"
retornos = pd.read_csv(f"{BASE}/ibov_acoes_selecionadas.csv", index_col=0, parse_dates=True)
indice   = pd.read_csv(f"{BASE}/ibov_indice_retornos.csv",   index_col=0, parse_dates=True)

dados = retornos.join(indice["Variação_Diária_%"], how="inner").dropna()
dados = dados.rename(columns={"Variação_Diária_%": "IBOV"})
dados["IBOV"] = dados["IBOV"] / 100.0
acoes = [c for c in dados.columns if c != "IBOV"]

treino = dados.loc[:"2024-01-31"]
janelas_teste = [
    ("2024-02-01", "2024-04-30"),
    ("2024-05-01", "2024-07-31"),
    ("2024-08-01", "2024-10-31"),
    ("2024-11-01", "2025-01-31"),
    ("2025-02-01", "2025-04-30"),
]
testes = [dados.loc[i:f] for i, f in janelas_teste]
X_train = treino[acoes].values
y_train = treino["IBOV"].values

def resolver_qp(X, y, ativos, cap=0.25):
    n = len(ativos)
    w = cp.Variable(n)
    erro = X @ w - y
    prob = cp.Problem(cp.Minimize(cp.sum_squares(erro)),
                      [cp.sum(w) == 1, w >= 0, w <= cap])
    prob.solve(solver=cp.OSQP)
    pesos = pd.Series(np.maximum(w.value, 0), index=ativos)
    return pesos / pesos.sum()

def avaliar_pesos(pesos, df):
    rc = df[pesos.index].values @ pesos.values
    ri = df["IBOV"].values
    d  = rc - ri
    return {
        "tracking_error":        float(np.std(d)),
        "correlacao":            float(np.corrcoef(rc, ri)[0,1]),
        "retorno_carteira_acum": float((1+rc).prod()-1),
        "retorno_ibov_acum":     float((1+ri).prod()-1),
    }, rc, ri

# ============================================================
# PARTE A — TRADE-OFF EXPANDIDO (K de 5 a 54)
# ============================================================
print("="*60)
print("PARTE A: TRADE-OFF EXPANDIDO DE CARDINALIDADE")
print("="*60)

# Ranking base (QP com todos os 54 ativos)
pesos_full = resolver_qp(X_train, y_train, acoes, cap=0.15)
ranking    = pesos_full.sort_values(ascending=False)

Ks = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 54]
rows = []
for K in Ks:
    top    = ranking.head(K).index.tolist()
    cap_k  = min(0.40, max(0.15, 1.5/K))
    pesos_k = resolver_qp(treino[top].values, y_train, top, cap=cap_k)

    tes, cors, rc_list = [], [], []
    for df_t in testes:
        m, rc, ri = avaliar_pesos(pesos_k, df_t)
        tes.append(m["tracking_error"])
        cors.append(m["correlacao"])
        rc_list.append(rc)

    # retorno acumulado sobre todas as 5 janelas concatenadas
    ret_total_cart  = float(np.prod([(1+rc).prod() for rc in rc_list])-1)
    ret_total_ibov  = float(np.prod([(1+df_t["IBOV"].values).prod()
                                     for df_t in testes])-1)

    rows.append({
        "K":                      K,
        "TE_medio_oos":           round(np.mean(tes)*100, 4),   # em %
        "TE_max_oos":             round(np.max(tes)*100,  4),
        "correlacao_media":       round(np.mean(cors),    4),
        "retorno_carteira_%":     round(ret_total_cart*100, 2),
        "retorno_ibov_%":         round(ret_total_ibov*100, 2),
        "custo_reducao_%TE":      None,   # preenchido abaixo
    })

df_kk = pd.DataFrame(rows)
te_ref = df_kk.loc[df_kk.K==54, "TE_medio_oos"].values[0]
df_kk["custo_reducao_%TE"] = ((df_kk["TE_medio_oos"] - te_ref) / te_ref * 100).round(1)
df_kk.to_csv(f"{BASE}/tradeoff_cardinalidade_expandido.csv", index=False)
print(df_kk.to_string(index=False))

# --- Gráfico curva de eficiência ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(df_kk["K"], df_kk["TE_medio_oos"], "o-", color="#2a6f97", linewidth=2)
for _, r in df_kk.iterrows():
    ax.annotate(f"K={int(r.K)}\n{r.TE_medio_oos:.3f}%",
                (r.K, r.TE_medio_oos),
                textcoords="offset points", xytext=(5, 6), fontsize=7.5)
ax.axvspan(14, 21, alpha=0.12, color="green",
           label="Zona de equilíbrio (15–20 ações)")
ax.set_xlabel("Número de ações na carteira (K)")
ax.set_ylabel("Tracking Error médio OOS (%)")
ax.set_title("Curva de Eficiência: ações vs erro de tracking")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax2 = axes[1]
custo = df_kk[df_kk.K < 54][["K","custo_reducao_%TE"]].copy()
bars = ax2.bar(custo["K"].astype(str), custo["custo_reducao_%TE"],
               color=["#d62728" if v > 20 else "#ff7f0e" if v > 8
                      else "#2ca02c" for v in custo["custo_reducao_%TE"]])
ax2.axhline(10, color="red", linestyle="--", linewidth=1,
            label="Limite de +10% de custo")
ax2.set_xlabel("K (número de ações)")
ax2.set_ylabel("Custo de redução (% acima do TE com 54 ações)")
ax2.set_title("Custo de simplificar a carteira")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, axis="y")

plt.suptitle("Trade-off: simplificação da carteira vs perda de precisão",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/tradeoff_expandido.png", dpi=140)
plt.close()
print("\n[OK] Gráfico de trade-off salvo.")

# ============================================================
# PARTE B — ML DE SINAL DE REBALANCEAMENTO (Random Forest)
# ============================================================
print("\n" + "="*60)
print("PARTE B: ML — SINAL DE REBALANCEAMENTO")
print("="*60)

# B1. Pesos da carteira base (K=30, igual ao modelo principal)
pesos_base = resolver_qp(
    treino[ranking.head(30).index.tolist()].values,
    y_train,
    ranking.head(30).index.tolist(),
    cap=0.15
)

# B2. Série de TE diário no período de treino
ret_cart_treino = treino[pesos_base.index].values @ pesos_base.values
ret_ibov_treino = treino["IBOV"].values
te_diario = np.abs(ret_cart_treino - ret_ibov_treino)   # erro absoluto diário

# B3. Gerar janelas deslizantes → features + label
LOOK = 60    # dias de histórico para calcular features
STEP = 20    # a cada 20 dias gera uma nova amostra
HORIZ= 20    # horizonte do label (próximos 20 dias)

X_ml, y_ml, datas_ml = [], [], []
for i in range(LOOK, len(te_diario) - HORIZ, STEP):
    janela    = te_diario[i-LOOK:i]
    proximos  = te_diario[i:i+HORIZ]
    ibov_ret  = ret_ibov_treino[i-LOOK:i]
    cart_ret  = ret_cart_treino[i-LOOK:i]

    feats = [
        np.mean(janela[-20:]),          # TE médio últimos 20d
        np.mean(janela[-60:]),          # TE médio últimos 60d
        np.std(janela[-20:]),           # Volatilidade do TE (20d)
        np.std(ibov_ret[-20:]),         # Volatilidade do IBOV (20d)
        np.mean(ibov_ret[-20:]),        # Retorno médio do IBOV (20d)
        np.corrcoef(cart_ret[-20:],
                    ibov_ret[-20:])[0,1],  # Correlação rolling (20d)
        np.sum(te_diario[max(0,i-LOOK):i] > np.median(te_diario[:i])),  # dias acima mediana
    ]
    label = 1 if np.mean(proximos) > np.median(te_diario[:i]) else 0
    X_ml.append(feats)
    y_ml.append(label)
    datas_ml.append(treino.index[i])

X_ml = np.array(X_ml)
y_ml = np.array(y_ml)
print(f"Amostras ML: {len(X_ml)} | Positivos (rebalancear): {y_ml.sum()} | Negativos: {(y_ml==0).sum()}")

# B4. Treinar Random Forest (70% treino / 30% validação within training)
split = int(len(X_ml) * 0.70)
Xtr, ytr = X_ml[:split], y_ml[:split]
Xvl, yvl = X_ml[split:], y_ml[split:]

rf = RandomForestClassifier(n_estimators=200, max_depth=4,
                             min_samples_leaf=3, random_state=42)
rf.fit(Xtr, ytr)
y_pred_val = rf.predict(Xvl)
print("\nDesempenho do Random Forest (validação within training):")
print(classification_report(yvl, y_pred_val, target_names=["Manter","Rebalancear"]))

feature_names = ["TE_med_20d","TE_med_60d","TE_vol_20d",
                 "IBOV_vol_20d","IBOV_ret_20d","Corr_20d","Dias_acima_mediana"]
importancias = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nImportância das features:")
print(importancias)

# B5. Aplicar sinal nas 5 janelas de teste → 3 estratégias
# Estratégia 1: Estática (pesos fixos desde o início)
# Estratégia 2: Sempre rebalanceia (refaz QP nos 252 dias antes de cada janela)
# Estratégia 3: ML-guiada (refaz QP só se Random Forest sinalizar "1")

resultados_estrategias = {s: [] for s in ["Estática", "Sempre-Rebalanceia", "ML-Guiada"]}
pesos_atual_ml = pesos_base.copy()

for k, (inicio, fim) in enumerate(janelas_teste):
    df_teste = dados.loc[inicio:fim]

    # --- features para o sinal ML no início desta janela ---
    data_inicio = df_teste.index[0]
    hist = dados.loc[:data_inicio].iloc[-LOOK-1:-1]  # últimos LOOK dias antes da janela
    rc_h = hist[pesos_base.index].values @ pesos_base.values
    ri_h = hist["IBOV"].values
    te_h = np.abs(rc_h - ri_h)
    te_all = np.abs(
        dados.loc[:"2024-01-31"][pesos_base.index].values @ pesos_base.values
        - dados.loc[:"2024-01-31"]["IBOV"].values
    )

    feats_live = np.array([[
        np.mean(te_h[-20:]),
        np.mean(te_h),
        np.std(te_h[-20:]),
        np.std(ri_h[-20:]),
        np.mean(ri_h[-20:]),
        np.corrcoef(rc_h[-20:], ri_h[-20:])[0,1],
        np.sum(te_h > np.median(te_all)),
    ]])
    sinal = int(rf.predict(feats_live)[0])
    prob  = float(rf.predict_proba(feats_live)[0][1])

    # Calcula pesos rebalanceados (últimos 252 dias antes da janela)
    hist252 = dados.loc[:data_inicio].iloc[-253:-1]
    top30   = ranking.head(30).index.tolist()
    pesos_rebalan = resolver_qp(
        hist252[top30].values, hist252["IBOV"].values, top30, cap=0.15
    )

    if sinal == 1:
        pesos_atual_ml = pesos_rebalan.copy()

    for nome, pesos in [("Estática",          pesos_base),
                         ("Sempre-Rebalanceia", pesos_rebalan),
                         ("ML-Guiada",          pesos_atual_ml)]:
        m, rc, ri = avaliar_pesos(pesos, df_teste)
        resultados_estrategias[nome].append({
            "janela":       k+1,
            "periodo":      f"{inicio} → {fim}",
            "sinal_ml":     sinal,
            "prob_rebalan": round(prob,2),
            "rebalanceou":  "sim" if nome=="Sempre-Rebalanceia"
                            else ("sim" if (nome=="ML-Guiada" and sinal==1) else "não"),
            **m
        })

print("\nResultados por estratégia e janela:")
for nome, rows in resultados_estrategias.items():
    df_r = pd.DataFrame(rows)
    print(f"\n--- {nome} ---")
    print(df_r[["janela","periodo","rebalanceou","tracking_error",
                "correlacao","retorno_carteira_acum","retorno_ibov_acum"]].to_string(index=False))

# Resumo comparativo
resumo_rows = []
for nome, rows in resultados_estrategias.items():
    df_r = pd.DataFrame(rows)
    resumo_rows.append({
        "Estratégia":          nome,
        "TE_médio":            round(df_r["tracking_error"].mean()*100, 4),
        "Correlação_média":    round(df_r["correlacao"].mean(), 4),
        "Retorno_carteira_%":  round(df_r["retorno_carteira_acum"].sum()*100, 2),
        "Nº_rebalanceamentos": df_r["rebalanceou"].eq("sim").sum(),
    })
resumo_df = pd.DataFrame(resumo_rows)
print("\nRESUMO COMPARATIVO (5 janelas OOS):")
print(resumo_df.to_string(index=False))
resumo_df.to_csv(f"{BASE}/ml_rebalanceamento_resumo.csv", index=False)
importancias.to_frame("importancia").to_csv(f"{BASE}/ml_feature_importances.csv")

# ---- Gráficos ML ----
# Fig 1: Importância das features
plt.figure(figsize=(7, 4))
importancias.plot(kind="barh", color="#2a6f97")
plt.xlabel("Importância (Gini)")
plt.title("Random Forest — O que mais indica 'hora de rebalancear'?")
plt.tight_layout()
plt.savefig("reports/figures/ml_feature_importances.png", dpi=130)
plt.close()

# Fig 2: Retorno acumulado total por estratégia (todas as 5 janelas concatenadas)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
cores = {"Estática": "#1f77b4", "Sempre-Rebalanceia": "#ff7f0e", "ML-Guiada": "#2ca02c"}
for k, df_teste in enumerate(testes):
    ax = axes[k]
    inicio, fim = janelas_teste[k]
    ri = df_teste["IBOV"].values
    ax.plot(df_teste.index, (1+ri).cumprod(), color="black",
            linewidth=2, label="IBOV")
    for nome, rows in resultados_estrategias.items():
        row = [r for r in rows if r["janela"]==k+1][0]
        pesos_plot = (pesos_base if nome == "Estática"
                      else resolver_qp(
                          dados.loc[:df_teste.index[0]].iloc[-253:-1][ranking.head(30).index.tolist()].values,
                          dados.loc[:df_teste.index[0]].iloc[-253:-1]["IBOV"].values,
                          ranking.head(30).index.tolist(), cap=0.15)
                      if nome == "Sempre-Rebalanceia"
                      else pesos_atual_ml)
        # Recalcular rc corretamente para cada estratégia
        if nome == "Estática":
            rc = df_teste[pesos_base.index].values @ pesos_base.values
        elif nome == "Sempre-Rebalanceia":
            hist252 = dados.loc[:df_teste.index[0]].iloc[-253:-1]
            top30 = ranking.head(30).index.tolist()
            p_r = resolver_qp(hist252[top30].values, hist252["IBOV"].values, top30, cap=0.15)
            rc = df_teste[p_r.index].values @ p_r.values
        else:
            # ML-guiada: pegar o sinal calculado acima
            sinal_j = [r["sinal_ml"] for r in resultados_estrategias["ML-Guiada"] if r["janela"]==k+1][0]
            if sinal_j == 1:
                hist252 = dados.loc[:df_teste.index[0]].iloc[-253:-1]
                top30 = ranking.head(30).index.tolist()
                p_r = resolver_qp(hist252[top30].values, hist252["IBOV"].values, top30, cap=0.15)
                rc = df_teste[p_r.index].values @ p_r.values
            else:
                rc = df_teste[pesos_base.index].values @ pesos_base.values
        ax.plot(df_teste.index, (1+rc).cumprod(),
                color=cores[nome], linewidth=1.4,
                linestyle="--" if nome=="Sempre-Rebalanceia" else "-",
                label=nome)
    rebalan = [r["rebalanceou"] for r in resultados_estrategias["ML-Guiada"] if r["janela"]==k+1][0]
    ax.set_title(f"Janela {k+1}\n{inicio}\n({rebalan.upper()} rebalanceamento)", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    if k == 0:
        ax.set_ylabel("Retorno acumulado (base 1.0)")

axes[0].legend(fontsize=7.5)
fig.suptitle("Estratégias de rebalanceamento: Estática vs Sempre-Rebalanceia vs ML-Guiada",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/ml_rebalanceamento_backtest.png", dpi=130)
plt.close()

print("\n[OK] Todos os arquivos da Parte B salvos.")
print("\nFim do script.")
