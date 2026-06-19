import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.linear_model import ElasticNet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

BASE = "data/processed"

# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
retornos = pd.read_csv(f"{BASE}/ibov_acoes_selecionadas.csv", index_col=0, parse_dates=True)
indice = pd.read_csv(f"{BASE}/ibov_indice_retornos.csv", index_col=0, parse_dates=True)

dados = retornos.join(indice["Variação_Diária_%"], how="inner").dropna()
dados = dados.rename(columns={"Variação_Diária_%": "IBOV"})
dados["IBOV"] = dados["IBOV"] / 100.0  # converter de % para fração

acoes = [c for c in dados.columns if c != "IBOV"]
print(f"Periodo: {dados.index.min().date()} a {dados.index.max().date()} | {len(dados)} dias | {len(acoes)} acoes")

# ---------------------------------------------------------------------------
# 2. Split treino / 5 janelas de teste fora da amostra (trimestres)
# ---------------------------------------------------------------------------
treino = dados.loc[:"2024-01-31"]

janelas_teste = [
    ("2024-02-01", "2024-04-30"),
    ("2024-05-01", "2024-07-31"),
    ("2024-08-01", "2024-10-31"),
    ("2024-11-01", "2025-01-31"),
    ("2025-02-01", "2025-04-30"),
]
testes = [dados.loc[i:f] for i, f in janelas_teste]

print(f"Treino: {treino.index.min().date()} a {treino.index.max().date()} | {len(treino)} dias")
for k, (i, f) in enumerate(janelas_teste, 1):
    print(f"  Janela {k}: {i} a {f} | {len(testes[k-1])} dias")

X_train = treino[acoes].values
y_train = treino["IBOV"].values

# ---------------------------------------------------------------------------
# 3. Modelo QP de minimo tracking error (long-only, soma = 1)
# ---------------------------------------------------------------------------
def resolver_qp(X, y, ativos, cap=0.25):
    n = len(ativos)
    w = cp.Variable(n)
    erro = X @ w - y
    objetivo = cp.Minimize(cp.sum_squares(erro))
    restricoes = [cp.sum(w) == 1, w >= 0, w <= cap]
    prob = cp.Problem(objetivo, restricoes)
    prob.solve(solver=cp.OSQP)
    pesos = pd.Series(np.maximum(w.value, 0), index=ativos)
    pesos = pesos / pesos.sum()
    return pesos

# 3a. Resolver com universo completo (54 acoes) para ranquear por peso
pesos_full = resolver_qp(X_train, y_train, acoes, cap=0.15)
ranking = pesos_full.sort_values(ascending=False)
print("\nTop 15 acoes (universo completo, treino):")
print(ranking.head(15))

# ---------------------------------------------------------------------------
# 4. Avaliar diferentes cardinalidades (K)
# ---------------------------------------------------------------------------
def avaliar_pesos(pesos, df_periodo):
    ativos = pesos.index.tolist()
    ret_carteira = df_periodo[ativos].values @ pesos.values
    ret_indice = df_periodo["IBOV"].values
    diff = ret_carteira - ret_indice
    return {
        "tracking_error": float(np.std(diff)),
        "erro_medio_abs": float(np.mean(np.abs(diff))),
        "correlacao": float(np.corrcoef(ret_carteira, ret_indice)[0, 1]),
        "retorno_carteira_acum": float((1 + ret_carteira).prod() - 1),
        "retorno_ibov_acum": float((1 + ret_indice).prod() - 1),
    }, ret_carteira, ret_indice

Ks = [10, 15, 20, 30, 54]
resultados_K = []
pesos_por_K = {}

for K in Ks:
    top_ativos = ranking.head(K).index.tolist()
    cap = min(0.30, max(0.15, 1.5 / K))
    Xk = treino[top_ativos].values
    pesos_k = resolver_qp(Xk, y_train, top_ativos, cap=cap)
    pesos_por_K[K] = pesos_k

    tes = []
    for df_teste in testes:
        m, _, _ = avaliar_pesos(pesos_k, df_teste)
        tes.append(m["tracking_error"])
    resultados_K.append({
        "K": K,
        "tracking_error_medio_oos": float(np.mean(tes)),
        "tracking_error_max_oos": float(np.max(tes)),
    })

df_K = pd.DataFrame(resultados_K)
print("\nComparacao de cardinalidade (tracking error fora da amostra, em fracao do retorno diario):")
print(df_K)

# Escolha do K: melhor custo-beneficio -> menor K cujo tracking error medio
# esta a no maximo 10% acima do melhor (54 acoes = referencia "sem reducao")
best_te = df_K["tracking_error_medio_oos"].min()
candidatos = df_K[df_K["tracking_error_medio_oos"] <= best_te * 1.10]
K_escolhido = int(candidatos["K"].min())
print(f"\nK escolhido: {K_escolhido}")

pesos_finais = pesos_por_K[K_escolhido]
pesos_finais = pesos_finais[pesos_finais > 0.001].sort_values(ascending=False)
print("\nPesos finais da carteira (QP):")
print(pesos_finais)

# ---------------------------------------------------------------------------
# 5. Backtest detalhado das 5 janelas com o K escolhido
# ---------------------------------------------------------------------------
backtest_qp = []
for k, df_teste in enumerate(testes, 1):
    m, ret_c, ret_i = avaliar_pesos(pesos_finais, df_teste)
    m["janela"] = k
    m["modelo"] = "QP_tracking"
    m["periodo"] = f"{janelas_teste[k-1][0]} a {janelas_teste[k-1][1]}"
    backtest_qp.append(m)

# ---------------------------------------------------------------------------
# 6. Benchmark Elastic Net (mesmo numero de ativos K)
# ---------------------------------------------------------------------------
enet = ElasticNet(alpha=0.0005, l1_ratio=0.5, positive=True, max_iter=20000)
enet.fit(X_train, y_train)
coefs = pd.Series(enet.coef_, index=acoes)
top_enet = coefs[coefs > 0].sort_values(ascending=False).head(K_escolhido)
pesos_enet = top_enet / top_enet.sum()
print("\nPesos Elastic Net (benchmark):")
print(pesos_enet)

backtest_enet = []
for k, df_teste in enumerate(testes, 1):
    m, ret_c, ret_i = avaliar_pesos(pesos_enet, df_teste)
    m["janela"] = k
    m["modelo"] = "ElasticNet"
    m["periodo"] = f"{janelas_teste[k-1][0]} a {janelas_teste[k-1][1]}"
    backtest_enet.append(m)

backtest_df = pd.DataFrame(backtest_qp + backtest_enet)
print("\nBacktest fora da amostra (5 janelas):")
print(backtest_df[["modelo", "janela", "periodo", "tracking_error", "correlacao",
                    "retorno_carteira_acum", "retorno_ibov_acum"]])

resumo = backtest_df.groupby("modelo")[["tracking_error", "correlacao"]].mean()
print("\nResumo medio por modelo:")
print(resumo)

# ---------------------------------------------------------------------------
# 7. Salvar resultados
# ---------------------------------------------------------------------------
pesos_finais.to_frame("peso").to_csv(f"{BASE}/index_tracking_pesos_qp.csv")
pesos_enet.to_frame("peso").to_csv(f"{BASE}/index_tracking_pesos_elasticnet.csv")
backtest_df.to_csv(f"{BASE}/index_tracking_backtest.csv", index=False)
df_K.to_csv(f"{BASE}/index_tracking_cardinalidade.csv", index=False)

with open(f"{BASE}/index_tracking_resumo.json", "w") as f:
    json.dump({
        "K_escolhido": K_escolhido,
        "n_acoes_universo": len(acoes),
        "tracking_error_medio_qp": float(resumo.loc["QP_tracking", "tracking_error"]),
        "tracking_error_medio_enet": float(resumo.loc["ElasticNet", "tracking_error"]),
        "correlacao_media_qp": float(resumo.loc["QP_tracking", "correlacao"]),
        "correlacao_media_enet": float(resumo.loc["ElasticNet", "correlacao"]),
    }, f, indent=2)

# ---------------------------------------------------------------------------
# 8. Figuras
# ---------------------------------------------------------------------------
import os
os.makedirs("reports/figures", exist_ok=True)

# 8a. Tracking error medio vs K
plt.figure(figsize=(7, 4.5))
plt.plot(df_K["K"], df_K["tracking_error_medio_oos"], marker="o", label="Tracking error medio (OOS)")
plt.axvline(K_escolhido, color="red", linestyle="--", label=f"K escolhido = {K_escolhido}")
plt.xlabel("Numero de acoes na carteira (K)")
plt.ylabel("Tracking error medio (desvio padrao da diferenca de retornos diarios)")
plt.title("Trade-off: numero de acoes vs erro de tracking (fora da amostra)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/index_tracking_cardinalidade.png", dpi=130)
plt.close()

# 8b. Pesos da carteira final
plt.figure(figsize=(8, 5))
pesos_finais.plot(kind="bar", color="#2a6f97")
plt.ylabel("Peso na carteira")
plt.title(f"Pesos da carteira de Index Tracking (QP, K={K_escolhido})")
plt.tight_layout()
plt.savefig("reports/figures/index_tracking_pesos.png", dpi=130)
plt.close()

# 8c. Retorno acumulado por janela (carteira QP vs ElasticNet vs IBOV)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for k, df_teste in enumerate(testes, 1):
    ret_c_qp = df_teste[pesos_finais.index].values @ pesos_finais.values
    ret_c_en = df_teste[pesos_enet.index].values @ pesos_enet.values
    ret_i = df_teste["IBOV"].values

    acc_qp = (1 + ret_c_qp).cumprod()
    acc_en = (1 + ret_c_en).cumprod()
    acc_i = (1 + ret_i).cumprod()

    ax = axes[k - 1]
    ax.plot(df_teste.index, acc_qp, label="Carteira QP")
    ax.plot(df_teste.index, acc_en, label="Carteira ElasticNet", linestyle="--")
    ax.plot(df_teste.index, acc_i, label="IBOV", color="black", linewidth=1.5)
    ax.set_title(f"Janela {k}\n{janelas_teste[k-1][0]} a {janelas_teste[k-1][1]}", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    if k == 1:
        ax.set_ylabel("Retorno acumulado (base 1.0)")

axes[0].legend(fontsize=8)
fig.suptitle("Carteira de Index Tracking vs IBOV - 5 janelas fora da amostra")
plt.tight_layout()
plt.savefig("reports/figures/index_tracking_backtest.png", dpi=130)
plt.close()

print("\nOK - arquivos salvos em data/processed/ e reports/figures/")
