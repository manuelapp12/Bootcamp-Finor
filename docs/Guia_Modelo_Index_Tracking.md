# Guia: Modelo de Index Tracking do IBOV

## 1. Recomendação

Com base nos dados já tratados (`ibov_acoes_selecionadas.csv`, `ibov_indice_retornos.csv`) e na bibliografia do briefing (Cornuejols & Tütüncü; Santanna, Filomena & Borenstein), o modelo que entrega o melhor resultado para o estágio atual do projeto é:

**Quadratic Programming (QP) de Tracking Error Mínimo**, em duas etapas:

1. **Seleção de ativos** (parcialmente feita): correlação + RFECV já reduziram o universo de ~80 para ~54 ações. Vale reduzir mais, para ~10-20 ações, que é o cenário real de "index tracking" (replicar o índice com poucos ativos).
2. **Otimização de pesos**: dado o subconjunto de ações, resolver um problema de otimização que encontra os pesos `w_i` que minimizam a diferença entre o retorno da carteira e o retorno do IBOV.

Isso é exatamente o "Modelo Matemático" pedido no briefing, é implementável 100% em Python open-source (sem Gurobi) com a biblioteca `cvxpy`, e é o modelo padrão da literatura citada.

### Por que não usar só Elastic Net / Lasso (sklearn)?
É uma alternativa válida e mais simples (vocês já têm `RFECV` rodando), mas:
- Regressão não garante `soma dos pesos = 1` (carteira 100% investida).
- Pode gerar pesos negativos (posições a descoberto), o que não é realista para um fundo passivo.
- O briefing pede explicitamente um "modelo de otimização" resolvido com solver — Elastic Net é um proxy estatístico, não um modelo de otimização de portfólio.

**Use Elastic Net como benchmark/baseline** (já que está quase pronto no `corellacao.py`), mas o modelo principal a apresentar deve ser o QP.

---

## 2. Passo a passo (para quem não é programador)

### Passo 0 — Instalar a ferramenta de otimização
No terminal, com o ambiente virtual ativado:
```bash
pip install cvxpy
```
`cvxpy` é gratuito e já vem com solvers open-source (OSQP, ECOS) — não precisa de licença.

### Passo 1 — Carregar os dados (vocês já têm isso pronto)
```python
import pandas as pd

retornos = pd.read_csv('data/processed/ibov_acoes_selecionadas.csv', index_col=0, parse_dates=True)
indice = pd.read_csv('data/processed/ibov_indice_retornos.csv', index_col=0, parse_dates=True)

# Alinhar os dois datasets pela data
dados = retornos.join(indice['Variação_Diária_%'], how='inner').dropna()
dados = dados.rename(columns={'Variação_Diária_%': 'IBOV'})
dados['IBOV'] = dados['IBOV'] / 100  # se a coluna estiver em %, converter para fração
```

### Passo 2 — Separar treino (in-sample) e teste (out-of-sample)
O briefing pede pelo menos 5 carteiras fora da amostra. Sugestão: usar janelas trimestrais nos últimos 18 meses como teste, e tudo antes como treino.

```python
treino = dados.loc[:'2024-12-31']
testes = []
# 5 janelas trimestrais consecutivas a partir de 2025
for inicio, fim in [
    ('2025-01-01','2025-03-31'),
    ('2025-04-01','2025-06-30'),
    ('2025-07-01','2025-09-30'),
    ('2025-10-01','2025-12-31'),
    ('2026-01-01','2026-03-31'),
]:
    testes.append(dados.loc[inicio:fim])
```

### Passo 3 — Montar e resolver o modelo de otimização (QP)
Ideia em palavras simples: "encontre os pesos `w` de cada ação, somando 100% e sem posições negativas, que fazem a carteira se mover o mais parecido possível com o IBOV no período de treino."

```python
import cvxpy as cp
import numpy as np

acoes = [c for c in dados.columns if c != 'IBOV']
R = treino[acoes].values        # retornos diários das ações (treino)
y = treino['IBOV'].values       # retorno diário do IBOV (treino)

w = cp.Variable(len(acoes))

erro_tracking = R @ w - y
objetivo = cp.Minimize(cp.sum_squares(erro_tracking))

restricoes = [
    cp.sum(w) == 1,   # carteira 100% investida
    w >= 0,           # sem posições a descoberto (long-only)
    w <= 0.15,        # opcional: limite de concentração por ativo (15%)
]

problema = cp.Problem(objetivo, restricoes)
problema.solve()

pesos = pd.Series(w.value, index=acoes).sort_values(ascending=False)
print(pesos[pesos > 0.001])  # mostra só os ativos com peso relevante
```

> Dica: ativos com peso ~0 são os que o modelo "descartou" — essa é a redução de cardinalidade na prática.

### Passo 4 — Backtest fora da amostra (as 5 carteiras)
```python
resultados = []
for i, teste in enumerate(testes, start=1):
    retorno_carteira = teste[acoes].values @ pesos.values
    retorno_indice = teste['IBOV'].values

    tracking_error = np.std(retorno_carteira - retorno_indice)
    correlacao = np.corrcoef(retorno_carteira, retorno_indice)[0, 1]

    resultados.append({
        'janela': i,
        'tracking_error': tracking_error,
        'correlacao': correlacao,
        'retorno_carteira_acumulado': (1 + retorno_carteira).prod() - 1,
        'retorno_ibov_acumulado': (1 + retorno_indice).prod() - 1,
    })

pd.DataFrame(resultados)
```

### Passo 5 — Visualizar
```python
import matplotlib.pyplot as plt

teste = testes[0]
carteira_acum = (1 + teste[acoes].values @ pesos.values).cumprod()
ibov_acum = (1 + teste['IBOV'].values).cumprod()

plt.plot(teste.index, carteira_acum, label='Carteira (tracking)')
plt.plot(teste.index, ibov_acum, label='IBOV')
plt.legend()
plt.title('Carteira de Index Tracking vs IBOV — Janela 1 (fora da amostra)')
plt.show()
```

---

## 3. Como melhorar o resultado (próximos passos)

1. **Reduzir o número de ativos**: testar o modelo com 10, 15 e 20 ações (os pesos mais altos do Passo 3) e comparar o tracking error — o "melhor resultado" geralmente é o menor número de ativos que ainda mantém tracking error baixo. Isso é o trade-off central do problema.
2. **Rebalanceamento**: ao invés de pesos fixos para todo o período de teste, refazer a otimização a cada janela (rolling window), reotimizando com os dados mais recentes.
3. **Comparar com Elastic Net** (o que o `corellacao.py` já está construindo): rodar os dois modelos e mostrar na apresentação qual teve menor tracking error fora da amostra — isso por si só é um resultado interessante para os 20 minutos de apresentação.
4. **Cardinalidade explícita (avançado)**: se quiserem ir além, a versão "completa" da literatura citada (Santanna et al.) usa variáveis binárias para limitar o número de ativos diretamente dentro da otimização (MIQP). Isso exige um solver mais robusto (Gurobi tem licença acadêmica gratuita) — mas não é necessário para entregar um resultado sólido na Entrega 1.

---

## 4. Resumo para a apresentação

- **O que o modelo faz** (linguagem humana): "Escolhemos um pequeno grupo de ações da B3 e calculamos o quanto investir em cada uma para que essa carteira simplificada se comporte o mais parecido possível com o Ibovespa."
- **Como validamos**: testamos a carteira em 5 períodos que o modelo nunca viu, e medimos o quão perto ela ficou do índice real (tracking error e correlação).
- **O que ainda não sabemos**: qual o número mínimo de ações que ainda entrega um tracking error aceitável — esse é o próximo experimento.
