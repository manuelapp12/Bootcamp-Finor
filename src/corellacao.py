import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Importa os dados de retornos simples das ações e do índice IBOV
dados = pd.read_csv('data/processed/ibov_retornos_simples.csv', index_col=0, parse_dates=True)
indice = pd.read_csv('data/processed/ibov_indice_retornos.csv', index_col=0, parse_dates=True)

# =====================================================================================
# Seleção através da correlação
# =====================================================================================


# Calcula a matriz de correlação entre todas as colunas numéricas
corr = dados.corr(method='pearson')
print("Matriz de Correlação:")
print(corr)

# Mantém apenas o triângulo superior da matriz para evitar duplicatas (ex: A-B e B-A) 
# e exclui a diagonal principal (A-A), que é sempre 1.0
filtro = np.triu(np.ones(corr.shape), k=1).astype(bool)
corr_arrumada = corr.where(filtro)

# Transforma a matriz em formato de tabela longa, remove os NaNs e renomeia
tabela_pares = corr_arrumada.unstack().dropna().reset_index()
tabela_pares.columns = ['Acao_1', 'Acao_2', 'Correlacao']

# Filtra apenas pares com correlação > 0.8
tabela_alta_corr = tabela_pares[(tabela_pares['Correlacao'] > 0.8) | (tabela_pares['Correlacao'] < -0.8)]
print("\nPares com correlação > |0.8|:")
print(tabela_alta_corr.sort_values(by='Correlacao', ascending=False))

# Cria novo dataframe sem essas ações
dados_filtrados = dados.drop(columns=['PETR3.SA','BBDC3.SA','GOAU4.SA','AXIA6.SA','ITUB4.SA','BRAP4.SA','BBDC4.SA'], errors='ignore')
print("\nDataFrame após remoção das ações com alta correlação:")
print(dados_filtrados)

# Puxa a série do índice diretamente do DataFrame 'indice' que foi carregada
serie_ibov = indice['Variação_Diária_%']

#Calcula a correlação cruzando o DataFrame de ações com a Série do índice
corr_com_indice = dados_filtrados.corrwith(serie_ibov)
print("\nCorrelação com IBOV (após remover ações escolhidas):")
print(corr_com_indice.sort_values(ascending=False))

# =====================================================================================
# SELEÇÃO AUTOMÁTICA ÓTIMA DE AÇÕES (RFECV)
# =====================================================================================

#Prepara os dados
dados_completos = dados_filtrados.copy()
dados_completos['IBOV_Alvo'] = serie_ibov
dados_completos = dados_completos.dropna()

X = dados_completos.drop(columns=['IBOV_Alvo', 'IBOV', '^BVSP'], errors='ignore')
y = dados_completos['IBOV_Alvo']

#Padroniza os dados
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

#Define o "Juiz"
modelo_juiz = LinearRegression()

# Configura o RFECV
seletor_rfecv = RFECV(estimator=modelo_juiz, step=1, cv=5)

#Treina o modelo (O algoritmo vai testar todas as combinações)
seletor_rfecv.fit(X_escalado, y)

#Ve os resultados dinâmicos
numero_ideal_acoes = seletor_rfecv.n_features_

print(f"\nO RFECV descobriu que o número IDEAL de ações para o modelo é: {numero_ideal_acoes}")

# Cria a tabela com os resultados (caso não tenha criado no bloco anterior)
tabela_rfecv = pd.DataFrame({
    'Acao': X.columns,
    'Sobreviveu': seletor_rfecv.support_,
    'Ranking': seletor_rfecv.ranking_
})

# Filtra as ações que não sobreviveram (False) e ordenamos pelo Ranking
acoes_perdedoras = tabela_rfecv[tabela_rfecv['Sobreviveu'] == False].sort_values(by='Ranking', ascending=False)

print("\n As Ações eliminadas")
print(acoes_perdedoras[['Acao', 'Ranking']])

# Pega apenas os nomes (tickers) das ações perdedoras numa lista
lista_perdedoras = acoes_perdedoras['Acao'].tolist()

# Cria o banco final simplesmente apagando as perdedoras do banco original
dados_finais = dados_filtrados.drop(columns=lista_perdedoras, errors='ignore')

print("\nNovo Banco de Dados ")
print(dados_finais)

#Sala os retornos das ações selecionadas em um novo arquivo CSV
dados_finais.to_csv('data/processed/ibov_acoes_selecionadas.csv')