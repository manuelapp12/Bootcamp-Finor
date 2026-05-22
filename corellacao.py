import pandas as pd
import numpy as np

dados = pd.read_csv('./data/ibov_retornos_simples.csv', index_col=0, parse_dates=True)
indice = pd.read_csv('./data/ibov_indice.csv', index_col=0, parse_dates=True)

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
dados_filtrados = dados.drop(columns=['Ações a ser escolhidas'], errors='ignore')
print("\nDataFrame após remoção das ações com alta correlação:")
print(dados_filtrados)

# Puxa a série do índice diretamente do DataFrame 'indice' que foi carregada
serie_ibov = indice['^BVSP']

#Calcula a correlação cruzando o DataFrame de ações com a Série do índice
corr_com_indice = dados_filtrados.corrwith(serie_ibov)
print("\nCorrelação com IBOV (após remover ações escolhidas):")
print(corr_com_indice.sort_values(ascending=False))

