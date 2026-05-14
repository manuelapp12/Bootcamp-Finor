import pandas as pd 
import numpy as np
import yfinance as yf 

dataset = pd.read_csv(
    './data/IBOVDia_11-05-26.csv',
    sep=';',
    skipfooter=2, 
    skiprows=1,
    decimal=',', 
    thousands='.',
    engine='python',
    encoding='latin-1',
    index_col=False
)

dataset.columns = ['ticker','name','type','qtd','part']
print(f"Dataset lido com {dataset.shape[0]} ações do IBOVESPA.")

tickers_ibov = dataset['ticker'].astype(str) + '.SA'
tickers_ibov_list = tickers_ibov.tolist()

ticker_idx_ibov = '^BVSP'
inicio = '2018-01-01'
fim = '2025-05-01'
print("\nBaixando dados do índice IBOVESPA...")
indice_ibov = yf.download(ticker_idx_ibov, start=inicio, end=fim, auto_adjust=True, progress=False)['Close']


print("\nBaixando dados das ações do IBOVESPA...")
dados_ibov = yf.download(tickers_ibov_list, start=inicio, end=fim, auto_adjust=True, progress=False)['Close']

dados_ibov.to_csv('./data/ibov_precos.csv')
indice_ibov.to_csv('./data/ibov_indice.csv')


print(f"\nDownload concluído!")
print(f"Dados Ações: {dados_ibov.shape[1]} ações x {dados_ibov.shape[0]} dias.")
print(f"Período: {dados_ibov.index[0].date()} até {dados_ibov.index[-1].date()}")

# ==============================================================================
# 5. DIAGNÓSTICO E LIMPEZA DE DADOS FALTANTES
# ==============================================================================

print("\nVerificando dados faltantes...")
nulos_ib = dados_ibov.isnull().sum()

pct_ib = (nulos_ib / len(dados_ibov) * 100).round(2)

print("Ações com mais de 5% de dados faltantes:")
print(pct_ib[pct_ib > 5].sort_values(ascending=False))

# Decisão: Remover ações com mais de 20% de dados faltantes (Ex: EMBJ3.SA)
limite_remocao = 20
acoes_validas_ib = pct_ib[pct_ib <= limite_remocao].index
dados_ibov_clean = dados_ibov[acoes_validas_ib].copy()

# Ações invalidas para mostrar no Jupyter
acoes_invalidas = pct_ib[pct_ib > limite_remocao].index
print(f'\nLimpeza: {dados_ibov.shape[1]} -> {dados_ibov_clean.shape[1]} ações válidas após remover as que tem >20% faltantes.')

# Preencher o resto com forward fill (feriados, suspensões temporárias)
nulos_ib = dados_ibov_clean.isnull().sum()
tickers_to_fowarfill = nulos_ib[nulos_ib > 0]
# 1. Aplica o ffill apenas nas colunas específicas que têm buracos
dados_ibov_clean[tickers_to_fowarfill.index] = dados_ibov_clean[tickers_to_fowarfill.index].ffill()

# 2. Depois, remove qualquer linha que tenha ficado 100% vazia (feriados globais)
dados_ibov_clean = dados_ibov_clean.dropna(how='all')

# ==============================================================================
# 6. AUDITORIA DE QUALIDADE (DUPLICATAS E TIPOS)
# ==============================================================================
print("\nRealizando auditoria final de qualidade...")
print(f"Linhas duplicadas no dataset: {dados_ibov_clean.index.duplicated().sum()}")
print(f"Formato da tabela limpa (Shape): {dados_ibov_clean.shape}")
print(f"Tipos de dados presentes no dataframe: {dados_ibov_clean.dtypes.unique()}")

# ==============================================================================
# 7. CÁLCULO DE RETORNOS LOGARÍTMICOS
# ==============================================================================
print("\nCalculando retornos logarítmicos...")
ret_ibov_clean = np.log(dados_ibov_clean / dados_ibov_clean.shift(1)).dropna()
ret_idx_ib = np.log(indice_ibov / indice_ibov.shift(1)).dropna()

print(f'Formato final dos retornos do IBOV: {ret_ibov_clean.shape}')
print(f'Retorno médio diário do IBOV (índice): {float(ret_idx_ib.mean().iloc[0]):.4%}')
print(f'Volatilidade diária do IBOV (índice): {float(ret_idx_ib.std().iloc[0]):.4%}')

# Salvar retornos para a análise exploratória (Jupyter Notebook)
ret_ibov_clean.to_csv('./data/ibov_retornos.csv')
print("\nRetornos salvos em './data/ibov_retornos.csv' com sucesso!")