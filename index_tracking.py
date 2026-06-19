import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os

def load_data(data_dir):
    """
    Carrega os dados processados dos retornos dos ativos e do índice.
    """
    # Carrega retornos dos ativos
    asset_returns_path = os.path.join(data_dir, 'ibov_retornos_simples.csv')
    df_assets = pd.read_csv(asset_returns_path)
    df_assets['Date'] = pd.to_datetime(df_assets['Date'])
    df_assets.set_index('Date', inplace=True)
    
    # Carrega retornos do índice
    index_returns_path = os.path.join(data_dir, 'ibov_indice_retornos.csv')
    df_index = pd.read_csv(index_returns_path)
    df_index['Date'] = pd.to_datetime(df_index['Date'])
    df_index.set_index('Date', inplace=True)
    
    # O arquivo ibov_indice_retornos.csv possui 'Variação_Diária_%' em porcentagem
    if 'Variação_Diária_%' in df_index.columns:
        df_index['Index_Return'] = df_index['Variação_Diária_%'] / 100.0
    else:
        df_index['Index_Return'] = df_index['^BVSP'].pct_change()
        
    # Alinha as datas
    common_dates = df_assets.index.intersection(df_index.index)
    df_assets = df_assets.loc[common_dates].dropna()
    df_index = df_index.loc[common_dates].dropna()
    
    return df_assets, df_index['Index_Return']

def optimize_index_tracking(returns_assets, returns_index, K, verbose=False):
    """
    Formula e resolve o problema de otimização de Index Tracking usando Gurobi.
    
    returns_assets: DataFrame de retornos dos ativos (T x N)
    returns_index: Series de retornos do índice (T)
    K: Número máximo de ativos permitidos
    """
    T, N = returns_assets.shape
    assets = returns_assets.columns.tolist()
    
    R = returns_assets.values
    R_idx = returns_index.values
    
    # Configuração do ambiente do Gurobi (suprimir log se verbose=False)
    env = gp.Env(empty=True)
    if not verbose:
        env.setParam('OutputFlag', 0)
    env.start()
    
    model = gp.Model("Index_Tracking", env=env)
    
    # Variáveis
    # w_i: peso contínuo do ativo i
    w = model.addVars(N, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")
    # z_i: variável binária se o ativo i está no portfólio
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    
    # Formulação matricial do objetivo para reduzir drasticamente o tamanho do modelo (evitar erro de licença)
    # Minimizar: (1/T) * sum_t (sum_i w_i * r_{t,i} - R_t)^2
    # Isso equivale a: w^T * Q * w + c^T * w + constante
    # Onde Q = (R^T @ R)/T e c = -2*(R^T @ R_idx)/T
    Q = (R.T @ R) / T
    c = -2 * (R.T @ R_idx) / T
    constant = (R_idx.T @ R_idx) / T
    
    # Montagem da Função Objetivo
    obj = gp.quicksum(Q[i, j] * w[i] * w[j] for i in range(N) for j in range(N))
    obj += gp.quicksum(c[i] * w[i] for i in range(N))
    obj += constant
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Restrição: A soma dos pesos deve ser 1 (orçamento)
    model.addConstr(gp.quicksum(w[i] for i in range(N)) == 1, name="budget")
    
    # Restrição de alocação: w_i <= z_i (um ativo só pode ter peso se for selecionado)
    for i in range(N):
        model.addConstr(w[i] <= z[i], name=f"link_{i}")
        
    # Restrição de cardinalidade: Máximo de K ativos na carteira
    model.addConstr(gp.quicksum(z[i] for i in range(N)) <= K, name="max_assets")
    
    # Otimiza o modelo
    model.optimize()
    
    weights = {}
    if model.status == GRB.OPTIMAL:
        for i in range(N):
            if w[i].X > 1e-6:
                weights[assets[i]] = w[i].X
    return weights, model.ObjVal

def evaluate_out_of_sample(df_assets, df_index, K, window_size=252, step_size=63, num_portfolios=5):
    """
    Avalia a performance fora da amostra (out-of-sample) através de janelas deslizantes.
    Constrói as 5 carteiras fora da amostra exigidas no briefing.
    """
    total_needed = window_size + num_portfolios * step_size
    if total_needed > len(df_assets):
        print(f"Dados insuficientes para criar {num_portfolios} carteiras. Necessário {total_needed} dias, mas só temos {len(df_assets)}.")
        return
        
    print(f"Construindo {num_portfolios} carteiras para teste Out-of-Sample...\n")
    
    portfolio_returns_out = []
    index_returns_out = []
    
    for p in range(num_portfolios):
        start_in = p * step_size
        end_in = start_in + window_size
        end_out = end_in + step_size
        
        # Dados In-sample (treino)
        in_assets = df_assets.iloc[start_in:end_in]
        in_index = df_index.iloc[start_in:end_in]
        
        # Dados Out-of-sample (teste)
        out_assets = df_assets.iloc[end_in:end_out]
        out_index = df_index.iloc[end_in:end_out]
        
        print(f"--- Carteira {p+1} ---")
        print(f"  Treino (In-sample): {in_assets.index[0].date()} a {in_assets.index[-1].date()}")
        print(f"  Teste (Out-of-sample): {out_assets.index[0].date()} a {out_assets.index[-1].date()}")
        
        # Otimiza na janela in-sample
        weights, in_mse = optimize_index_tracking(in_assets, in_index, K)
        print(f"  MSE In-sample (Treino): {in_mse:.8f}")
        
        # Calcula retornos e performance na janela out-of-sample
        if weights:
            out_port_returns = out_assets[list(weights.keys())].dot(list(weights.values()))
            out_mse = ((out_port_returns - out_index) ** 2).mean()
            print(f"  MSE Out-of-sample (Teste): {out_mse:.8f}")
            
            portfolio_returns_out.extend(out_port_returns.values)
            index_returns_out.extend(out_index.values)
        else:
            print("  Erro: Nenhum peso viável encontrado na otimização!")
        print("")
        
    if portfolio_returns_out:
        overall_mse = np.mean((np.array(portfolio_returns_out) - np.array(index_returns_out))**2)
        print(f"--- Resumo Geral ---")
        print(f"MSE Total Out-of-Sample agregado das {num_portfolios} carteiras: {overall_mse:.8f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Otimização de Index Tracking usando Gurobi")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Caminho para o diretório de dados")
    parser.add_argument("--k", type=int, default=10, help="Número máximo de ativos na carteira (K)")
    parser.add_argument("--run_oos", action="store_true", help="Rodar a avaliação Out-of-Sample construindo 5 carteiras")
    args = parser.parse_args()
    
    print("Carregando dados...")
    df_assets, df_index = load_data(args.data_dir)
    print(f"Dados carregados com sucesso: {df_assets.shape[0]} dias, {df_assets.shape[1]} ativos disponíveis.\n")
    
    if args.run_oos:
        # Atendendo ao Briefing: Construir ao menos 5 carteiras fora da amostra
        evaluate_out_of_sample(df_assets, df_index, args.k, window_size=252, step_size=63, num_portfolios=5)
    else:
        print(f"Otimizando a carteira para K={args.k} usando todo o período...")
        weights, obj_val = optimize_index_tracking(df_assets, df_index, args.k, verbose=True)
        
        print("\nOtimização concluída.")
        print(f"MSE (Erro Quadrático Médio) final: {obj_val:.8f}")
        print(f"\nAtivos selecionados e seus respectivos pesos (Total: {len(weights)} ativos):")
        for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {asset}: {weight:.4f}")
