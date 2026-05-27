import pandas as pd
import os

def main():
    # Caminho do arquivo de entrada e saída
    input_file = "data/ibov_indice.csv"
    output_file = "data/ibov_retornos.csv"
    
    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Erro: O arquivo {input_file} não foi encontrado.")
        return
        
    print(f"Lendo dados de {input_file}...")
    df = pd.read_csv(input_file)
    
    # Certificar-se de que a coluna de data é do tipo datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Ordenar por data caso não esteja ordenado (importante para pct_change)
    df = df.sort_values(by='Date')
    
    # Calcular a variação percentual simples diária para a coluna ^BVSP
    # A função pct_change() calcula: (Atual - Anterior) / Anterior
    # Multiplicamos por 100 para converter para porcentagem
    df['Variação_Diária_%'] = df['^BVSP'].pct_change() * 100
    
    # Salvar o resultado na pasta data
    print(f"Salvando o resultado em {output_file}...")
    df.to_csv(output_file, index=False)
    print("Operação concluída com sucesso!")

if __name__ == "__main__":
    main()
