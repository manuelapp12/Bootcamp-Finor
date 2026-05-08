# Bootcamp-Finor

# Problemática Inicial
Os modelos de index tracking buscam reproduzir o comportamento de índices de ações com um número de ações inferior ao do índice original, simplificando a composição do portifólio e, consequentemente, o custo de manutenção do portfólio. O objetivo desse projeto é desenvolver um modelo de index tracking para os índices S&P100 e IBOV.

# Motivação:
Os índices de mercado (por exemplo, S&P500) compostos por ações individuais e representam termômetros dos mercados de
ações. Fundos de investimento passivo visam replicar o desempenho de um índice de mercado ou de um segmento específico do mercado.

# Dados: 
Os participantes devem buscar os dados históricos dos últimos 7 anos dos índices S&P 100 e IBOVESPA para serem utilizados como inputs. Os dados podem ser encontrados no Yahoo Finance e outros locais de interesse do grupo.

# Index tracking (IT):
      IT consiste em uma estratégia de replicar um índice de mercado utilizando um número menor de ações.
      IT apresenta menores custos e, por ter menos ativos, fornece uma carteira de ações mais simples de ser gerida.
      IT no formato de problema de otimização pode ser incrementado com restrições de negócio.

# Etapas sugeridas:
      A implementação do modelo de otimização deve ser realizada em Python, podendo ser utilizadas API’s de Python de solvers como Gurobi ou de solvers open-source:
      1. Importar os dados a partir de arquivos ou a partir de API’s de fontes de dados de ações
      2. Explorar os dados das ações e verificar se há dados faltantes para alguma delas para o período de análise
      3. Tratar informações se necessário, removendo valores discrepantes.
      4. Desenvolver o modelo de otimização.
      5. Resolver o problema de otimização.
      6. Analisar os diferentes resultados do modelo, comparando a performance do índice com a carteira. Definir um período para teste dentro da amostra e fora da amostra: construir ao menos 5 carteiras fora da amostra e avaliar sua performance.
      7. Apresentar os resultados em um jupyter notebook.

# Etapas desenhadas pelo grupo:
      1. Criar um modelo de Index Tracking (IT) do IBOV
      // Isso inclui documentar as etapas no notebook, descrevendo as fontes de dados, os critérios de escolha e as variáveis selecionadas, além de tratar dados ausentes, outliers e realizar feature engineering. A análise exploratória deve revelar insights iniciais.
            1 Realizar: 
                  1.2 coleta;
                        1.2.1 Buscar os dados históricos dos últimos 7 anos dos índices IBOV;
                  1.3 limpeza;
                        1.3.1 Carregar o dataset e verificar shape, dtypes e head
                        1.3.2 Identificar e documentar o significado de cada coluna
                        1.3.3 Verificar o cabeçalho e o delimitador do arquivo
                  1.4 pré-processamento; e
                        1.4.1 Analisar a % de valores ausentes por coluna
                        1.4.2 Definir e aplicar a estratégia de tratamento de missing values
                        1.4.3 Identificar e documentar os outliers encontrador
                        1.4.4 Decidir o que fazer com cada outlier (manter, recmover ou transformar)
                        1.4.5 Verificar e remover duplicatas
                        1.4.6 Padronizar tipos de dados (datas, textos, números)
                  1.5 análise exploratória dos dados.  
            
      2. Criar um modelo de Index Tracking (IT) do S&P100
            2.2 Buscar os dados históricos dos últimos 7 anos dos índices S&P100
