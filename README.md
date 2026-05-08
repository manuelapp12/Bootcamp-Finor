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

# Etapas:
A implementação do modelo de otimização deve ser realizada em Python, podendo ser utilizadas API’s de Python de solvers como Gurobi ou de solvers open-source:
      1. Importar os dados a partir de arquivos ou a partir de API’s de fontes de dados de ações
      2. Explorar os dados das ações e verificar se há dados faltantes para alguma delas para o período de análise
      3. Tratar informações se necessário, removendo valores discrepantes.
      4. Desenvolver o modelo de otimização.
      5. Resolver o problema de otimização.
      6. Analisar os diferentes resultados do modelo, comparando a performance do índice com a carteira. Definir um período para teste dentro da amostra e fora da amostra: construir ao menos 5 carteiras fora da amostra e avaliar sua performance.
      7. Apresentar os resultados em um jupyter notebook.
