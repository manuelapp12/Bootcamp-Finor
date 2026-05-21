# Bootcamp-Finor

# Problemática Inicial
Os modelos de index tracking buscam reproduzir o comportamento de índices de ações com um número de ações inferior ao do índice original, simplificando a composição do portifólio e, consequentemente, o custo de manutenção do portfólio. O objetivo desse projeto é desenvolver um modelo de index tracking para os índices S&P100 e IBOV.

# Motivação:
Os índices de mercado (por exemplo, S&P500) compostos por ações individuais e representam termômetros dos mercados de
ações. Fundos de investimento passivo visam replicar o desempenho de um índice de mercado ou de um segmento específico do mercado.

# Dados: 
Os participantes devem buscar os dados históricos dos últimos 7 anos dos índices S&P 100 e IBOVESPA para serem utilizados como inputs. Os dados podem ser encontrados no Yahoo Finance e outros locais de interesse do grupo.

### Dicionário de Dados (Carteira IBOVESPA)
A base de dados original da carteira teórica (`IBOVDia_11-05-26.csv`) contém as seguintes colunas, que foram identificadas e renomeadas no script `eda.py` (linha 17) para facilitar o processamento:
* **ticker** (Original: `Código`): Código de negociação do ativo na B3 (ex: PETR4). No código, adicionamos dinamicamente o sufixo `.SA` para compatibilidade com o Yahoo Finance.
* **name** (Original: `Ação`): Nome resumido da empresa emissora do ativo (ex: PETROBRAS).
* **type** (Original: `Tipo`): Classe da ação (ex: ON = Ordinária, PN = Preferencial).
* **qtd** (Original: `Qtde. Teórica`): Quantidade teórica de ações daquela empresa que compõem o índice.
* **part** (Original: `Part. (%)`): Peso percentual daquele ativo na composição total do índice IBOVESPA.

# Configuração do Ambiente

Este projeto é multiplataforma e pode ser rodado em **Linux, macOS ou Windows**. 

Siga os passos abaixo para configurar seu ambiente de desenvolvimento:

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/[seu-usuario]/Bootcamp-Finor.git
   cd Bootcamp-Finor
   ```

2. **Criar um ambiente virtual (venv):**
   *   No **Linux/macOS**:
       ```bash
       python3 -m venv .venv
       ```
   *   No **Windows**:
       ```bash
       python -m venv .venv
       ```

3. **Ativar o ambiente virtual:**
   *   No **Linux/macOS**:
       ```bash
       source .venv/bin/activate
       ```
   *   No **Windows (PowerShell)**:
       ```bash
       .venv\Scripts\Activate.ps1
       ```
   *   No **Windows (Prompt de Comando)**:
       ```bash
       .venv\Scripts\activate
       ```

4. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

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
      Isso inclui documentar as etapas no notebook, descrevendo as fontes de dados, os critérios de escolha e as variáveis selecionadas, além de tratar dados ausentes, outliers e realizar feature engineering. A análise exploratória deve revelar insights iniciais.
            1 Briefing_Entrega_TG1: 
                  1.1 coleta;
                        1.1.1 - Arthur
                              [x] Buscar os dados históricos dos últimos 7 anos dos índices IBOV;
                  1.2 limpeza;
                        1.2.1 - Arthur
                              [x] Carregar o dataset e verificar shape, dtypes e head
                        1.2.2 - Laura
                              [x] Identificar e documentar o significado de cada coluna
                        1.2.3 - Arthur
                              [x] Verificar o cabeçalho e o delimitador do arquivo
                        1.2.4 - Decisão Técnica Conjunta
                              [x] Identificado que a ação da Embraer estava listada como `EMBJ3` no dataset bruto. Devido à mudança de ticker e ausência de histórico consolidado para `EMBJ3.SA` (ou falha na API para `EMBR3.SA`) no Yahoo Finance, o ativo retornou 100% de dados nulos. A decisão técnica para a limpeza de dados foi enquadrar o ativo como **Dado Faltante (Missing Data)** estrutural e **remover a ação da análise** (aplicando o critério de exclusão para ações com >20% de dados faltantes). Como o modelo de Index Tracking visa selecionar um subconjunto restrito de ações, a exclusão pontual deste ativo não inviabiliza o projeto.

                        Estratégias para limpeza
                           1. Artur
                              [x] Criar a tabela com as porcentagens de diferença entre os valores da ação, para assim conseguir ver melhor os outliers.
                           2. Rafael
                              Criar matriz de correlação - tabela com os pares de ações com correlaçãoa acima de 0,9.
                           3. Laura
                              Fazer as estatísticas descritivas de cada uma das ações.
                           4. Rafael
                              Olhar 1 a 1 esses pares e escolher as ações para ser retiradas, de acordo com as estatísticas descritivas.
                           5. Manu
                              Documentar outliers dos retornos 
                  1.3 pré-processamento e análise exploratória dos dados;
                        1.3.1 - Ígor
                              [x] Analisar a % de valores ausentes por coluna
                        1.3.2 - Ígor
                              [x] Definir e aplicar a estratégia de tratamento de missing values
                        1.3.3 - Rafael
                              Identificar e documentar os outliers encontrador
                        1.3.4 - Rafael
                              Decidir o que fazer com cada outlier (manter, recmover ou transformar)
                        1.3.5 - Arthur
                              [x] Verificar e remover duplicatas (Verificado no código: 0 duplicatas devido à ingestão nativa via DatetimeIndex do `yfinance`)
                        1.3.6 - Arthur
                              [x] Padronizar tipos de dados (datas, textos, números) (Validado no código: todos os preços padronizados como `float64`)
                  1.4 (28/05) entrega e apresentação
                        1.4.1 - Manuela & Laura
                              Preparar uma apresentação de alta qualidade explicando as estratégias utilizadas para o pre-processamento, a análise exploratória da base de dados a seleção de modelos, as reamostragens e os resultados das previsões/inferências com modelos lineares. A apresentação será feita em sala com utilização de PPT e outros recursos visuais que o grupo achar melhor. Cada grupo terá 20 minutos.
                        1.4.2 Laura
                              Essa apresentação deverá ser enviada no link “Entrega Apresentação 1” no Moodle. Cada grupo deve subir o notebook no repositório do grupo no GitHub. O endereço do repositório do GitHub deverá ser informado na entrega da tarefa no Moodle.
                  
            
      ~~2. Criar um modelo de Index Tracking (IT) do S&P100~~
            ~~2.2 Buscar os dados históricos dos últimos 7 anos dos índices S&P100~~
