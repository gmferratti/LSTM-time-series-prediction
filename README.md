
# Tech Challenge 04 - Prevendo preço de ações com LSTM

## Índice
- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologia](#tecnologia)
- [Instalação Básica](#instalação-básica)
- [Etapa de Rodagem Local](#etapa-de-rodagem-local)
- [Etapa de Rodagem na Cloud](#etapa-de-rodagem-na-cloud)
- [Sobre o Modelo](#sobre-o-modelo)
- [Características do Modelo](#características-do-modelo)
- [Pipeline de Treinamento](#pipeline-de-treinamento)
- [Exemplo de Dados de Entrada](#exemplo-de-dados-de-entrada)
- [Funcionamento da API](#funcionamento-da-api)
- [Passo-a-passo do funcionamento](#passo-a-passo-do-funcionamento)
- [Colaborando](#colaborando)
- [Licença](#licença)
- [Integrantes do Projeto/ Contato](#integrantes-do-projeto-contato)

---

## Visão Geral

Este repositório contém o projeto desenvolvido para o quarto Tech Challenge do programa de pós-graduação em Machine Learning Engineering da FIAP, entregue em 03/12/2024.

O desafio consiste na criação de um modelo de Deep Learning utilizando a biblioteca PyTorch para prever os preços de fechamento das ações da Vivo (VIVT3.SA) na bolsa de valores. A implementação utiliza a biblioteca yfinance para a coleta de dados financeiros históricos e aplica redes neurais do tipo Long Short Term Memory (LSTM) para modelar as séries temporais históricas.

O projeto abrange todas as etapas do ciclo de vida de uma aplicação de Machine Learning, desde a coleta e pré-processamento de dados financeiros até o deploy do modelo em uma API RESTful hospedada no Google Cloud Platform, permitindo previsões de séries temporais em tempo real.

---

## Estrutura do Projeto

```plaintext
LSTM-time-series-prediction/
├── .flake8                     # Configuração para linting com flake8
├── .gitignore                  # Arquivo para ignorar arquivos no Git
├── makefile                    # Scripts automatizados para build e deploy
├── model.pth                   # Modelo treinado salvo
├── pyproject.toml              # Configuração de projeto (Poetry)
├── README.md                   # Documentação do projeto
├── requirements.dev.txt        # Dependências para ambiente de desenvolvimento
├── requirements.txt            # Dependências para produção
├── scaler.pkl                  # Escalador salvo (feature scaling)
├── setup.py                    # Script para empacotamento e instalação
│
├── Dockerfiles/                # Configurações de contêiner Docker
│   └── api/
│       └── Dockerfile          # Dockerfile para a API
│
├── examples/                   # Exemplos para uso da API
│   └── test_request.json       # Exemplo de requisição
│
├── notebooks/                  # Notebooks Jupyter para exploração
│
└── src/                        # Código-fonte do projeto
    ├── __init__.py             # Arquivo de inicialização do pacote
    ├── api/                    # Código da API RESTful
    │   ├── main.py             # Endpoint principal da API
    │   └── __init__.py         # Arquivo de inicialização do módulo
    └── model/                  # Modelos e lógica de treinamento
        └── train.py            # Script de treinamento do modelo
```

---

## Tecnologia

Este projeto foi desenvolvido utilizando **Python 3.10** e as seguintes libs complementares para garantir boa eficiência e escalabilidade:

**Coleta de Dados**  

- `yfinance`: Biblioteca utilizada para coletar dados históricos financeiros de ações.

**Armazenamento em Nuvem**  

- `google-cloud-storage`: Utilizada para gerenciar o armazenamento dos modelos e outros artefatos no Google Cloud.

**Manipulação e Análise de Dados**  

- `Pandas`: Utilizado para manipulação e análise de dados tabulares.  
- `Numpy`: Para cálculos matemáticos avançados e manipulação de arrays.  
- `Matplotlib`: Ferramenta para visualização gráfica dos dados.

**Machine Learning e Deep Learning**  

- `PyTorch`: Biblioteca poderosa para construção e treinamento do modelo LSTM (Obs.: usamos a versão light para deploy na API).  
- `Scikit-learn`: Ferramenta essencial para pré-processamento, avaliação de métricas e manipulação de dados.

**Frameworks para API e Deploy**  

- `FastAPI`: Framework para o desenvolvimento de APIs RESTful.  
- `Uvicorn`: Servidor ASGI leve e rápido para execução da API.

**Monitoramento e Gestão de Experimentos**  

- `MLflow`: Plataforma para rastreamento de experimentos e registro de modelos.  
- `Cloud Monitoring`: Solução integrada ao Google Cloud para monitorar métricas de desempenho da API e da infraestrutura.

**Outras Dependências**  

- `python-multipart`: Gerenciamento de uploads e formulários na API.
- `Pydantic`: Para validação e parsing de dados em APIs.  
- `Joblib`: Usado para serialização de objetos e paralelização de tarefas.

---

## Instalação Básica

1. Faça um fork do repositório: [LSTM-time-series-prediction](https://github.com/mauricioarauujo/LSTM-time-series-prediction)
2. Clone o repositório na sua máquina:

    ```bash
    git clone https://github.com/seu-usuario/LSTM-time-series-prediction.git
    cd LSTM-time-series-prediction
    ```

3. Crie e ative um ambiente virtual (virtualenv):

    - **No Windows**:

      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```

    - **No Linux/MacOS**:

      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

4. Configurar o ambiente no Windows:
    - Garanta que você consegue executar o comando `make`.
    - **Sugestão**: Utilize a biblioteca `py-make`:

      ```bash
      pip install py-make
      ```

5. Instale as dependências:

    ```bash
    make setup
    ```

    ou

    ```bash
    pip install -r requirements.txt
    ```

---

## Rodagem Local

1. Treine o modelo localmente:

    ```bash
    make train
    ```

2. Rode a interface do MLflow localmente para monitorar os experimentos:

    ```bash
    make mlflow-ui
    ```

    Acesse [http://localhost:5000](http://localhost:5000) no seu navegador.
3. Rode a API localmente para testes:

    ```bash
    make run-api-local
    ```

4. Em um outro terminal, envie uma requisição de teste para a API local:

    ```bash
    make test-local-api
    ```

---

## Deploy e Rodagem na Nuvem

1. Crie uma pasta `secrets` e insira suas credenciais do Google Cloud dentro do arquivo `keys.json`:

    ```bash
    mkdir secrets
    # Coloque seu arquivo keys.json dentro da pasta secrets
    ```

2. Crie um bucket no Google Cloud Storage (se ainda não existir):

    ```bash
    make create-bucket
    ```

3. Autentique-se no Google Cloud usando a service account:

    ```bash
    make auth-gcloud
    ```

4. Crie o repositório no Artifact Registry (se ainda não existir):

    ```bash
    make create-repo
    ```

5. Build e deploy da API no Cloud Run:

    ```bash
    make deploy-all
    ```

6. Teste a API em produção:

    ```bash
    make test-api
    ```

---

## Sobre o Modelo

O modelo utilizado neste projeto é uma **Rede Neural Long Short Term Memory (LSTM)** desenvolvida com a biblioteca **PyTorch**.
Ele foi projetado para prever os preços de fechamento das ações da **Vivo (VIVT3.SA)**, utilizando séries temporais como entrada.
Para mais informações do funcionamento específico de um LSTM veja este artigo do [Towards DS](https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf).

### Características do Modelo

- **Entrada**:  
  O modelo recebe como entrada sequências de séries temporais com um comprimento configurável (default: `10`). Os dados de entrada do treino incluem:
  - Preço de **abertura** (Open)
  - **Máximo** do dia (High)
  - **Mínimo** do dia (Low)
  - Preço de **fechamento** (Close)
  - **Volume** negociado (Volume)
  
  Contudo, para a predição, na prática, utilizamos somente:

  - Preço de **fechamento** (Close)
  - **Volume** negociado (Volume)

- **Camadas**:  
  - **Camadas LSTM**: Capturam as dependências temporais dos dados históricos.  
  - **Camadas totalmente conectadas**: Realizam a transformação final para a previsão do preço de fechamento.

- **Treinamento**:  
  - **Função de perda**: Mean Squared Error (MSE) para minimizar o erro quadrático médio entre os valores previstos e reais.  
  - **Otimizador**: Adam Optimizer com uma taxa de aprendizado de `0.001`.  
  - **Normalização**: Os dados de entrada foram normalizados utilizando `MinMaxScaler` para acelerar o processo de treinamento e melhorar a estabilidade do modelo.  
  - **Configuração do treinamento**:
    - Número de épocas: `100`
    - Tamanho do batch: `32`
    - Dispositivo: `cpu`

- **Avaliação**:  
  Durante o treinamento, o modelo é avaliado em um conjunto de validação separado. Métricas de desempenho incluem:  
  - **Mean Absolute Error (MAE)**: Média dos erros absolutos.  
  - **Root Mean Squared Error (RMSE)**: Raiz quadrada do erro médio quadrático.  
  - **Mean Absolute Percentage Error (MAPE)**: Média percentual dos erros absolutos.  

### Pipeline de Treinamento

1. **Coleta e Preparação dos Dados**:  
   - Os dados são extraídos do Yahoo Finance usando a biblioteca `yfinance`.  
   - Sequências de comprimento `10` são criadas a partir dos dados históricos.  
   - O conjunto de dados é dividido em partes de treinamento e validação.

2. **Treinamento do Modelo**:  
   - O modelo é treinado em lotes utilizando os dados de treinamento.  
   - Após cada época, o modelo é avaliado no conjunto de validação para monitorar o desempenho.  
   - Os resultados do treinamento e da validação são registrados no MLflow.

3. **Salvar Modelo e Scaler**:  
   Após o treinamento, o modelo é salvo no formato `.pth`, junto com o `MinMaxScaler` usado para normalizar os dados. Ambos são enviados para o **Google Cloud Storage** para uso futuro.

### Exemplo de Dados de Entrada

Os dados de entrada para o modelo seguem o formato:

| Date                | Symbol   | Open    | High    | Low     | Close   | Volume  |
|---------------------|----------|---------|---------|---------|---------|---------|
| 2019-11-18 00:00:00 | VIVT3.SA | 33.9284 | 34.6390 | 33.9284 | 34.6390 | 47100   |
| 2019-11-19 00:00:00 | VIVT3.SA | 34.5649 | 34.5649 | 33.6620 | 33.7952 | 114900  |
| 2019-11-21 00:00:00 | VIVT3.SA | 33.8692 | 33.9358 | 33.3585 | 33.7878 | 41200   |

O modelo processa essas séries temporais e prevê o preço de fechamento para os próximos dias.

---

## Funcionamento da API

A API desenvolvida com **FastAPI** permite prever o preço de fechamento das ações da **Vivo (VIVT3.SA)** com base em dados históricos de preços e volumes.

- **Endpoint Principal**: `POST /predict`
  - **Descrição**: Recebe dados de preços de fechamento e volumes dos últimos 10 dias e retorna a previsão do preço de fechamento para o próximo dia.
  - **Requisição**:
  
    ```json
    {
    "close_prices": [34.63, 33.79, 33.78, 34.01, 34.50, 34.95, 34.74, 34.88, 35.10, 34.82],
    "volumes": [47100, 114900, 41200, 35200, 48100, 53100, 49200, 50700, 49500, 51000]
    }
    ```

  - **Resposta**:
  
    ```json
    {
      "predicted_price": 35.12
    }
    ```

  - **Erros Possíveis**:
    - `400 Bad Request`: Se o número de pontos em `close_prices` ou `volumes` não for exatamente 10.
    - `500 Internal Server Error`: Em caso de erro interno durante a predição.

- **Endpoint de Verificação**: `GET /health`
  - **Descrição**: Verifica o status da API.
  - **Resposta**:
  
    ```json
    {
      "status": "healthy"
    }
    ```

- **Carregamento do Modelo**: Ao iniciar, a API carrega o modelo LSTM e o scaler do **Google Cloud Storage**.
- **Processamento da Requisição**:
  - Os dados recebidos são normalizados utilizando o scaler.
  - Os dados normalizados são convertidos em tensores e passados pelo modelo LSTM.
  - A saída do modelo é desnormalizada para obter o preço previsto em escala original.
- **Resposta**: O preço previsto é retornado em formato JSON.

---

## Colaborando

Contribuições com este repo são bem-vindas! Siga os passos abaixo para colaborar com este projeto:

1. **Faça um fork do repositório**.
2. **Crie uma branch para sua feature**:

    ```bash
    git checkout -b minha-feature
    ```

3. **Faça commits das suas alterações**:

    ```bash
    git commit -m "Descrição da feature"
    ```

4. **Envie as alterações para o repositório remoto**:

    ```bash
    git push origin minha-feature
    ```

5. **Abra um Pull Request** explicando suas mudanças.

---

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## Integrantes do Projeto/ Contato

- Luiz Claudio Santana Barbosa/xavecobarbosa@gmail.com

- Mauricio de Araujo Pintor/mauricio97_araujo@hotmail.com

- Antonio Eduardo de Oliveira Lima/devilre27@gmail.com

- Gustavo Mendonça Ferratti/gmferratti@gmail.com

- Rodolfo Olivieri Sivieri/rodolfo.olivieri3@gmail.com

---
