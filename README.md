# Projeto de Modelagem de Machine Learning para HR Analytics

Este repositório contém um pipeline completo de ciência de dados e machine learning criado para prever o turnover (Attrition) de funcionários da empresa TechCorp Brasil. O projeto foi desenvolvido usando Python, Pandas, Matplotlib, Seaborn, e diversos algoritmos avançados de Machine Learning.

## Estrutura do Projeto

```
.
├── notebooks
│   ├── 1_EDA_feature_eng.ipynb           # Análise exploratória e engenharia de features
│   ├── 2_modelagem_final.ipynb           # Notebook final com a melhor modelagem
│   ├── notebooks_analise
│   │   └── modelagem_desban.ipynb          # Testes preliminares
│   │   └── modelagem_lite copy.ipynb       # Testes preliminares
│   │   └── modelagem_lite.ipynb            # Testes preliminares
│   │   └── modelagem1.ipynb                # Testes preliminares
│   ├── data
│   │   ├── csv                         # Dados em CSV
│   │   ├── img                         # Visualizações dos resultados
│   │   └── pickle                      # Modelos salvos em formato pickle (Não vão estar aqui devido o tamanho dos .pickle)
│   └── utils                           # Funções auxiliares para gráficos e processamento
├── pyproject.toml                      # Configuração do Poetry
├── poetry.lock                         # Dependências geradas pelo Poetry
└── README.md                           # Este documento
```

## Contexto do Problema

A TechCorp Brasil enfrenta alta rotatividade de funcionários, causando prejuízos significativos estimados em R\$ 45 milhões anuais. Cada saída gera perdas em conhecimento institucional, impacto na produtividade e atrasos em projetos críticos.

## Objetivo

Criar um sistema preditivo capaz de identificar funcionários com alto risco de deixar a empresa, permitindo ações preventivas pelo RH.

## Dataset

O dataset é sintético, baseado no IBM HR Analytics, contendo 1 milhão de registros com 35 variáveis sobre funcionários, como idade, cargo, salário, satisfação, equilíbrio trabalho-vida pessoal, e histórico de desempenho.

## Pipeline do Projeto

### 1. Análise Exploratória e Feature Engineering (`1_EDA_feature_eng.ipynb`)

- Estatísticas descritivas e visualizações das variáveis.
- Análise detalhada das correlações numéricas e categóricas.
- Criação de mais de 10 novas features baseadas em hipóteses de negócio (ex.: burnout, estabilidade na empresa, interação entre cargo e horas extras).
- Insights relevantes para o negócio obtidos através de gráficos detalhados.

### 2. Pré-processamento e Encoding (`2_modelagem_final.ipynb`)

- Tratamento e encoding das variáveis categóricas usando OneHotEncoder.
- Divisão dos dados em treino e teste (80-20) com estratificação.

### 3. Tratamento do Desbalanceamento

- Técnicas de oversampling (SMOTE, ADASYN).
- Técnicas de undersampling (RandomUnderSampler).

### 4. Seleção e Normalização de Features

- Aplicação de Variance Threshold, SelectKBest e análise de correlação.
- Normalização das variáveis utilizando StandardScaler, Normalizer e MinMaxScaler conforme necessidade.

### 5. Modelagem e Otimização de Hiperparâmetros

- Implementação de diversos algoritmos: Logistic Regression, KNN, Decision Tree, Random Forest, e XGBoost.
- Otimização usando métodos avançados como GridSearchCV, RandomizedSearchCV, BayesSearchCV e Optuna.

### 6. Avaliação e Interpretação dos Modelos

- Métricas específicas para dados desbalanceados: Precision, Recall, F1-Score, ROC-AUC.
- Técnicas interpretativas avançadas como SHAP e Permutation Importance.

## Visualizações e Resultados

Gráficos detalhados estão disponíveis na pasta `data/img`, ilustrando insights exploratórios, performance dos modelos e importância das features.

## Modelos Salvos

Os modelos treinados estão organizados na pasta `data/pickle`, categorizados por método de balanceamento, técnica de normalização e otimização utilizada.
Não estarão salvos os modelos devido o tamanho dos arquivos

## Instalação

Utilize Poetry para instalar as dependências do projeto:

```bash
poetry install
```

ou se preferir use
```bash
pip install -r requirements.txt
```

## Execução

Abra os notebooks usando Jupyter Notebook

## Próximos Passos

- Implementação de um sistema de deployment (API REST ou dashboard interativo).
- Melhoria nos cortes
- Ajustes em funções para deixar mais generico
- Colocar mais casos de uso
- Monitoramento contínuo e ajustes finos nos modelos em produção.

---

**Autores**: Hermes, Lucio, Nathalia, Vitor
