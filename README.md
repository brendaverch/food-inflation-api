# 📊 Food Inflation ML API - Docker

Este projeto contém um **modelo de Machine Learning** para previsão da inflação de alimentos, exposto como uma **API Flask**. Ele foi empacotado em um **container Docker** para garantir fácil execução e portabilidade.

---

## 📌 1. Descrição do Projeto

O objetivo é analisar e prever a inflação de alimentos utilizando um dataset do Kaggle e um modelo de regressão Random Forest.

Este modelo foi transformado em um **serviço de predição online via API**, permitindo que aplicações externas enviem dados e recebam previsões em tempo real.

---

## 🛠️ 2. Tecnologias Utilizadas

| Tecnologia        | Descrição                                      |
|------------------|----------------------------------------------|
| **Python 3.9**   | Linguagem principal para análise e modelagem |
| **Docker**       | Empacotamento e execução do ambiente isolado |
| **Flask**        | Framework leve para criação da API           |
| **Scikit-Learn** | Treinamento do modelo Random Forest          |
| **Pandas / NumPy** | Manipulação de dados                        |
| **Joblib**       | Serialização do modelo treinado              |

---

## 📂 3. Estrutura do Projeto

```bash
.
├── app/
│   ├── main.py              # Código da API Flask
│   └── model_utils.py       # Funções auxiliares (carregar modelo, preparar dados)
├── model/
│   ├── train_model.py       # Script de treinamento
│   ├── modelo_rf.pkl        # Modelo treinado
│   └── input_columns.pkl    # Colunas esperadas pelo modelo
├── data/
│   └── CPI_dataset.csv      # Dataset base
├── requirements.txt         # Dependências do projeto
├── Dockerfile               # Arquivo para build da imagem Docker


```

## 📈 Registro de métricas com MLflow

O script `model/train_model.py` registra hiperparâmetros e métricas de
avaliação usando o **MLflow**. Para executar o treinamento e iniciar a
interface de visualização dos experimentos, utilize:

```bash
python model/train_model.py
mlflow ui
```

Isso cria a pasta `mlruns/` com os resultados e permite comparar as
execuções para escolher o melhor modelo.

## 🚀 **4. Como Rodar o Docker**

### Construir a imagem Docker

``` bash
docker build -t food-inflation-api .

```
### Rodar o container

``` bash
docker run -p 5000:5000 food-inflation-api
```

---
## **5. Como Usar a API**

🔁 Endpoint: POST /predict

URL: http://localhost:5000/predict

Método: POST

Content-Type: application/json

Body de exemplo:

{
  "consumer_price_index_item": "Eggs",
  "month_of_forecast": "March",
  "attribute": "Mid point of prediction interval"
}

✅ Exemplo de resposta:

{
  "previsao_forecast_percent_change": 1.37
}

🧪 Testar com curl:

``` bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"consumer_price_index_item": "Eggs", "month_of_forecast": "March", "attribute": "Mid point of prediction interval"}'

```
