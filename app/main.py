# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

# Caminho base do projeto (dois niveis acima deste arquivo)
BASE_DIR = Path(__file__).resolve().parent.parent

# Carrega o modelo e as colunas usadas no treino
model = joblib.load(BASE_DIR / "model" / "modelo_rf.pkl")
input_columns = joblib.load(BASE_DIR / "model" / "input_columns.pkl")

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"mensagem": "API de Previsão de Inflação Alimentar está rodando!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Converte para DataFrame
        input_df = pd.DataFrame([data])

        # Garante que todas as colunas estejam presentes, com 0 onde faltar
        input_df = pd.get_dummies(input_df)
        for col in input_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[input_columns]  # Garante a ordem das colunas

        # Previsão
        prediction = model.predict(input_df)[0]

        return jsonify({"previsao_forecast_percent_change": round(prediction, 2)})

    except Exception as e:
        return jsonify({"erro": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
