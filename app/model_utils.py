import joblib
import pandas as pd

def carregar_modelo(path_modelo="model/modelo_rf.pkl", path_colunas="model/input_columns.pkl"):
    modelo = joblib.load(path_modelo)
    colunas = joblib.load(path_colunas)
    return modelo, colunas

def preparar_dados(input_json, colunas_esperadas):
    input_df = pd.DataFrame([input_json])
    input_df = pd.get_dummies(input_df)

    for col in colunas_esperadas:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[colunas_esperadas]  
    return input_df
