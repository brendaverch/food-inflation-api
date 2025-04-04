import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

# 1) Leitura e limpeza
df = pd.read_csv("data/CPI_dataset.csv") 
df.dropna(subset=["forecast_percent_change"], inplace=True)

# 2) One-hot encoding
categorical_cols = ["consumer_price_index_item", "month_of_forecast", "attribute"]
df = pd.get_dummies(df, columns=categorical_cols)

# 3) Separação X e y
X = df.drop(columns=["forecast_percent_change"])
y = df["forecast_percent_change"]

# 4) Treino/teste split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5) Treinamento
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6) Salva modelo e colunas de entrada
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/modelo_rf.pkl")
joblib.dump(X.columns.tolist(), "model/input_columns.pkl")

print("Modelo salvo com sucesso em /model")
