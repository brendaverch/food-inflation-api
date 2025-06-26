import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

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
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 5) Treinamento
n_estimators = 100
model = RandomForestRegressor(
    n_estimators=n_estimators, random_state=random_state, n_jobs=-1
)

mlflow.set_experiment("food-inflation-rf")
with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)

    model.fit(X_train, y_train)

    # Avaliação
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # 6) Salva modelo e colunas de entrada
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/modelo_rf.pkl")
    joblib.dump(X.columns.tolist(), "model/input_columns.pkl")
    mlflow.sklearn.log_model(model, "rf_model")
    mlflow.log_artifact("model/input_columns.pkl")

print("Modelo salvo com sucesso em /model e registrado no MLflow")
