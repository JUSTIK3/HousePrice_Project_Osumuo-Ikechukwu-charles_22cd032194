import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1) Load Dataset
# -----------------------------
# Put train.csv in the project root (same level as app.py)
DATA_PATH = "../train.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2) Feature Selection (6 features)
# -----------------------------
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "Neighborhood"
]
TARGET = "SalePrice"

df = df[FEATURES + [TARGET]].dropna(subset=[TARGET])

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# 3) Preprocessing
#    a) missing values
#    b) encoding categorical
# -----------------------------
numeric_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath"]
categorical_features = ["Neighborhood"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 4) Model: Random Forest Regressor
# -----------------------------
model = RandomForestRegressor(
    n_estimators=250,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# 5) Train/Test Split + Training
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# -----------------------------
# 6) Evaluation Metrics
# -----------------------------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R^2  : {r2:.4f}")

# -----------------------------
# 7) Save Model (Joblib)
# -----------------------------
os.makedirs(".", exist_ok=True)
MODEL_PATH = "house_price_model.pkl"
joblib.dump(pipeline, MODEL_PATH)

print(f"\nSaved trained model to: model/{MODEL_PATH}")
