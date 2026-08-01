"""
STEP 4 — MODEL TRAINING
STEP 5 — MODEL EVALUATION & SELECTION
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "ghana_housing_processed.csv"))

X = df.drop(columns=["price_ghs"])
y = df["price_ghs"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": (LinearRegression(), True),
    "Ridge Regression": (Ridge(alpha=1.0), True),
    "Random Forest": (RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1), False),
    "Gradient Boosting": (GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42), False),
}

results = []
fitted_models = {}

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    results.append([name, mae, rmse, r2, mape])
    fitted_models[name] = model
    print(f"{name:20s} | MAE: GHS {mae:,.0f} | RMSE: GHS {rmse:,.0f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2", "MAPE"])
results_df = results_df.sort_values("R2", ascending=False)
print("\n=== MODEL COMPARISON (sorted by R2) ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print(f"\nBEST MODEL: {best_model_name}")

# Feature importance if available
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n=== TOP 10 FEATURE IMPORTANCES ===")
    print(importances.head(10))

# Save best model + scaler + feature list + metadata for deployment
joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "needs_scaling": models[best_model_name][1],
    "feature_names": feature_names,
    "model_name": best_model_name,
    "test_r2": float(results_df.iloc[0]["R2"]),
    "test_mae": float(results_df.iloc[0]["MAE"]),
}, os.path.join(MODELS_DIR, "ghana_housing_model.pkl"))

results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
print("\nSaved best model to models/ghana_housing_model.pkl")
