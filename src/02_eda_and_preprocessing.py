"""
STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
STEP 3 — FEATURE ENGINEERING / PREPROCESSING
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "ghana_housing.csv"))

print("=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== AVG PRICE BY CITY ===")
print(df.groupby("city")["price_ghs"].mean().sort_values(ascending=False).round(0))

print("\n=== AVG PRICE BY PROPERTY TYPE ===")
print(df.groupby("property_type")["price_ghs"].mean().sort_values(ascending=False).round(0))

print("\n=== CORRELATION WITH PRICE (numeric features) ===")
numeric = df.select_dtypes(include=[np.number])
print(numeric.corr()["price_ghs"].sort_values(ascending=False).round(3))

# ---- Feature engineering ----
# 1. One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=["city", "property_type"], drop_first=True)

# 2. A couple of engineered features
df_encoded["price_per_sqm_proxy"] = df["size_sqm"]  # keep raw, model will learn interaction
df_encoded["total_rooms"] = df["bedrooms"] + df["bathrooms"]
df_encoded["is_far_from_cbd"] = (df["distance_to_cbd_km"] > 15).astype(int)

df_encoded.to_csv(os.path.join(DATA_DIR, "ghana_housing_processed.csv"), index=False)
print("\nProcessed shape:", df_encoded.shape)
print("Columns:", list(df_encoded.columns))
