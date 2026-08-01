"""
STEP 1 — DATA COLLECTION (SIMULATED)
=====================================
In a real project, this data would come from sources like meQasa, Jiji Ghana,
Ghana Statistical Service, or a real estate agency's internal database.

No clean public dataset of Ghana housing prices is readily accessible here,
so we SIMULATE a realistic dataset using known Ghana real-estate patterns:
- Accra (esp. East Legon, Airport Residential) and Tema command premiums
- Kumasi, Takoradi, Cape Coast are mid-tier
- Gated communities, pools, garages add value
- Distance from CBD reduces value
- Property age slightly reduces value

This keeps the FULL PIPELINE realistic even without live data access.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
N = 3000

cities = ["Accra", "Kumasi", "Tema", "Takoradi", "Cape Coast", "Tamale"]
city_base_price_per_sqm = {   # GHS per sqm, rough relative tiers
    "Accra": 9500,
    "Tema": 7200,
    "Kumasi": 5800,
    "Takoradi": 5200,
    "Cape Coast": 4300,
    "Tamale": 3500,
}

property_types = ["Apartment", "Detached House", "Townhouse", "Semi-Detached"]
type_multiplier = {"Apartment": 0.9, "Semi-Detached": 1.0, "Townhouse": 1.15, "Detached House": 1.3}

rows = []
for _ in range(N):
    city = np.random.choice(cities, p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08])
    ptype = np.random.choice(property_types, p=[0.4, 0.3, 0.2, 0.1])
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.2, 0.35, 0.25, 0.1, 0.05])
    bathrooms = max(1, bedrooms - np.random.choice([0, 1], p=[0.6, 0.4]))
    size_sqm = np.random.normal(70 + bedrooms * 35, 25)
    size_sqm = max(35, size_sqm)
    plot_size_sqm = size_sqm * np.random.uniform(1.2, 2.5) if ptype != "Apartment" else 0
    age_years = np.random.randint(0, 30)
    distance_to_cbd_km = np.random.exponential(8) + 0.5
    gated_community = np.random.choice([0, 1], p=[0.55, 0.45])
    has_pool = np.random.choice([0, 1], p=[0.85, 0.15]) if ptype in ["Detached House", "Townhouse"] else 0
    has_garage = np.random.choice([0, 1], p=[0.4, 0.6]) if ptype != "Apartment" else np.random.choice([0, 1], p=[0.7, 0.3])
    has_backup_power = np.random.choice([0, 1], p=[0.6, 0.4])  # common concern in Ghana (dumsor)
    has_borehole = np.random.choice([0, 1], p=[0.5, 0.5])      # water reliability

    base = city_base_price_per_sqm[city] * size_sqm * type_multiplier[ptype]
    price = base
    price *= (1 - 0.012 * age_years)                      # depreciation with age
    price *= (1 - 0.018 * min(distance_to_cbd_km, 25))    # farther from CBD = cheaper
    price *= (1 + 0.12 * gated_community)
    price *= (1 + 0.10 * has_pool)
    price *= (1 + 0.05 * has_garage)
    price *= (1 + 0.06 * has_backup_power)
    price *= (1 + 0.04 * has_borehole)
    price += plot_size_sqm * 150   # land value component

    # add realistic noise (market isn't perfectly rational)
    price *= np.random.normal(1, 0.12)
    price = max(80000, price)  # floor price

    rows.append([
        city, ptype, bedrooms, bathrooms, round(size_sqm, 1), round(plot_size_sqm, 1),
        age_years, round(distance_to_cbd_km, 2), gated_community, has_pool,
        has_garage, has_backup_power, has_borehole, round(price, 2)
    ])

df = pd.DataFrame(rows, columns=[
    "city", "property_type", "bedrooms", "bathrooms", "size_sqm", "plot_size_sqm",
    "age_years", "distance_to_cbd_km", "gated_community", "has_pool",
    "has_garage", "has_backup_power", "has_borehole", "price_ghs"
])

df.to_csv(os.path.join(DATA_DIR, "ghana_housing.csv"), index=False)
print(df.shape)
print(df.head())
print(df["price_ghs"].describe())
