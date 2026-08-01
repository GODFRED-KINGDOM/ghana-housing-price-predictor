# Ghana Housing Price Prediction — End-to-End Pipeline

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ghana-housing-predictor.streamlit.app/)

## ⚠️ Important note on data

No live, clean public dataset of Ghana housing prices was accessible in this environment.
To demonstrate the **full pipeline** (not just modeling), a realistic synthetic dataset (3,000 listings)
was generated using known Ghana real-estate patterns: Accra/Tema price premiums, gated-community
and pool premiums, distance-to-CBD depreciation, property age depreciation, etc.
**Before using this in production, `01_generate_data.py` must be replaced with a real data source**
(e.g. scraped/licensed data from meQasa, Jiji Ghana, or Ghana Statistical Service records).
The pipeline architecture (EDA → features → training → evaluation → deployment) stays identical either way.

## Pipeline stages

|File|Stage|What it does|
|-|-|-|
|`src/01_generate_data.py`|Data Collection|Builds `data/ghana_housing.csv` (3,000 rows, 14 columns)|
|`src/02_eda_and_preprocessing.py`|EDA + Feature Engineering|Checks nulls, price by city/type, correlations, one-hot encoding, engineered features|
|`src/03_train_models.py`|Modeling + Evaluation|Trains 4 models, compares MAE/RMSE/R²/MAPE, saves the best one|
|`src/04_app.py`|Deployment|Flask REST API (`/predict`, `/health`) serving live predictions|

## Results (on held-out 20% test set)

|Model|MAE (GHS)|RMSE (GHS)|R²|MAPE|
|-|-|-|-|-|
|**Gradient Boosting (chosen)**|139,624|194,359|**0.891**|12.8%|
|Ridge Regression|158,857|210,611|0.872|17.3%|
|Linear Regression|158,878|210,620|0.872|17.3%|
|Random Forest|159,886|219,165|0.862|15.7%|

**Gradient Boosting** was selected because it gave the best accuracy **(highest R²)** and the lowest prediction errors, making it the strongest model for housing price prediction.

## Top price drivers found by the model

- Plot size
- Floor size
- City (Tamale/Cape Coast/Kumasi vs. Accra)
- Distance to CBD
- Property age
- Property type

## Running it

```bash
pip install pandas numpy scikit-learn joblib flask
python3 src/01_generate_data.py
python3 src/02_eda_and_preprocessing.py
python3 src/03_train_models.py
python3 src/04_app.py          # starts API at http://localhost:5000
```

### Example API call

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "city": "Accra", "property_type": "Detached House", "bedrooms": 4, "bathrooms": 4,
  "size_sqm": 280, "plot_size_sqm": 450, "age_years": 3, "distance_to_cbd_km": 8,
  "gated_community": 1, "has_pool": 1, "has_garage": 1, "has_backup_power": 1, "has_borehole": 1
}'
```

Response:

```json
{
  "predicted_price_ghs": 2901053.62,
  "model_used": "Gradient Boosting",
  "note": "Model explains ~89.1% of price variation on held-out test data; typical error is about GHS 139,624."
}
```

## Streamlit demo

A simple point-and-click UI is included (`streamlit_app.py`) — type in house details and get an
instant price estimate, no API calls needed.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This is the easiest way to demo the model to a non-technical audience (e.g. a supervisor or client)
and can be deployed for free on Streamlit Community Cloud or Hugging Face Spaces by pointing them
at this repo.

## Known limitation

Tree-based models can extrapolate outside the training distribution (e.g. very cheap, small,
far-out properties), occasionally producing unrealistic values. A floor of GHS 80,000 is
applied in `04_app.py` as a safeguard — this should be revisited with real data and wider
coverage of the low end of the market.
