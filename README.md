# Short-Term Demand Forecasting

This project implements an AI-driven short-term demand forecasting solution that enables businesses to accurately predict future demand for products using historical sales data and machine learning models.

## 🔎 Project Overview

Demand forecasting is the process of estimating future product demand based on past sales and related factors. Accurate forecasts help businesses optimise inventory, reduce stockouts, and improve planning across supply chains. :contentReference[oaicite:2]{index=2}

This repository includes:
- Data preprocessing and feature engineering
- Machine learning model training (e.g., XGBoost)
- A simple web or script-based app for inference
- Serialized model and feature objects for deployment

## 🧠 Approach

1. **Data Preparation:**  
   Historical sales data is cleaned, transformed, and engineered with relevant features such as date components and lag values.

2. **Model Training:**  
   A machine learning model (e.g., XGBoost) is trained to learn patterns in the historical data and forecast demand for upcoming periods.

3. **Inference Application:**  
   A Python script or lightweight app (`app.py`) loads the trained model and predicts demand based on new input data.

4. **Deployment:**  
   The model and preprocessing objects are saved using `joblib` (`xgboost_demand_model.joblib`, `model_features.joblib`) for fast loading and inference.

## 🛠️ Tech Stack

- **Python**  
- **XGBoost** (Extreme Gradient Boosting)  
- **Pandas, NumPy** (Data handling)  
- **joblib** (Model serialization)  
- **Machine Learning / Time Series Forecasting**
