# 🏭 Short-Term Demand Forecasting for Manufacturing Plants

This project implements a machine learning–based system to forecast short-term product demand for manufacturing plants using historical sales data. The system uses an XGBoost regression model trained on time-series features and is deployed through an interactive Streamlit web application for real-time predictions.

The app supports both single-day and date-range demand forecasting to assist production planning and inventory management.

## 🎯 Problem Statement

Accurate demand forecasting is critical for manufacturing operations to reduce inventory costs, avoid stock-outs, and optimize production scheduling. Traditional forecasting methods often fail to capture complex seasonal and temporal patterns.

This project applies machine learning with engineered time-series features to improve short-term demand prediction accuracy.

##🧠 Approach

### 🔹 Data Processing

- Historical sales data loaded from train.csv
- Aggregated daily sales (multiple orders per day combined)
- Date converted to datetime index and sorted chronologically

### 🔹 Feature Engineering

Time-based features:
   - Year, month, day, week of year, day of week
   - Quarter indicators
   - Month/quarter/year start and end flags

Lag & rolling features:
   - Previous day sales (lag-1)
   - Sales from 7 days ago (lag-7)
   - 7-day rolling mean and standard deviation (shifted to prevent leakage)

### 🔹 Model Training

- Algorithm: XGBoost Regressor
- Objective: Squared error regression
- Chronological train-test split (last 12 months as test set)
- Evaluation metrics:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)

### 🔹 Model Persistence
- Trained model saved using joblib
- Feature list saved separately to ensure consistent inference

## 🌐 Streamlit Web Application

The Streamlit app allows users to:
- Predict demand for a single future date
- orecast demand across a date range
- Visualize predicted sales trends
- Automatically generate time-series features for inference

Lag and rolling features are computed using recent historical data from train.csv to maintain consistency with training conditions.

## 🛠️ Tech Stack

- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn (metrics)
- Joblib
- Matplotlib
- Streamlit

## 📁 Project Structure
ShortTermDemandForecasting/
│
├── train.csv
├── model_training.ipynb / model.py
├── xgboost_demand_model.joblib
├── model_features.joblib
├── app.py (Streamlit app)
└── README.md

