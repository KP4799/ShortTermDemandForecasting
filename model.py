# Short-Term Demand Forecasting for Manufacturing Plants - IPython Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the file isn't found

# Convert 'Order Date' column to datetime objects
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

# Set 'Order Date' as index for time series operations
# Aggregate sales by 'Order Date' as there can be multiple orders on the same day
df = df.groupby('Order Date')['Sales'].sum().reset_index()
df = df.set_index('Order Date').sort_index()

# print("\nMissing values after aggregation:\n",df.isnull().sum())

def create_features(df):
    # Creates time series features from the date index.

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)

    # Adding lag features (e.g., sales from previous day, 7 days ago)
    # Be careful not to create features that leak future information.
    # We will use shifts based on the target variable itself.
    df['sales_lag_1'] = df['Sales'].shift(1)
    df['sales_lag_7'] = df['Sales'].shift(7) # Weekly seasonality

    # Adding rolling window features (e.g., 7-day rolling mean)
    df['sales_rolling_mean_7'] = df['Sales'].rolling(window=7).mean().shift(1) # Shift to prevent data leakage
    df['sales_rolling_std_7'] = df['Sales'].rolling(window=7).std().shift(1)   # Shift to prevent data leakage

    return df.copy() # Return a copy to avoid SettingWithCopyWarning

df_features = create_features(df.copy()) # Pass a copy to avoid modifying original df

# Drop rows with NaN values created by lag and rolling features
df_features.dropna(inplace=True)

# For time series, we split chronologically. Let's use the last few months as the test set
split_date = df_features.index.max() - pd.DateOffset(months=12) # last 12 months for testing

train_df = df_features[df_features.index < split_date]
test_df = df_features[df_features.index >= split_date]

# Define features (X) and target (y)
features = [col for col in df_features.columns if col != 'Sales']
target = 'Sales'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Number of features: {len(features)}\n")

# Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror', # For regression tasks
    n_estimators=600,             # Number of boosting rounds
    learning_rate=0.01,           # Step size shrinkage
    max_depth=4,                  # Maximum depth of a tree
    subsample=0.8,                # Subsample ratio of the training instance
    colsample_bytree=0.8,
    gamma=0.2,
    reg_lambda=1,
    reg_alpha=0.5,
    random_state=42,
    n_jobs=-1                     # Use all available cores
)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],verbose=False) # Set to True for detailed training output

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print("Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Create a DataFrame for actual vs. predicted values
forecast_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=y_test.index)

plt.figure(figsize=(18, 8))
plt.plot(forecast_df['Actual'], label='Actual Sales', color='blue')
plt.plot(forecast_df['Predicted'], label='Predicted Sales', color='red', linestyle='--')
plt.title('Actual vs. Predicted Sales (Test Set)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# Save the trained XGBoost model and its features
model_filename = 'xgboost_demand_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as '{model_filename}'")

features_filename = 'model_features.joblib'
joblib.dump(features, features_filename)
print(f"Features list saved as '{features_filename}'")

print("\nNotebook execution complete. Model and features saved.")
