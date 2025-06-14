from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("arrivals_soe.csv")
df['date'] = pd.to_datetime(df['date'])

# Preprocess the data (similar to your previous code)
# Filter and prepare the data as needed for the SARIMA model
# This should include the same preprocessing steps you used before

# Example preprocessing (you may need to adjust this based on your actual data)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['log_arrivals'] = np.log1p(df['arrivals'])

# Define the best SARIMA parameters (from your tuning process)
best_order = (1, 1, 1)  # Example values, replace with your tuned values
best_seasonal_order = (1, 1, 1, 12)  # Example values, replace with your tuned values

# Fit the SARIMA model on the training data
train_df = df[df['date'] < '2024-01-01']
model = SARIMAX(train_df['log_arrivals'], order=best_order, seasonal_order=best_seasonal_order)
model_fit = model.fit(disp=False)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_month = data['selected_month']

    # Prepare the input for prediction
    # You may need to create a new DataFrame for the prediction
    # For example, you can create a date range for the next months
    future_dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    future_df = pd.DataFrame({'date': future_dates})
    future_df['month'] = future_df['date'].dt.month
    future_df['year'] = future_df['date'].dt.year
    future_df['month_count'] = (future_df['year'] - train_df['year'].min()) * 12 + future_df['month']
    
    # Add cyclic encoding for month
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)

    # Prepare the features for prediction
    # You may need to adjust this based on your model's requirements
    future_df['country_enc'] = 0  # Placeholder, replace with actual encoding if needed
    future_df['state_enc'] = 0    # Placeholder, replace with actual encoding if needed
    future_df['lag1_log'] = model_fit.predict(start=len(train_df), end=len(train_df) + 11)
    future_df['lag2_log'] = future_df['lag1_log'].shift(1)
    future_df['lag3_log'] = future_df['lag1_log'].shift(2)

    # Drop NaN values
    future_df = future_df.dropna()

    # Make predictions
    forecast = model_fit.get_forecast(steps=12)
    pred = forecast.predicted_mean

    # Prepare the response
    response = {
        'month': selected_month,
        'prediction': np.expm1(pred.values).tolist()  # Inverse log transformation
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
