import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess the data
print("Loading data...")
url = 'https://raw.githubusercontent.com/infinadox/Sales-Data-Forecasting/main/rossmann-store-sales/processed_train_data.csv'
data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
print("Data loaded successfully.")

# Select only the sales column and use a smaller subset for testing
data = data[['Sales']].head(10000)  # Example: use only the first 10,000 rows

# Normalize the data
print("Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print("Data normalized.")

# Define prev_data period
prev_data = 942  # Use past 942 days to predict

# Create features and target variables
print("Creating features and target variables...")
def create_features_target(data, prev_data):
    X, y = [], []
    for i in range(prev_data, len(data)):
        X.append(data[i-prev_data:i].flatten())
        y.append(data[i])
    return np.array(X), np.array(y).ravel()  # Flatten y to 1D array

X, y = create_features_target(scaled_data, prev_data)
print("Features and target variables created.")

# Split into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split successfully.")

# Initialize and train the Random Forest model with reduced complexity
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced number of trees
model.fit(X_train, y_train)
print("Model trained successfully.")

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_val)

# Calculate and save performance metrics
print("Calculating performance metrics...")
train_loss = mean_squared_error(y_train, model.predict(X_train))
val_loss = mean_squared_error(y_val, y_pred)

performance_txt = 'model_performance.txt'
with open(performance_txt, 'w') as f:
    f.write(f'Training Loss (MSE): {train_loss}\n')
    f.write(f'Validation Loss (MSE): {val_loss}\n')
print(f"Performance metrics saved to {performance_txt}.")

# Save training and validation loss to CSV
loss_history_df = pd.DataFrame({
    'Metric': ['Training Loss', 'Validation Loss'],
    'Value': [train_loss, val_loss]
})
loss_history_csv = 'training_validation_loss.csv'
loss_history_df.to_csv(loss_history_csv, index=False)
print(f"Training and validation loss saved to {loss_history_csv}.")

# Save the trained model
print("Saving the trained model...")
model_filename = 'sales_prediction_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}.")

# Make predictions on the entire dataset
print("Making full dataset predictions...")
full_pred = model.predict(X)

# Inverse transform to original scale
print("Inverse transforming predictions to original scale...")
full_pred_inv = scaler.inverse_transform(full_pred.reshape(-1, 1))
y_inv = scaler.inverse_transform(y.reshape(-1, 1))

# Save results (true vs predicted sales) to CSV
results_df = pd.DataFrame({
    'Date': data.index[prev_data:len(y_inv) + prev_data],
    'True Sales': y_inv.flatten(),
    'Predicted Sales': full_pred_inv.flatten()
})
results_csv = 'sales_predictions.csv'
results_df.to_csv(results_csv, index=False)
print(f"Sales predictions saved to {results_csv}.")

# Plot results
print("Plotting sales predictions...")
plt.figure(figsize=(14, 7))
plt.plot(data.index[prev_data:len(y_inv) + prev_data], y_inv, label='True Sales')
plt.plot(data.index[prev_data:len(y_inv) + prev_data], full_pred_inv, label='Predicted Sales', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Prediction')
plt.legend()
plt.savefig('sales_prediction.png')
plt.close()
print("Sales prediction plot saved as sales_prediction.png.")

# Forecast future sales
print("Forecasting future sales...")
forecast_horizon = 42
last_sequence = scaled_data[-prev_data:].flatten().reshape(1, -1)

forecast = []
for _ in range(forecast_horizon):
    pred = model.predict(last_sequence)
    forecast.append(pred[0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[0, -1] = pred[0]  # Extract single element from array

forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Create future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)

# Ensure lengths match
print(f"Length of future_dates: {len(future_dates)}")
print(f"Length of forecast_inv: {len(forecast_inv)}")

# Fix lengths if needed
if len(forecast_inv) > len(future_dates):
    forecast_inv = forecast_inv[:len(future_dates)]
elif len(forecast_inv) < len(future_dates):
    future_dates = future_dates[:len(forecast_inv)]

# Save forecast results to CSV
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast Sales': forecast_inv.flatten()
})
forecast_csv = 'sales_forecast.csv'
forecast_df.to_csv(forecast_csv, index=False)
print(f"Forecast results saved to {forecast_csv}.")

# Plot forecast results
print("Plotting sales forecast...")
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Sales'], label='Historical Sales')
plt.plot(future_dates, forecast_inv, label='Forecast Sales', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.savefig('sales_forecast.png')
plt.close()
print("Sales forecast plot saved as sales_forecast.png.")

print("All tasks completed successfully.")
