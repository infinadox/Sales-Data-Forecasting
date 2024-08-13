import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
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

# Initialize and train the Random Forest model with regularization
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=50,          # Reduced number of trees
    max_depth=10,             # Limit the maximum depth of the trees
    min_samples_split=10,     # Increase the minimum number of samples required to split a node
    min_samples_leaf=5,       # Increase the minimum number of samples required to be at a leaf node
    max_features='sqrt',      # Consider sqrt(number of features) when looking for the best split
    random_state=42
)
model.fit(X_train, y_train)
print("Model trained successfully.")

# Cross-validation to assess the model
print("Performing cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f"Cross-validated MSE: {cv_mse}")

# Make predictions and calculate performance metrics
print("Making predictions...")
y_pred = model.predict(X_val)
train_loss = mean_squared_error(y_train, model.predict(X_train))
val_loss = mean_squared_error(y_val, y_pred)

# Save performance metrics
performance_txt = 'model_performance.txt'
with open(performance_txt, 'w') as f:
    f.write(f'Training Loss (MSE): {train_loss}\n')
    f.write(f'Validation Loss (MSE): {val_loss}\n')
    f.write(f'Cross-validated MSE: {cv_mse}\n')
print(f"Performance metrics saved to {performance_txt}.")

# Save training and validation loss to CSV
loss_history_df = pd.DataFrame({
    'Metric': ['Training Loss', 'Validation Loss', 'Cross-validated MSE'],
    'Value': [train_loss, val_loss, cv_mse]
})
loss_history_csv = 'training_validation_loss.csv'
loss_history_df.to_csv(loss_history_csv, index=False)
print(f"Training and validation loss saved to {loss_history_csv}.")

# Save the trained model
print("Saving the trained model...")
model_filename = 'sales_prediction_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}.")

# Forecast sales for 42 days after 2015-07-31
print("Forecasting future sales after 2015-07-31...")
last_sequence = scaled_data[-prev_data:].flatten().reshape(1, -1)
specific_last_date = pd.Timestamp('2015-07-31')
future_dates, forecast_inv = forecast_sales(model, scaler, last_sequence, specific_last_date)

# Save forecast results to CSV
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast Sales': forecast_inv.flatten()
})
forecast_csv = 'sales_forecast_after_2015-07-31.csv'
forecast_df.to_csv(forecast_csv, index=False)
print(f"Forecast results saved to {forecast_csv}.")

# Plot forecast results
print("Plotting sales forecast after 2015-07-31...")
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Sales'], label='Historical Sales')
plt.plot(future_dates, forecast_inv, label='Forecast Sales', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('42-Day Sales Forecast After 2015-07-31')
plt.legend()
plt.savefig('sales_forecast_after_2015-07-31.png')
plt.close()
print("Sales forecast plot saved as sales_forecast_after_2015-07-31.png.")

print("All tasks completed successfully.")
