import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the datasets
store_data_url = "https://raw.githubusercontent.com/infinadox/Sales-Data-Forecasting/main/rossmann-store-sales/store.csv"
test_data_url = "https://raw.githubusercontent.com/infinadox/Sales-Data-Forecasting/main/rossmann-store-sales/test.csv"
train_data_url = "https://raw.githubusercontent.com/infinadox/Sales-Data-Forecasting/main/rossmann-store-sales/train.csv"

store_data = pd.read_csv(store_data_url)
test_data = pd.read_csv(test_data_url)
train_data = pd.read_csv(train_data_url)

# Display the first few rows of each dataframe
print("Store Data:")
print(store_data.head())
print("\nTest Data:")
print(test_data.head())
print("\nTrain Data:")
print(train_data.head())

# Merge train and test datasets with store data
train_data = pd.merge(train_data, store_data, on='Store')
test_data = pd.merge(test_data, store_data, on='Store')

# Display the first few rows after merging
print("\nMerged Train Data:")
print(train_data.head())
print("\nMerged Test Data:")
print(test_data.head())

# Handle missing values
# Fill missing values in 'CompetitionDistance' with the median value
train_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].median(), inplace=True)
test_data['CompetitionDistance'].fillna(test_data['CompetitionDistance'].median(), inplace=True)

# Fill missing values in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' with mode values
train_data['CompetitionOpenSinceMonth'].fillna(train_data['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
train_data['CompetitionOpenSinceYear'].fillna(train_data['CompetitionOpenSinceYear'].mode()[0], inplace=True)
test_data['CompetitionOpenSinceMonth'].fillna(test_data['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
test_data['CompetitionOpenSinceYear'].fillna(test_data['CompetitionOpenSinceYear'].mode()[0], inplace=True)

# Fill missing values in 'Promo2SinceWeek' and 'Promo2SinceYear' with mode values
train_data['Promo2SinceWeek'].fillna(train_data['Promo2SinceWeek'].mode()[0], inplace=True)
train_data['Promo2SinceYear'].fillna(train_data['Promo2SinceYear'].mode()[0], inplace=True)
test_data['Promo2SinceWeek'].fillna(test_data['Promo2SinceWeek'].mode()[0], inplace=True)
test_data['Promo2SinceYear'].fillna(test_data['Promo2SinceYear'].mode()[0], inplace=True)

# Normalize numerical features
scaler = StandardScaler()
train_data[['CompetitionDistance']] = scaler.fit_transform(train_data[['CompetitionDistance']])
test_data[['CompetitionDistance']] = scaler.transform(test_data[['CompetitionDistance']])

# Convert 'Date' to datetime format and extract new features
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['Day'] = train_data['Date'].dt.day
train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek

test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['Year'] = test_data['Date'].dt.year
test_data['Month'] = test_data['Date'].dt.month
test_data['Day'] = test_data['Date'].dt.day
test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek

# Display the first few rows after preprocessing
print("\nPreprocessed Train Data:")
print(train_data.head())
print("\nPreprocessed Test Data:")
print(test_data.head())

# Save the preprocessed data to CSV files (optional)
train_data.to_csv('/content/processed_train_data.csv', index=False)
test_data.to_csv('/content/processed_test_data.csv', index=False)

print("Data preprocessing complete. Files saved as 'processed_train_data.csv' and 'processed_test_data.csv'.")