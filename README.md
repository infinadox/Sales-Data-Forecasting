
# Sales Data Forecasting for Rossmann Stores

## Project Overview
This project aims to develop a time series forecasting model to predict daily sales for Rossmann stores. By utilizing historical sales data and various store attributes, we seek to create an accurate predictive model that supports business planning and inventory management.

**Project Team:** Kevin Tran, Shameer Razaak, and Juan Mantilla  
**Dataset Source:** Kaggle Rossmann Store Sales Dataset

## Objectives
- Retrieve and process sales data from the Rossmann Stores dataset.
- Conduct Exploratory Data Analysis (EDA) to uncover sales trends and patterns.
- Engineer features to enhance the model's predictive accuracy.
- Explore advanced techniques, such as Random Forest, to potentially improve performance.
- Develop and optimize a time series forecasting model with a target R-squared score of at least 0.80.
- Document the entire process, from data preprocessing to model evaluation.
- Present findings clearly and engagingly.

## Exploratory Data Analysis (EDA)
EDA focuses on understanding the sales data through various aspects:
- **Sales vs Customers:** Analyzed the relationship between sales and the number of customers.
- **Sales per Day:** Evaluated daily sales trends.
- **Customers per Day:** Investigated daily customer patterns.
- **Sales per Customer:** Assessed average sales per customer.
- **Average Sales per Week:** Reviewed weekly sales trends.
- **Weekday Sales:** Analyzed sales trends by day of the week.
- **Sales Overtime:** Examined sales over 6 weeks.
- **Locality:** Studied the impact of store locality on sales.
- **Seasonality:** Identified seasonal patterns in the sales data.
- **Promotions:** Evaluated the effect of promotions on sales.

## Model Training
**Dependencies:**
Ensure required libraries are installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

**Running the Script:**
Save the code in a Python script file (e.g., `Random_forest.py`) and run:
```bash
python Random_forest.py
```

**Understanding the Code:**
- **Data Loading and Preprocessing:** Normalizes the data.
- **Feature Creation:** Generates lag features for the Random Forest model.
- **Training and Evaluation:** Trains the model and assesses performance.
- **Plotting Results:** Visualizes predictions and saves plots.
- **Forecasting:** Produces future sales forecasts and saves results.

## Random Forest Overview
**What is Random Forest?**
Random Forest is a powerful machine-learning algorithm that combines multiple decision trees to enhance predictive accuracy. It is effective for both classification and regression tasks.

- **Decision Trees:** Make decisions based on a series of questions.
- **Ensemble Methods:** Uses bagging and feature randomness to build an ensemble of uncorrelated decision trees.

**How It Works:**
Involves key hyperparameters like node size, number of trees, and number of features sampled. Predictions are made through averaging (for regression) or majority voting (for classification).

**Benefits:**
- Reduces overfitting.
- Flexible for various tasks.
- Provides insights into feature importance.

**Challenges:**
- Time-consuming and resource-intensive.
- Complex interpretation compared to single decision trees.

**Applications:**
Used in finance, healthcare, and e-commerce for tasks such as credit risk evaluation, gene classification, and recommendation engines.

## First Iteration
- **Training Loss (MSE):** 0.0004147702751754034
- **Validation Loss (MSE):** 0.0028658269385234197

**Interpretation:**
- The Training Loss is low, indicating good performance on the training data.
- The Validation Loss is higher, suggesting potential overfitting. The model performs well on training data but may not generalize as well to new data.
- 
## Second Iteration
- **Training Loss (MSE)**: 0.0033199530559485
- **Validation Loss (MSE)**: 0.006474465520687854
- **Cross-validated MSE**: 0.006220120581014061

**Recommendations:**
- Apply regularization techniques (e.g., L1/L2 regularization) to reduce overfitting.

## Summary of Updates
**Model Regularization:**
Improved generalization by adding constraints:
- `max_depth=10`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `max_features='sqrt'`

**Cross-Validation:**
Enhanced evaluation using cross-validation:
- Added `cross_val_score` to assess performance more robustly.
- Calculated and saved cross-validated MSE.

**Forecasting:**
- Added functionality to forecast future sales for the next 42 days:
  - `forecast_sales` function predicts future values and returns the forecasted values and dates.
  - Included forecasts for the next 42 days and starting from 2015-07-31.

**Plotting:**
- Plotted historical and forecasted sales for:
  - The next 42 days.
  - 42 days after 2015-07-31.

**Saving Results:**
- Saved forecast results to CSV files and plots to PNG files:
  - `sales_forecast.csv`
  - `sales_forecast_after_2015-07-31.csv`

## EDA Graphs
- **Total Forecast Sales by Day**
- **Average Forecast Sales per Week**
- **42 Day Sales Forecast**

## Conclusions
- Store type B shows the highest sales and customer traffic.
- Sales strongly correlate with the number of customers.
- Promotions boost sales and customer counts.
- Sales are higher during school holidays compared to other holidays.
- Increased sales during Christmas week.
- Missing values in certain features do not necessarily indicate a lack of competition.
- Slight seasonality observed in sales data.
- Saturdays are popular shopping days.

---

Feel free to adjust any part according to your needs!
