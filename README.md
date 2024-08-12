# Sales Data Forecasting for Rossmann Stores

## Project Overview

This project aims to develop a time series forecasting model to predict daily sales for Rossmann stores. By utilizing historical sales data and various store attributes, we seek to create an accurate predictive model that supports business planning and inventory management.

**Project Team:** Kevin Tran, Shameer Razaak, and Juan Mantilla

**Dataset Source:** [Kaggle Rossmann Store Sales Dataset](https://www.kaggle.com/c/rossmann-store-sales/data)

## Objectives

- Retrieve and process sales data from the Rossmann Stores dataset.
- Conduct Exploratory Data Analysis (EDA) to uncover sales trends and patterns.
- Engineer features to enhance the model's predictive accuracy.
- Explore advanced techniques, such as Random Forest to potentially improve performance.
- Develop and optimize a time series forecasting model with a target R-squared score of at least 0.80.
- Document the entire process, from data preprocessing to model evaluation.
- Present findings clearly and engagingly.

## Exploratory Data Analysis (EDA)

EDA focuses on understanding the sales data through various aspects:

- **Sales vs Customers:** Analyzed the relationship between sales and the number of customers.
- **Sales per Day:** Evaluated sales trends daily.
- **Customers per Day:** Investigated daily customer patterns.
- **Sales per Customer:** Assessed average sales per customer.
- **Average Sales per Week:** Reviewed weekly sales trends.
- **Weekday Sales:** Analyzed sales trends by day of the week.
- **Sales Overtime:** Examined sales over 6 weeks.
- **Locality:** Studied the impact of store locality on sales.
- **Seasonality:** Identified seasonal patterns in the sales data.
- **Promotions:** Evaluated the effect of promotions on sales.

## Model Training

1. **Ensure Dependencies are Installed:**
   Install required libraries using:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Save the Code:**
   Save the provided code in a Python script file, e.g., `Random_forest.py`.

3. **Run the Script:**
   Execute the script from the command line or an IDE:
   ```bash
   python Random_forest.py
   ```

4. **Understanding the Code:**
   - **Data Loading and Preprocessing:** Normalizes the data.
   - **Feature Creation:** Generates lag features for the Random Forest model.
   - **Training and Evaluation:** Trains the model and assesses performance.
   - **Plotting Results:** Visualizes predictions and saves plots.
   - **Forecasting:** Produces future sales forecasts and saves results.

## Random Forest Overview

**What is Random Forest?**  
Random Forest is a powerful machine learning algorithm combining multiple decision trees' results to enhance predictive accuracy. It is effective for both classification and regression tasks.

**Decision Trees:**  
Decision trees make decisions based on a series of questions, and Random Forest improves their performance by averaging predictions from many such trees.

**Ensemble Methods:**  
Random Forest uses bagging and feature randomness to build an ensemble of uncorrelated decision trees, reducing variance and improving model accuracy.

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
They are used in finance, healthcare, and e-commerce for credit risk evaluation, gene classification, and recommendation engines.

## Model Performance Interpretation

**Training Loss (MSE):** 0.0004147702751754034  
**Validation Loss (MSE):** 0.0028658269385234197

### Interpretation
- The **Training Loss** is quite low, indicating that the model is performing well on the training data.
- The **Validation Loss** is higher compared to the training loss, which suggests that the model might be overfitting to the training data. The model performs well on training data but not as well on new, unseen data.

### Recommendations
To address this, you might consider:
- Applying regularization techniques (e.g., L1/L2 regularization) to reduce overfitting.

## Summary of Updates

### Model Regularization
Improved model generalization by adding constraints to prevent overfitting:
- **`max_depth=10:`** Limits the maximum depth of the trees to prevent overfitting.
- **`min_samples_split=10:`** Increases the minimum number of samples required to split an internal node.
- **`min_samples_leaf=5:`** Increases the minimum number of samples required to be at a leaf node.
- **`max_features='sqrt':`** Limits the number of features considered for the best split to the square root of the total number of features.

### Cross-Validation
Enhanced model evaluation with cross-validation:
- Added cross-validation using `cross_val_score` to assess model performance more robustly.
- Calculated cross-validated MSE and saved it to performance metrics.

### Forecasting
Added functionality to forecast future sales for the next 42 days:
- **`forecast_sales` function:** Predicts future values and returns the forecasted values and corresponding dates.
- Included forecasts for future sales starting from the last date in the dataset.
- Added functionality to forecast sales for 42 days after a specific date (2015-07-31).

### Plotting
Added plotting of the forecast results:
- Plotted historical sales and forecasted sales for the next 42 days.
- Plotted historical sales and forecasted sales for the 42 days after 2015-07-31.

### Saving Results
Saved forecast results to CSV files and plots to PNG files:
- **`sales_forecast.csv:`** For general forecast results.
- **`sales_forecast_after_2015-07-31.csv:`** For forecast results starting from 2015-07-31.
- Added PNG plots for visualizing the forecasts.


## EDA Graphs generated from the machine learning model's dataset

1. **Total Forecast Sales by Day**  
   ![Total Forecast Sales by Day](images/sc14.png)
   
2. **Average Forecast Sales per Week**  
   ![Average Forecast Sales per Week](images/sc13.png)
   
3. **42 Day Sales Forecast**  
   ![42 Day Sales Forecast](images/sc15.png)

## Conclusions

- Store type B shows the highest sales and customer traffic.
- Sales correlate strongly with the number of customers.
- Promotions boost sales and customer counts.
- Sales are higher during school holidays compared to other holidays.
- Increased sales during Christmas week.
- Missing values in certain features don’t necessarily indicate lack of competition.
- Slight seasonality observed in sales data.
- Saturdays are popular shopping days.
