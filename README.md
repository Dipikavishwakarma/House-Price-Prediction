# House Price Prediction — Data Science Internship Project

**Organization:** Next Gen Software Hub Pvt. Ltd.  
**Domain:** Data Science (4-Week Internship Program)

## Project Overview
This project demonstrates an end-to-end 4-week Data Science workflow using Python, covering programming fundamentals, data handling with NumPy and Pandas, Exploratory Data Analysis (EDA) and data visualization, and predictive modeling using Linear Regression.

## Dataset Description
- **File:** `house_prices.csv` (generated via `house_price_prediction.py`, 200 records)
- **Features:**
  - `House_ID`: Unique identifier for each property
  - `City`: Property location (`Bhopal`, `Indore`, `Pune`, `Bengaluru`)
  - `Area_SqFt`: Total built-up area in square feet
  - `Bedrooms`: Number of bedrooms
  - `Age_Years`: Age of the property in years
- **Target Variable:**
  - `Price_Lakhs`: Property price in Indian Rupees (Lakhs)

## Weekly Workflow
- **Week 1 — Python Basics:** Used lists, dictionaries, conditional statements, loops, and a helper function (`categorize_house`) to classify property sizes.
- **Week 2 — Data Handling (Pandas & NumPy):** Created and loaded `house_prices.csv`, performed filtering, sorted top-priced properties, and computed city-wise averages using `groupby`.
- **Week 3 — Data Visualization & EDA:** Handled missing values (median, mode, and mean imputation) and created visual charts:
  - `bar_chart_city_price.png` (Average House Price by City)
  - `histogram_price_dist.png` (Distribution of House Prices)
  - `scatter_area_vs_price.png` (House Area vs. Price)
  - `correlation_heatmap.png` (Feature Correlation Heatmap)
- **Week 4 — Machine Learning:** Applied one-hot encoding to `City`, split the dataset into 80% training and 20% testing sets, trained a `LinearRegression` model, evaluated performance using **MAE**, **RMSE**, and **R² Score**, and saved `actual_vs_predicted.png`.

## Tools & Libraries Used
- **Language:** Python 3.x
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn
- **IDE / Version Control:** VS Code, Git & GitHub

## How to Run
1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn