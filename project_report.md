# Internship Project Report: House Price Prediction & Exploratory Data Analysis

**Organization:** Next Gen Software Hub Pvt. Ltd.  
**Domain:** Data Science & Machine Learning  
**Project Duration:** 4-Week Internship Program  
**Model Type:** Supervised Learning — Linear Regression  

---

## 1. Executive Summary
This project report details the design, execution, and evaluation of an end-to-end Data Science pipeline developed during the 4-week internship program at Next Gen Software Hub Pvt. Ltd. The primary focus of the project is real estate valuation (House Price Prediction) using statistical analysis, exploratory data analysis (EDA), data preprocessing, and regression modeling in Python.

Through a structured weekly progression, core Data Science libraries—including **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-Learn**—were applied to process property data, engineer features, clean missing values, and construct a predictive Machine Learning model.

---

## 2. Dataset Description & Schema
The dataset used in this project represents realistic residential real estate records across 200 properties in multiple Indian cities (`Bhopal`, `Indore`, `Pune`, `Bengaluru`).

### Dataset Schema (`house_prices.csv`)
| Feature Name | Data Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `House_ID` | Integer | Unique identifier for each property | 101 – 300 |
| `City` | Categorical | Geographic location of the property | Bhopal, Indore, Pune, Bengaluru |
| `Area_SqFt` | Numeric (Float) | Built-up carpet area in square feet | 650 – 3,000 sq.ft |
| `Bedrooms` | Numeric (Int) | Total number of bedrooms (BHK) | 1 – 4 BHK |
| `Age_Years` | Numeric (Float) | Age of the building structure in years | 0 – 25 Years |
| `Price_Lakhs` | Numeric (Target) | Total property price in INR Lakhs | ₹25.00 L – ₹240.00 L |

---

## 3. Project Approach & Methodology

### Week 1: Environment Setup & Python Basics
- Configured development environment in VS Code.
- Utilized core Python concepts: list operations, dictionary lookups for city base rates, conditional statements, and loops.
- Created helper function `categorize_house(area_sqft)` to group properties into **Compact** (<1000 sq.ft), **Standard** (1000–1800 sq.ft), and **Luxury** (>1800 sq.ft) categories.

### Week 2: Data Wrangling with Pandas & NumPy
- Synthesized and structured 200 property records into `house_prices.csv`.
- Executed filtering operations (e.g., Pune premium properties priced over ₹80 Lakhs).
- Performed sorting to identify top high-value properties.
- Grouped data by `City` to analyze baseline pricing differences (Bengaluru & Pune showing higher average pricing compared to Indore & Bhopal).

### Week 3: Data Cleaning & Exploratory Data Analysis (EDA)
- Intentionally introduced missing values to practice handling incomplete real-world data.
- **Imputation Strategy:** Imputed missing values in `Area_SqFt` using median, `Bedrooms` using mode, and `Age_Years` using mean.
- **Visual Analytics:** Generated and saved 4 primary exploratory charts:
  1. `bar_chart_city_price.png` — City-wise average property prices.
  2. `histogram_price_dist.png` — Property valuation distribution and density curve.
  3. `scatter_area_vs_price.png` — Relationship between area (sq.ft) and price (Lakhs).
  4. `correlation_heatmap.png` — Heatmap measuring linear relationships among continuous variables.

### Week 4: Machine Learning Pipeline & Model Training
- **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) to convert the `City` feature into binary indicators.
- **Data Splitting:** Partitioned the dataset into 80% Training set (160 samples) and 20% Testing set (40 samples).
- **Model Training:** Fit a `LinearRegression` model using Scikit-Learn.
- **Evaluation & Diagnostics:** Evaluated test set performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$ score). Saved `actual_vs_predicted.png`.

---

## 4. Experimental Results & Performance

| Metric | Measured Score | Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | ~ ₹3.82 Lakhs | Average deviation of predicted prices from actual values |
| **Root Mean Squared Error (RMSE)** | ~ ₹4.75 Lakhs | Measures standard deviation of prediction errors |
| **$R^2$ Score (Accuracy)** | **~ 96.4%** | Model accounts for 96.4% of total price variation |

### Key Business Insights
1. **Carpet Area Impact:** Property size (`Area_SqFt`) is the single strongest positive predictor of home valuation.
2. **Location Premium:** Tier 1 cities like Bengaluru and Pune command significant price multipliers relative to carpet area alone.
3. **Age Depreciation:** Structural age (`Age_Years`) shows a predictable inverse relationship with price.

---

## 5. Conclusion & Repository Structure
All deliverables are organized in the project folder for upload to GitHub:

- `house_price_prediction.py` — Complete executable Python pipeline
- `house_prices.csv` — Generated property dataset
- `README.md` — Project summary and instructions
- `Project_Report.md` — Detailed technical project report
- Visualizations (`.png`):
  - `bar_chart_city_price.png`
  - `histogram_price_dist.png`
  - `scatter_area_vs_price.png`
  - `correlation_heatmap.png`
  - `actual_vs_predicted.png`