# ==========================================
# WEEK 1: Python Basics & Environment Setup
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Lists and dictionaries for sample cities and base price per sq.ft
cities = ["Bhopal", "Indore", "Pune", "Bengaluru"]
city_base_rate = {
    "Bhopal": 3500,
    "Indore": 4200,
    "Pune": 6500,
    "Bengaluru": 8000,
}


def categorize_house(area_sqft):
    """Classify house size based on area in square feet."""
    if area_sqft < 1000:
        return "Compact"
    elif area_sqft <= 1800:
        return "Standard"
    else:
        return "Luxury"


sample_areas = [850, 1400, 2200]
for area in sample_areas:
    print(f"Area: {area} sq.ft -> Category: {categorize_house(area)}")


# ==========================================
# WEEK 2: Data Handling with Pandas & NumPy
# ==========================================
np.random.seed(42)
n_samples = 200

data = {
    "House_ID": range(101, 101 + n_samples),
    "City": np.random.choice(cities, n_samples),
    "Area_SqFt": np.random.randint(650, 3000, n_samples),
    "Bedrooms": np.random.randint(1, 5, n_samples),
    "Age_Years": np.random.randint(0, 25, n_samples),
}

df = pd.DataFrame(data)

# Generate realistic house prices (in Lakhs) with noise
df["Price_Lakhs"] = (
    (df["Area_SqFt"] * df["City"].map(city_base_rate) / 100000)
    + (df["Bedrooms"] * 4.5)
    - (df["Age_Years"] * 0.4)
    + np.random.normal(0, 5, n_samples)
).round(2)

# Introduce missing values for Week 3 data cleaning practice
df.loc[5, "Area_SqFt"] = np.nan
df.loc[15, "Bedrooms"] = np.nan
df.loc[25, "Age_Years"] = np.nan

# Save and reload the CSV dataset
df.to_csv("house_prices.csv", index=False)
df = pd.read_csv("house_prices.csv")

print("\n--- First 5 Rows ---")
print(df.head())

# Filter houses in Pune priced above 80 Lakhs
pune_premium = df[(df["City"] == "Pune") & (df["Price_Lakhs"] > 80)]
print(f"\nPune Premium Houses (>80 Lakhs): {len(pune_premium)}")

# Sort top 5 most expensive houses
top_houses = df.sort_values(by="Price_Lakhs", ascending=False).head(5)
print("\n--- Top 5 Most Expensive Houses ---")
print(top_houses[["House_ID", "City", "Area_SqFt", "Price_Lakhs"]])

# Group by City to compute average area and price
city_summary = df.groupby("City")[["Area_SqFt", "Price_Lakhs"]].mean().round(2)
print("\n--- City-Wise Average Summary ---")
print(city_summary)


# ==========================================
# WEEK 3: Data Visualization & EDA
# ==========================================
print("\nMissing values before cleaning:\n", df.isnull().sum())

df["Area_SqFt"] = df["Area_SqFt"].fillna(df["Area_SqFt"].median())
df["Bedrooms"] = df["Bedrooms"].fillna(df["Bedrooms"].mode()[0])
df["Age_Years"] = df["Age_Years"].fillna(df["Age_Years"].mean())

print("\nMissing values after cleaning:\n", df.isnull().sum())

sns.set_theme(style="whitegrid")

# 1. Bar Chart: Average House Price by City
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="City",
    y="Price_Lakhs",
    estimator="mean",
    palette="Blues_d",
    errorbar=None,
)
plt.title("Average House Price by City (in Lakhs)")
plt.xlabel("City")
plt.ylabel("Average Price (Lakhs)")
plt.savefig("bar_chart_city_price.png", bbox_inches="tight")
plt.show()

# 2. Histogram: Distribution of House Prices
plt.figure(figsize=(8, 5))
sns.histplot(df["Price_Lakhs"], bins=20, kde=True, color="teal")
plt.title("Distribution of House Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Frequency")
plt.savefig("histogram_price_dist.png", bbox_inches="tight")
plt.show()

# 3. Scatter Plot: Area vs. Price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Area_SqFt", y="Price_Lakhs", hue="City", palette="viridis"
)
plt.title("House Area (Sq.Ft) vs. Price (Lakhs)")
plt.xlabel("Area (Sq.Ft)")
plt.ylabel("Price (Lakhs)")
plt.savefig("scatter_area_vs_price.png", bbox_inches="tight")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(7, 5))
numeric_cols = df[["Area_SqFt", "Bedrooms", "Age_Years", "Price_Lakhs"]]
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.show()


# ==========================================
# WEEK 4: Machine Learning (Linear Regression)
# ==========================================
# One-hot encode categorical 'City' column
df_encoded = pd.get_dummies(
    df.drop(columns=["House_ID"]), columns=["City"], drop_first=True
)

# Define features (X) and target (y)
X = df_encoded.drop(columns=["Price_Lakhs"])
y = df_encoded["Price_Lakhs"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")
print(f"R² Score: {r2 * 100:.2f}%")

# Actual vs. Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="royalblue", alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
)
plt.title("Actual vs. Predicted House Prices")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.savefig("actual_vs_predicted.png", bbox_inches="tight")
plt.show()