# ==========================================================
# Marketing Campaign Performance Prediction
# Step 3: Regression Model (Revenue Prediction)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==========================================================
# Configuration
# ==========================================================

DATASET = "marketing_campaign_feature_engineered.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv(DATASET)

print("=" * 70)
print("REGRESSION MODEL - REVENUE PREDICTION")
print("=" * 70)

# ==========================================================
# Target Variable
# ==========================================================

TARGET = "Revenue"

# ==========================================================
# Features to Remove
# ==========================================================

drop_columns = [
    "Campaign_ID",
    "Revenue",
    "Date",
    "ROI",
    "Profit_Flag",
    "Revenue_Per_Click",
    "Revenue_Per_Conversion",
]

X = df.drop(columns=drop_columns)
y = df[TARGET]

with open("models/regression_feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# ==========================================================
# Separate Feature Types
# ==========================================================

categorical_columns = X.select_dtypes(include="object").columns.tolist()
numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical Features :", categorical_columns)
print("Numeric Features     :", len(numeric_columns), "columns")

# ==========================================================
# Preprocessing Pipeline
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", StandardScaler(), numeric_columns),
    ]
)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Models
# ==========================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300, max_depth=8, learning_rate=0.05, random_state=42
    ),
}

results = []
best_model = None
best_score = -999
best_name = ""

# ==========================================================
# Model Training
# ==========================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, mse, rmse, r2])

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_name = name

# ==========================================================
# Results Table
# ==========================================================

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "MSE", "RMSE", "R2"]
)

print("\n")
print(results_df)
results_df.to_csv("results/regression_results.csv", index=False)

# ==========================================================
# Save Best Model (pickle)
# ==========================================================

with open("models/best_revenue_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nBest Model :", best_name)
print("Best Model Saved -> models/best_revenue_model.pkl")

# ==========================================================
# Actual vs Predicted
# ==========================================================

pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.tight_layout()
plt.savefig("results/revenue_prediction.png")
plt.close()

# ==========================================================
# Residual Plot
# ==========================================================

residuals = y_test - pred

plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig("results/residual_distribution.png")
plt.close()

# ==========================================================
# Feature Importance
# ==========================================================

print("\nComputing feature importance (permutation importance)...")

from sklearn.inspection import permutation_importance

sample_idx = X_test.sample(n=min(3000, len(X_test)), random_state=42).index
perm_result = permutation_importance(
    best_model,
    X_test.loc[sample_idx],
    y_test.loc[sample_idx],
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_result.importances_mean,
}).sort_values("Importance", ascending=False)

print("\nTop 15 Important Features")
print(importance.head(15))

plt.figure(figsize=(10, 8))
sns.barplot(data=importance.head(15), y="Feature", x="Importance")
plt.title(f"Feature Importance ({best_name})")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 70)
print("Regression Completed")
print("=" * 70)
print("\nGenerated Files")
print("------------------------------")
print("models/best_revenue_model.pkl")
print("models/regression_feature_columns.pkl")
print("results/regression_results.csv")
print("results/revenue_prediction.png")
print("results/residual_distribution.png")
print("results/feature_importance.png")

print("\nBest Model    :", best_name)
print("Best R\u00b2 Score :", round(best_score, 4))

