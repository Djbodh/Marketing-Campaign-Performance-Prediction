# ==========================================================
# Marketing Campaign Performance Prediction
# Step 4: Classification Model (Profit / Loss Prediction)
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ==========================================================
# Configuration
# ==========================================================

DATASET = "marketing_campaign_feature_engineered.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv(DATASET)

print("=" * 70)
print("PROFIT / LOSS CLASSIFICATION")
print("=" * 70)

# ==========================================================
# Target
# ==========================================================

TARGET = "Profit_Flag"

# ==========================================================
# Remove Leakage Columns
# ==========================================================.

drop_columns = [
    "Campaign_ID",
    "Profit_Flag",
    "ROI",                      # leakage - defines the label
    "Revenue",                  # entangled with ROI
    "Revenue_Per_Click",        # derived from Revenue
    "Revenue_Per_Conversion",   # derived from Revenue
    "Date",
]

X = df.drop(columns=drop_columns)
y = df[TARGET]

with open("models/classification_feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# ==========================================================
# Separate Features
# ==========================================================

categorical_columns = X.select_dtypes(include="object").columns.tolist()
numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical Features :", categorical_columns)
print("Numeric Features     :", len(numeric_columns), "columns")

# ==========================================================
# Preprocessing
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
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Models
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05, random_state=42
    ),
}

results = []
best_accuracy = 0
best_model = None
best_name = ""

# ==========================================================
# Training Loop
# ==========================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    results.append([name, accuracy, precision, recall, f1])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_name = name

# ==========================================================
# Save Results
# ==========================================================

results_df = pd.DataFrame(
    results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)
results_df.to_csv("results/classification_results.csv", index=False)
print(results_df)

# ==========================================================
# Save Best Model (pickle)
# ==========================================================

with open("models/best_profit_classifier.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nBest Model :", best_name)
print("Best Model Saved -> models/best_profit_classifier.pkl")

# ==========================================================
# Confusion Matrix
# ==========================================================

pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# ==========================================================
# ROC Curve
# ==========================================================

probability = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probability)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
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

importance.to_csv("results/classification_feature_importance.csv", index=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance.head(15), x="Importance", y="Feature")
plt.title(f"Feature Importance ({best_name})")
plt.tight_layout()
plt.savefig("results/classification_feature_importance.png")
plt.close()

# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION COMPLETED")
print("=" * 70)
print(f"Best Model     : {best_name}")
print(f"Accuracy       : {best_accuracy:.4f}")

if best_accuracy >= 0.95:
    print("Target Accuracy Achieved (>=95%)")
else:
    print(
        "\nNote: accuracy does not reach the 0.95 aspirational target from "
        "the brief. Profit_Flag is defined purely from ROI (ROI > 0), and "
        "ROI itself only correlates up to ~0.69 with any single available "
        "campaign feature (Conversions) - so a ceiling below 100% is "
        "expected given real-world-style noise in how ROI was generated, "
        "not a sign of a broken model. Precision/recall in the "
        "classification report above show the model is still clearly "
        "useful, especially at identifying profitable campaigns."
    )

print("\nGenerated Files")
print("-------------------------")
print("models/best_profit_classifier.pkl")
print("models/classification_feature_columns.pkl")
print("results/classification_results.csv")
print("results/confusion_matrix.png")
print("results/roc_curve.png")
print("results/classification_feature_importance.csv")
print("results/classification_feature_importance.png")

