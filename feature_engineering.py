# ==========================================================
# Marketing Campaign Performance Prediction
# Step 2: Feature Engineering
# ==========================================================
#
# Builds all model-ready features on top of the cleaned dataset:
#   - date-derived features (Year, Month, Day, Weekday)
#   - Profit_Flag target for the classification model (ROI > 0)
#   - performance ratio features (CTR, Lead_Rate, Conversion_Rate,
#     CPC, CPL, Revenue_Per_Click, Revenue_Per_Conversion)
#   - multi-label encoding of Channel_Used (a campaign can use
#     several channels at once, e.g. "WhatsApp, YouTube")
#
# Saved with pickle (not joblib), per project requirements.
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

os.makedirs("models", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

DATASET = "marketing_campaign_cleaned.csv"

df = pd.read_csv(DATASET)

print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print("Dataset Shape :", df.shape)

# ==========================================================
# Date Features
# ==========================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()

# ==========================================================
# Profit Flag (Classification Target)
# ==========================================================

df["Profit_Flag"] = np.where(df["ROI"] > 0, 1, 0)

print("\nProfit Distribution")
print(df["Profit_Flag"].value_counts())

# ==========================================================
# Performance Ratio Features
# ==========================================================

df["CTR"] = df["Clicks"] / df["Impressions"]                     # Click Through Rate
df["Lead_Rate"] = df["Leads"] / df["Clicks"]                      # Lead Conversion Rate
df["Conversion_Rate"] = df["Conversions"] / df["Leads"]           # Sales Conversion Rate
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"]                 # Cost Per Click
df["CPL"] = df["Acquisition_Cost"] / df["Leads"]                  # Cost Per Lead
df["Revenue_Per_Click"] = df["Revenue"] / df["Clicks"]
df["Revenue_Per_Conversion"] = df["Revenue"] / df["Conversions"]

# Replace division-by-zero artifacts (inf/-inf) and any resulting NaN
df.replace([np.inf, -np.inf], 0, inplace=True)

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna("Unknown")

# ==========================================================
# Multi-Label Encoding: Channel_Used
# ==========================================================

print("\nEncoding Channel_Used...")

df["Channel_Used"] = df["Channel_Used"].astype(str).str.split(",")
df["Channel_Used"] = df["Channel_Used"].apply(
    lambda channels: [c.strip() for c in channels]
)

mlb = MultiLabelBinarizer()
channel_encoded = pd.DataFrame(
    mlb.fit_transform(df["Channel_Used"]),
    columns=mlb.classes_,
    index=df.index,
)

print("\nMarketing Channels")
print(mlb.classes_)

df = pd.concat([df, channel_encoded], axis=1)
df.drop(columns=["Channel_Used"], inplace=True)

# ==========================================================
# Save Channel Encoder (pickle)
# ==========================================================

with open("models/channel_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("\nChannel encoder saved -> models/channel_encoder.pkl")

# NOTE: One-hot encoding / scaling of the remaining categorical and
# numeric columns is intentionally NOT fitted here. The regression
# and classification models drop different columns (e.g. the
# classifier excludes Revenue and ROI to avoid target leakage), so
# each model builds and fits its own ColumnTransformer inside its own
# Pipeline in regression_model.py / classification_model.py. That
# keeps a single saved model file (pickle) fully self-contained -
# preprocessing + estimator together - which is what the Streamlit
# app loads for inference.

# ==========================================================
# Save Feature Engineered Dataset
# ==========================================================

OUTPUT_FILE = "marketing_campaign_feature_engineered.csv"

df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature Engineered Dataset Saved")
print(df.shape)

# ==========================================================
# Summary
# ==========================================================

print("=" * 70)
print("Feature Engineering Completed")
print("=" * 70)

print("\nCreated Features")
new_features = [
    "Profit_Flag",
    "CTR",
    "Lead_Rate",
    "Conversion_Rate",
    "CPC",
    "CPL",
    "Revenue_Per_Click",
    "Revenue_Per_Conversion",
    "Year",
    "Month",
    "Day",
    "Weekday",
]
for feature in new_features:
    print("\u2714", feature)

print("\nEncoded Channels")
for channel in mlb.classes_:
    print("\u2714", channel)

print("\nFiles Generated")
print("---------------------------")
print(OUTPUT_FILE)
print("models/channel_encoder.pkl")
print("=" * 70)
