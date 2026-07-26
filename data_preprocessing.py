# ==========================================================
# Marketing Campaign Performance Prediction
# Step 1: Data Preprocessing Pipeline
# ==========================================================
#
# Loads the three brand-wise raw CSV exports (Nykaa, Purplle, Tira),
# merges them into a single dataset, and cleans it up:
#   - missing Campaign_IDs are regenerated (don't collapse into "nan")
#   - missing values are imputed (brand-aware for numeric columns)
#   - dates are parsed as DD-MM-YYYY (source format), not the pandas
#     default MM-DD-YYYY, which silently corrupts ambiguous dates
#   - ROI is sanity-checked rather than blindly overwritten by a
#     formula that does not match how this dataset was generated
#     (see note below)
#   - duplicate records and impossible negative values are removed
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

NYKAA_FILE = "nykaa_campaign_data_with_nulls.csv"
PURPLLE_FILE = "purplle_campaign_data_with_nulls.csv"
TIRA_FILE = "tira_campaign_data_with_nulls.csv"

OUTPUT_FILE = "marketing_campaign_cleaned.csv"

NUMERIC_COLUMNS = [
    "Duration",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Revenue",
    "Acquisition_Cost",
    "ROI",
    "Engagement_Score",
]

CATEGORICAL_COLUMNS = [
    "Campaign_Type",
    "Target_Audience",
    "Channel_Used",
    "Language",
    "Customer_Segment",
]

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("Loading datasets...")
print("=" * 70)

nykaa = pd.read_csv(NYKAA_FILE)
purplle = pd.read_csv(PURPLLE_FILE)
tira = pd.read_csv(TIRA_FILE)

nykaa["Brand"] = "Nykaa"
purplle["Brand"] = "Purplle"
tira["Brand"] = "Tira"

df = pd.concat([nykaa, purplle, tira], ignore_index=True)

print("\nDatasets Loaded Successfully")
print(f"Nykaa Rows     : {len(nykaa)}")
print(f"Purplle Rows   : {len(purplle)}")
print(f"Tira Rows      : {len(tira)}")
print(f"\nTotal Rows     : {len(df)}")

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values (raw)")
print(df.isnull().sum())

# ==========================================================
# Duplicate Records
# ==========================================================

duplicate_count = df.duplicated().sum()
print("\nDuplicate Records :", duplicate_count)

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates Removed.")

# ==========================================================
# Fix Missing Campaign_ID
# ==========================================================
# Purplle and Tira have null Campaign_IDs. Casting NaN -> str() turns
# every one of them into the literal string "nan", which collapses
# thousands of distinct campaigns into a single duplicate ID. Instead
# we regenerate a unique ID from the brand + row position.

brand_prefix = {"Nykaa": "NY", "Purplle": "PU", "Tira": "TI"}

missing_id_mask = df["Campaign_ID"].isnull()
print("\nMissing Campaign_ID Records :", missing_id_mask.sum())

if missing_id_mask.sum() > 0:
    generated_ids = (
        df.loc[missing_id_mask, "Brand"].map(brand_prefix)
        + "-GEN-"
        + df.loc[missing_id_mask].index.astype(str)
    )
    df.loc[missing_id_mask, "Campaign_ID"] = generated_ids

df["Campaign_ID"] = df["Campaign_ID"].astype(str)

# ==========================================================
# Fill Numeric Missing Values (median, per Brand)
# ==========================================================
# Imputing per brand keeps the fill value representative of that
# brand's own campaign scale instead of blending all three together.

for column in NUMERIC_COLUMNS:
    if column in df.columns:
        df[column] = df.groupby("Brand")[column].transform(
            lambda s: s.fillna(s.median())
        )

print("\nNumeric Missing Values Filled (brand-wise median)")

# ==========================================================
# Fill Categorical Missing Values (mode)
# ==========================================================

for column in CATEGORICAL_COLUMNS:
    if column in df.columns:
        mode_value = df[column].mode(dropna=True)[0]
        df[column] = df[column].fillna(mode_value)

print("Categorical Missing Values Filled (mode)")

# ==========================================================
# Date Conversion
# ==========================================================
# Source dates are DD-MM-YYYY (e.g. 06-04-2025 = 6 April 2025).
# pandas' default parser assumes MM-DD-YYYY, which either silently
# swaps day/month for day <= 12, or raises/NaTs everything else.
# dayfirst=True parses this dataset correctly.

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

invalid_dates = df["Date"].isnull().sum()
print("\nUnparseable Dates :", invalid_dates)

if invalid_dates > 0:
    median_date = df["Date"].median()
    df["Date"] = df["Date"].fillna(median_date)

# ==========================================================
# ROI Sanity Check
# ==========================================================
# The textbook formula ROI = ((Revenue - Cost) / Cost) * 100 does NOT
# reconcile with this dataset: Revenue is generated on a much larger
# scale than Acquisition_Cost, so applying that formula would replace
# realistic ROI values (range ~ -1 to 80) with numbers in the millions
# of percent for almost every row. Rather than overwrite good data
# with a formula that doesn't match how ROI was actually generated,
# we keep the provided ROI and only guard against values that are
# logically impossible (a campaign cannot lose more than 100% of
# investment, i.e. ROI below -1 as a ratio).

print("\nValidating ROI...")

impossible_roi = df["ROI"] < -1
print("Impossible ROI Records (< -1):", impossible_roi.sum())

if impossible_roi.sum() > 0:
    median_roi = df.loc[~impossible_roi, "ROI"].median()
    df.loc[impossible_roi, "ROI"] = median_roi

print("ROI Validation Completed")

# ==========================================================
# Negative Value Checks
# ==========================================================

print("\nChecking Negative Values")
for column in [
    "Revenue",
    "Acquisition_Cost",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Duration",
    "Engagement_Score",
]:
    negative = (df[column] < 0).sum()
    print(f"{column:20} : {negative}")

# Remove impossible values (defensive - dataset currently has none)
df = df[df["Revenue"] >= 0]
df = df[df["Acquisition_Cost"] >= 0]
df = df[df["Impressions"] >= 0]
df = df[df["Clicks"] >= 0]
df = df[df["Leads"] >= 0]
df = df[df["Conversions"] >= 0]
df = df[df["Duration"] >= 0]

# ==========================================================
# Final Checks
# ==========================================================

print("\nClean Dataset Shape")
print(df.shape)

print("\nRemaining Missing Values")
print(df.isnull().sum())

# ==========================================================
# Save Clean Dataset
# ==========================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nClean dataset saved as")
print(OUTPUT_FILE)
print("\nData preprocessing completed successfully.")
