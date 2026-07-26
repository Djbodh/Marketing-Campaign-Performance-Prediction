# ==========================================================
# Marketing Campaign Performance Prediction
# Exploratory Data Analysis (EDA)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("plotly not installed - skipping the two interactive HTML charts.")
    print("Install with: pip install plotly\n")

# ==========================================================
# Configuration
# ==========================================================

DATASET = "marketing_campaign_cleaned.csv"

OUTPUT_FOLDER = "EDA_Plots"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

sns.set_style("whitegrid")

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATASET)

df["Date"] = pd.to_datetime(df["Date"])

print("=" * 70)
print("Dataset Loaded")
print("=" * 70)

print(df.shape)

# ==========================================================
# Basic Information
# ==========================================================

print("\nInformation\n")
print(df.info())

print("\nSummary Statistics\n")
print(df.describe(include="all"))

# ==========================================================
# Missing Values
# ==========================================================

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/missing_values.png")
plt.close()

# ==========================================================
# Correlation Matrix
# ==========================================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/correlation_heatmap.png")
plt.close()

# ==========================================================
# Brand Distribution
# ==========================================================

plt.figure(figsize=(8, 5))
sns.countplot(x="Brand", data=df, hue="Brand", palette="Set2", legend=False)
plt.title("Campaign Count by Brand")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/brand_distribution.png")
plt.close()

# ==========================================================
# Campaign Type
# ==========================================================

plt.figure(figsize=(10, 6))
sns.countplot(y="Campaign_Type", data=df, hue="Campaign_Type", palette="Set3", legend=False)
plt.title("Campaign Type Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/campaign_type.png")
plt.close()

# ==========================================================
# Revenue Distribution
# ==========================================================

plt.figure(figsize=(10, 6))
sns.histplot(df["Revenue"], bins=40, kde=True, color="green")
plt.title("Revenue Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/revenue_distribution.png")
plt.close()

# ==========================================================
# ROI Distribution
# ==========================================================

plt.figure(figsize=(10, 6))
sns.histplot(df["ROI"], bins=40, kde=True, color="orange")
plt.title("ROI Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/roi_distribution.png")
plt.close()

# ==========================================================
# Revenue vs Acquisition Cost
# ==========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Acquisition_Cost", y="Revenue", hue="Brand", data=df, alpha=0.4)
plt.title("Revenue vs Acquisition Cost")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/revenue_vs_cost.png")
plt.close()

# ==========================================================
# ROI by Brand
# ==========================================================

plt.figure(figsize=(8, 6))
sns.boxplot(x="Brand", y="ROI", data=df, hue="Brand", palette="Pastel1", legend=False)
plt.title("ROI by Brand")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/roi_brand.png")
plt.close()

# ==========================================================
# Revenue by Brand
# ==========================================================

plt.figure(figsize=(8, 6))
sns.barplot(x="Brand", y="Revenue", data=df, hue="Brand", estimator=np.mean, legend=False)
plt.title("Average Revenue by Brand")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/avg_revenue_brand.png")
plt.close()

# ==========================================================
# Engagement Score
# ==========================================================

plt.figure(figsize=(10, 6))
sns.histplot(df["Engagement_Score"], bins=30, kde=True, color="purple")
plt.title("Engagement Score Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/engagement_score.png")
plt.close()

# ==========================================================
# Clicks vs Revenue
# ==========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Clicks", y="Revenue", hue="Brand", data=df, alpha=0.4)
plt.title("Clicks vs Revenue")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/clicks_vs_revenue.png")
plt.close()

# ==========================================================
# Impressions vs Clicks
# ==========================================================

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Impressions", y="Clicks", hue="Brand", data=df, alpha=0.4)
plt.title("Impressions vs Clicks")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/impressions_vs_clicks.png")
plt.close()

# ==========================================================
# Customer Segment
# ==========================================================

plt.figure(figsize=(10, 6))
sns.countplot(y="Customer_Segment", data=df, hue="Customer_Segment", legend=False)
plt.title("Customer Segment Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/customer_segment.png")
plt.close()

# ==========================================================
# Language Distribution
# ==========================================================

plt.figure(figsize=(8, 6))
sns.countplot(y="Language", data=df, hue="Language", legend=False)
plt.title("Language Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/language.png")
plt.close()

# ==========================================================
# Channel Usage
# ==========================================================

channels = df["Channel_Used"].str.split(",").explode().str.strip()

plt.figure(figsize=(10, 6))
channels.value_counts().plot.bar()
plt.title("Marketing Channel Usage")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/channels.png")
plt.close()

# ==========================================================
# Monthly Revenue Trend
# ==========================================================

monthly = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
monthly.index = monthly.index.astype(str)

plt.figure(figsize=(14, 6))
monthly.plot()
plt.title("Monthly Revenue Trend")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/monthly_revenue.png")
plt.close()

# ==========================================================
# Top 10 Campaigns
# ==========================================================

top = df.sort_values("Revenue", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x="Revenue", y="Campaign_ID", data=top, hue="Campaign_ID", legend=False, order=top["Campaign_ID"])
plt.title("Top 10 Campaigns by Revenue")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/top_campaigns.png")
plt.close()

# ==========================================================
# Interactive Plotly Charts
# ==========================================================

if PLOTLY_AVAILABLE:

    fig = px.scatter(
        df, x="Acquisition_Cost", y="Revenue", color="Brand", hover_data=["Campaign_Type"]
    )
    fig.write_html(OUTPUT_FOLDER + "/interactive_revenue.html")

    sunburst_df = df.dropna(subset=["Brand", "Campaign_Type", "Customer_Segment"])
    fig2 = px.sunburst(
        sunburst_df, path=["Brand", "Campaign_Type", "Customer_Segment"], values="Revenue"
    )
    fig2.write_html(OUTPUT_FOLDER + "/campaign_sunburst.html")

# ==========================================================
# Business Insights
# ==========================================================

print("=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)

print("\nAverage Revenue by Brand")
print(df.groupby("Brand")["Revenue"].mean())

print("\nAverage ROI by Brand")
print(df.groupby("Brand")["ROI"].mean())

print("\nTop 10 Campaigns")
print(top[["Campaign_ID", "Brand", "Revenue", "ROI"]])

print("\nMost Used Marketing Channels")
print(channels.value_counts().head(10))

print("\nCampaign Types")
print(df["Campaign_Type"].value_counts())

print("\nCustomer Segments")
print(df["Customer_Segment"].value_counts())

print("=" * 70)
print("EDA COMPLETED")
print("=" * 70)
print(f"Plots saved in folder: {OUTPUT_FOLDER}")
