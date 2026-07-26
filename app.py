# ==========================================================
# Marketing Campaign Performance Prediction
# Step 5: Production Streamlit Application
# ==========================================================

import pickle
import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Marketing Campaign Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Marketing Campaign Performance Prediction")
st.caption("Nykaa · Purplle · Tira — Revenue forecasting and Profit/Loss prediction")
st.markdown("---")

# ==========================================================
# Load Models and Schema (pickle)
# ==========================================================


@st.cache_resource
def load_assets():
    with open("models/best_revenue_model.pkl", "rb") as f:
        revenue_m = pickle.load(f)
    with open("models/best_profit_classifier.pkl", "rb") as f:
        profit_m = pickle.load(f)
    with open("models/channel_encoder.pkl", "rb") as f:
        mlb_enc = pickle.load(f)
    with open("models/regression_feature_columns.pkl", "rb") as f:
        regression_cols = pickle.load(f)
    with open("models/classification_feature_columns.pkl", "rb") as f:
        classification_cols = pickle.load(f)
    return revenue_m, profit_m, mlb_enc, regression_cols, classification_cols


try:
    (
        revenue_model,
        profit_model,
        mlb,
        regression_feature_columns,
        classification_feature_columns,
    ) = load_assets()
except Exception:
    st.error(
        "Model files missing. Please run data_preprocessing.py, "
        "feature_engineering.py, regression_model.py and "
        "classification_model.py first (in that order)."
    )
    st.stop()

# ==========================================================
# Sidebar - Campaign Inputs
# ==========================================================

st.sidebar.header("Campaign Details")

brand = st.sidebar.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
campaign = st.sidebar.selectbox(
    "Campaign Type", ["Social Media", "Paid Ads", "Influencer", "Email", "SEO"]
)
audience = st.sidebar.selectbox(
    "Target Audience",
    ["College Students", "Tier 2 City Customers", "Youth", "Working Women", "Premium Shoppers"],
)
language = st.sidebar.selectbox("Language", ["Hindi", "English", "Tamil", "Bengali"])
customer = st.sidebar.selectbox(
    "Customer Segment",
    ["College Students", "Tier 2 City Customers", "Premium Shoppers", "Youth", "Working Women"],
)
campaign_date = st.sidebar.date_input("Campaign Date", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.header("Reach & Engagement")

duration = st.sidebar.number_input("Duration (Days)", 1, 365, 30)
impressions = st.sidebar.number_input("Impressions", 100, 10_000_000, 100_000, step=1000)
clicks = st.sidebar.number_input("Clicks", 0, 1_000_000, 5_000, step=100)
leads = st.sidebar.number_input("Leads", 0, 500_000, 500, step=10)
conversions = st.sidebar.number_input("Conversions", 0, 100_000, 100, step=10)
engagement = st.sidebar.slider("Engagement Score", 0.0, 100.0, 50.0)

st.sidebar.markdown("---")
st.sidebar.header("Spend")

cost = st.sidebar.number_input("Acquisition Cost (₹)", 0.0, 10_000_000.0, 10_000.0, step=500.0)

st.sidebar.markdown("---")
st.sidebar.header("Marketing Channels")

all_channels = list(mlb.classes_)
selected_channels = st.sidebar.multiselect(
    "Channels Used", all_channels, default=[all_channels[0]]
)

# ==========================================================
# Input Sanity Warnings (non-blocking)
# ==========================================================

warnings_list = []
if clicks > impressions:
    warnings_list.append("Clicks exceed Impressions - that's not usually possible.")
if leads > clicks:
    warnings_list.append("Leads exceed Clicks - that's not usually possible.")
if conversions > leads:
    warnings_list.append("Conversions exceed Leads - that's not usually possible.")
if not selected_channels:
    warnings_list.append("No marketing channel selected - pick at least one for a realistic prediction.")

if warnings_list:
    st.warning(" / ".join(warnings_list))

# ==========================================================
# Feature Derivation (mirrors feature_engineering.py)
# ==========================================================

ctr = clicks / impressions if impressions > 0 else 0
lead_rate = leads / clicks if clicks > 0 else 0
conversion_rate = conversions / leads if leads > 0 else 0
cpc = cost / clicks if clicks > 0 else 0
cpl = cost / leads if leads > 0 else 0

year, month, day = campaign_date.year, campaign_date.month, campaign_date.day
weekday = campaign_date.strftime("%A")

# ==========================================================
# Build Inference Row
# ==========================================================
# Note: Revenue_Per_Click / Revenue_Per_Conversion are NOT included -
# they require knowing Revenue, which is exactly what we're predicting.
# Both trained models were built without them (see regression_model.py /
# classification_model.py) to avoid target leakage, so neither is needed
# here either.

input_dict = {
    "Campaign_Type": campaign,
    "Target_Audience": audience,
    "Language": language,
    "Customer_Segment": customer,
    "Brand": brand,
    "Weekday": weekday,
    "Duration": duration,
    "Impressions": impressions,
    "Clicks": clicks,
    "Leads": leads,
    "Conversions": conversions,
    "Acquisition_Cost": cost,
    "Engagement_Score": engagement,
    "CTR": ctr,
    "Lead_Rate": lead_rate,
    "Conversion_Rate": conversion_rate,
    "CPC": cpc,
    "CPL": cpl,
    "Year": year,
    "Month": month,
    "Day": day,
}

for channel in all_channels:
    input_dict[channel] = 1 if channel in selected_channels else 0

input_df = pd.DataFrame([input_dict])

# Each model gets its own exact column set/order, in case the two ever
# diverge (e.g. if the models are retrained differently in future).
regression_input = input_df[regression_feature_columns]
classification_input = input_df[classification_feature_columns]

# ==========================================================
# Prediction Actions - two independent buttons
# ==========================================================

st.subheader("Run a Prediction")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    predict_revenue_clicked = st.button("💰 Predict Revenue", use_container_width=True)

with col_btn2:
    predict_profit_clicked = st.button("📊 Predict Profit / Loss", use_container_width=True)

if predict_revenue_clicked:
    revenue_pred = revenue_model.predict(regression_input)[0]
    st.session_state["revenue_pred"] = revenue_pred

if predict_profit_clicked:
    profit_pred = profit_model.predict(classification_input)[0]
    profit_proba = profit_model.predict_proba(classification_input)[0][1]
    st.session_state["profit_pred"] = profit_pred
    st.session_state["profit_proba"] = profit_proba

st.markdown("---")

# ==========================================================
# Results Display (persists across reruns via session_state)
# ==========================================================

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.markdown("### Revenue Prediction")
    if "revenue_pred" in st.session_state:
        st.metric("Predicted Revenue", f"₹ {st.session_state['revenue_pred']:,.2f}")
    else:
        st.info("Click **Predict Revenue** to see a forecast.")

with result_col2:
    st.markdown("### Profit / Loss Prediction")
    if "profit_pred" in st.session_state:
        proba = st.session_state["profit_proba"]
        if st.session_state["profit_pred"] == 1:
            st.success(f"Profitable Campaign  (confidence: {proba * 100:.2f}%)")
        else:
            st.error(f"Loss-Making Campaign  (confidence: {(1 - proba) * 100:.2f}%)")
    else:
        st.info("Click **Predict Profit / Loss** to see a forecast.")

# ==========================================================
# Visualize Key Inputs
# ==========================================================

if "revenue_pred" in st.session_state or "profit_pred" in st.session_state:

    st.markdown("---")
    st.subheader("Campaign Input Overview")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Performance Funnel**")
        funnel_data = pd.DataFrame({
            "Stage": ["Impressions", "Clicks", "Leads", "Conversions"],
            "Value": [impressions, clicks, leads, conversions],
        }).set_index("Stage")
        st.bar_chart(funnel_data)

    with viz_col2:
        st.markdown("**Efficiency Ratios**")
        ratio_data = pd.DataFrame({
            "Metric": ["CTR", "Lead Rate", "Conversion Rate"],
            "Value": [ctr, lead_rate, conversion_rate],
        }).set_index("Metric")
        st.bar_chart(ratio_data)

    st.markdown("**Raw Model Inputs**")
    st.dataframe(input_df, use_container_width=True)
