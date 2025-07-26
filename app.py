import numpy as np
import streamlit as st
import pandas as pd
from utility.eda_charts import show_eda
from utility.prediction import predict_sales
import joblib

st.set_page_config(page_title="Sales Forecast App", layout="wide")

st.title("📊 Retail Sales Forecast Dashboard")

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("Walmart_Sales.csv", parse_dates=['Date'])

df = load_data()

xgb_model = joblib.load("xgb_model.pkl")
prophet_model = joblib.load("prophet_model.pkl")



# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["EDA Dashboard", "Predict Sales"])

if page == "EDA Dashboard":
    show_eda(df)

elif page == "Predict Sales":
    predict_sales(df, xgb_model, prophet_model)
