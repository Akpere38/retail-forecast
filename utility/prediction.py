import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import plotly.graph_objects as go


def predict_sales(df, xgb_model, prophet_model):

    metrics = joblib.load('all_model_metrics.pkl')
    # Assign values
    prophet_mae = metrics['prophet']['mae']
    prophet_rmse = metrics['prophet']['rmse']
    prophet_r2 = metrics['prophet']['r2']

    xgb_mae = metrics['xgboost']['mae']
    xgb_rmse = metrics['xgboost']['rmse']
    xgb_r2 = metrics['xgboost']['r2']

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)  # Ensure Date is in datetime format
    # Extract month and week from the date
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    # Optional: Define Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Season'] = df['Month'].apply(get_season)

    
    st.subheader("📊 Predict Weekly Sales")

    # Form
    with st.form("prediction_form"):
        date = st.date_input("Select Date")
        store = st.selectbox("Store", sorted(df['Store'].unique()))
        temperature = st.number_input("Temperature", value=float(df["Temperature"].mean()))
        fuel_price = st.number_input("Fuel Price", value=float(df["Fuel_Price"].mean()))
        cpi = st.number_input("CPI", value=float(df["CPI"].mean()))
        unemployment = st.number_input("Unemployment", value=float(df["Unemployment"].mean()))
        holiday = st.selectbox("Holiday", ["No", "Yes"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        # ----------------------------
        # XGBOOST PREDICTION
        # ----------------------------
        input_df = pd.DataFrame([{
            'Store': store,
            'Date': pd.to_datetime(date),
            'Temperature': temperature,
            'Fuel_Price': fuel_price,
            'CPI': cpi,
            'Unemployment': unemployment,
            'Holiday_Flag': 1 if holiday == "Yes" else 0
        }])

        # Date-based features
        input_df['Month'] = input_df['Date'].dt.month
        input_df['Year'] = input_df['Date'].dt.year
        input_df['Week'] = input_df['Date'].dt.isocalendar().week
        input_df['Day'] = input_df['Date'].dt.day
        input_df['Is_Weekend'] = input_df['Date'].dt.weekday >= 5

        # Most frequent season
        input_df['Season'] = df['Season'].mode()[0]
        input_df = pd.get_dummies(input_df, columns=['Season'], drop_first=True)

        # Fill missing columns to match training
        for col in xgb_model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[xgb_model.feature_names_in_]

        xgb_prediction = xgb_model.predict(input_df)[0]

        # ----------------------------
        # PROPHET PREDICTION
        # ----------------------------
        future_df = pd.DataFrame({'ds': [pd.to_datetime(date)]})
        prophet_forecast = prophet_model.predict(future_df)
        prophet_prediction = prophet_forecast['yhat'].values[0]

        # ----------------------------
        # DISPLAY RESULTS
        # ----------------------------
        st.markdown("### 🔮 Predictions:")
        st.metric(f"XGBoost Predicted Weekly Sales for store {store}", f"${xgb_prediction:,.2f}")
        st.metric("Prophet Predicted total Weekly Sales", f"${prophet_prediction:,.2f}")

        st.markdown("### 📊 Model Performance:")
        # Comparison Bar Chart
        comparison_fig = go.Figure()

        comparison_fig.add_trace(go.Bar(
            x=["XGBoost", "Prophet"],
            y=[xgb_prediction, prophet_prediction],
            text=[f"${xgb_prediction:,.2f}", f"${prophet_prediction:,.2f}"],
            textposition='auto',
            marker_color=['#636EFA', '#00CC96']
        ))

        comparison_fig.update_layout(
            title="🔍 Sales Prediction Comparison",
            xaxis_title="Model",
            yaxis_title="Predicted Weekly Sales",
            yaxis_tickprefix="$",
            height=400,
            template='plotly_white'
        )

        st.plotly_chart(comparison_fig, use_container_width=True)

    st.markdown("### 📈 Model Metrics:")
    st.markdown("### ✅ Model Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔮 Prophet Model**")
        st.metric("MAE", f"${prophet_mae:,.2f}")
        st.metric("RMSE", f"${prophet_rmse:,.2f}")
        st.metric("R²", f"{prophet_r2:.4f}")

    with col2:
        st.markdown("**⚡ XGBoost Model**")
        st.metric("MAE", f"${xgb_mae:,.2f}")
        st.metric("RMSE", f"${xgb_rmse:,.2f}")
        st.metric("R²", f"{xgb_r2:.4f}")
    
    st.markdown("""
#### 📘 Interpretation of Metrics

- **MAE (Mean Absolute Error)** measures the average difference between the predicted and actual sales. Lower values indicate better accuracy.
- **RMSE (Root Mean Squared Error)** penalizes larger errors more than MAE. It gives you a sense of how far predictions deviate from actual values.
- **R² (R-squared)** tells you how well the model explains the variance in the data. A value closer to 1 means the model fits the data very well.

In this case:

- The **XGBoost model** has a much higher R² (≈ 0.98), meaning it captures the patterns in the data extremely well. Which makes it suitable for forecasting weekly sales for each store However, its MAE and RMSE are slightly higher than Prophet's, suggesting it may overfit or be influenced by extreme values.
  
- The **Prophet model**, while having a lower R², performs better in terms of MAE and RMSE, indicating it may be more stable and generalizable for forecasting, in this case total weekly sales of all stores.

You can choose between the two based on whether you prioritize variance explanation as with the case with each store(XGBoost) or prediction consistency i.e total weekly sales regardless of the input features (Prophet).
""")



