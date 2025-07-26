import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
# ----------------------------------------------------------------------------------
# Function to show EDA charts
# ----------------------------------------------------------------------------------
@st.cache_data

def show_eda(df):
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

    # define high sales based on IQR and boxplot upper fence
    Q1 = df["Weekly_Sales"].quantile(0.25)
    Q3 = df["Weekly_Sales"].quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR

    df["High_Sales"] = df["Weekly_Sales"] > upper_fence


# ----------------------------------------------------------------------------------
    st.subheader("Weekly Sales Distribution")
    fig_week_distri = px.histogram(df, x='Weekly_Sales', nbins=50)
    st.plotly_chart(fig_week_distri, use_container_width=True)

    st.subheader("Box Plot of Weekly Sales")
    fig_boxplot = px.box(df, y='Weekly_Sales', title="Box Plot of Weekly Sales")
    st.plotly_chart(fig_boxplot, use_container_width=True)

# ----------------------------------------------------------------------------------
    #  Feature Comparison: High Sales vs Normal Sales vs Holiday sales)
    st.subheader("Feature Comparison: High Sales vs Normal Sales vs Holiday sales")
    temp_df = df.copy()
    # Define categories
    temp_df["Sales_Type"] = temp_df.apply(
        lambda row: "High Sales" if row["High_Sales"] else (
            "Holiday Sale" if row["Holiday_Flag"] == 1 else "Normal Sale"
        ),
        axis=1
    )
    grouped = temp_df.groupby("Sales_Type")["Weekly_Sales"].mean().reset_index()
    fig1 = px.bar(grouped,
                 x="Sales_Type",
                 y="Weekly_Sales",
                 color="Sales_Type",
                 title="Average Weekly Sales: High Sales vs Holiday vs Normal",
                 labels={"Weekly_Sales": "Avg Weekly Sales", "Sales_Type": "Sale Type"})
    
    st.plotly_chart(fig1, use_container_width=True)

# ----------------------------------------------------------------------------------
    # Remove outliers from the dataset for better visualization
    # df = df[df["Weekly_Sales"] <= upper_fence].copy()

# ----------------------------------------------------------------------------------
    # Total Weekly Sales per Store
    st.subheader("Total Weekly Sales per Store")
    store_sales = df.groupby("Store")["Weekly_Sales"].sum().reset_index()
    store_fig = px.bar(
        store_sales,
        x="Store",
        y="Weekly_Sales",
        title="Total Weekly Sales by Store"
    )
    st.plotly_chart(store_fig, use_container_width=True)

# ----------------------------------------------------------------------------------
    # ✅ Compare Sales During Holidays vs Non-Holidays for Each Store
    st.subheader("Holiday vs Non-Holiday Sales Per Store")
    holiday_comparison = df.groupby(["Store", "Holiday_Flag"])["Weekly_Sales"].mean().reset_index()

    fig_holiday = px.bar(
        holiday_comparison,
        x="Store",
        y="Weekly_Sales",
        color="Holiday_Flag",
        barmode="group",
        title="Average Weekly Sales: Holiday vs Non-Holiday by Store",
        labels={"Holiday_Flag": "Holiday (1 = Yes, 0 = No)"}
    )
    st.plotly_chart(fig_holiday, use_container_width=True)

    # Compare Sales During Holidays vs Non-Holidays for Each Store (Grouped Bars)
    st.subheader("Holiday vs Non-Holiday Sales Per Store")

    holiday_comparison = df.groupby(["Store", "Holiday_Flag"])["Weekly_Sales"].mean().reset_index()

    fig_holiday = px.bar(
        holiday_comparison,
        x="Store",
        y="Weekly_Sales",
        color="Holiday_Flag",
        barmode="group",
        title="Average Weekly Sales: Holiday vs Non-Holiday by Store",
        labels={
            "Holiday_Flag": "Holiday (1 = Yes, 0 = No)",
            "Weekly_Sales": "Avg Weekly Sales",
            "Store": "Store"
        }
    )

    fig_holiday.update_layout(
        xaxis=dict(type='category'),
        legend_title="Holiday"
    )
    st.plotly_chart(fig_holiday, use_container_width=True)

# ----------------------------------------------------------------------------------
    # Store-wise Comparison: Top 5 vs Bottom 5 Performing Stores
    st.subheader("Top 5 vs Bottom 5 Performing Stores (Average Weekly Sales)")

    store_avg_sales = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
    store_avg_sales.columns = ["Store", "Avg_Weekly_Sales"]
    store_avg_sales["Store"] = store_avg_sales["Store"].astype(str)

    # Get Top 5 and Bottom 5
    top_5 = store_avg_sales.nlargest(5, "Avg_Weekly_Sales")
    bottom_5 = store_avg_sales.nsmallest(5, "Avg_Weekly_Sales")

    # Combine them for plotting
    combined = pd.concat([top_5, bottom_5])
    combined["Performance"] = ["Top"] * 5 + ["Bottom"] * 5

    fig_compare = px.bar(
        combined,
        x="Store",
        y="Avg_Weekly_Sales",
        color="Performance",
        title="Top 5 vs Bottom 5 Stores by Avg Weekly Sales",
        labels={"Store": "Store", "Avg_Weekly_Sales": "Average Weekly Sales"}
    )
    st.plotly_chart(fig_compare, use_container_width=True)

#---------------------------------------------------------------------------------------
    # Total Weekly Sales Over Time (Smoothed)
    st.subheader("Total Weekly Sales trend Over Time")

    # Aggregate sales by date
    sales_over_time = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    sales_over_time.sort_values("Date", inplace=True)

    # Apply smoothing using rolling mean to smooth the sales data
    sales_over_time["Smoothed_Sales"] = sales_over_time["Weekly_Sales"].rolling(window=4, min_periods=1).mean()

    fig_time = px.line(
        sales_over_time,
        x="Date",
        y="Smoothed_Sales",
        title="Total Weekly Sales Over Time (4-Week Rolling Average)",
        labels={"Smoothed_Sales": "Weekly Sales"}
    )
    st.plotly_chart(fig_time, use_container_width=True)

# ----------------------------------------------------------------------------------

    # Heatmap of Weekly Sales by Store
    st.subheader("Heatmap of Weekly Sales by Store")

    weekly_sales = df.groupby(['Store', 'Week'])['Weekly_Sales'].sum().reset_index()

    # Create pivot table: rows = Store, columns = Date, values = Weekly Sales
    sales_pivot = weekly_sales.pivot_table(index='Store', columns='Week', values='Weekly_Sales', aggfunc='sum', fill_value=0)

    # Sort columns chronologically
    sales_pivot = sales_pivot.sort_index(axis=1)

    # Plot heatmap
    fig_heatmap = px.imshow(
        sales_pivot,
        labels=dict(x="Date", y="Store", color="Weekly Sales"),
        x=sales_pivot.columns,
        y=sales_pivot.index,
        aspect="auto",
        color_continuous_scale='Plasma',
    )
    fig_heatmap.update_layout(
        xaxis_nticks=20,
        yaxis_nticks=20,
        title="Weekly Sales Heatmap per Store",
        xaxis=dict(tickangle=45),  # Rotate date labels
        yaxis=dict(title="Store Number"),
        coloraxis_colorbar_title="Weekly Sales (M)"
    )
    fig_heatmap.update_coloraxes(colorbar_len=0.5, colorbar_thickness=20)
    st.plotly_chart(fig_heatmap, use_container_width=True)


    # ----------------------------------------------------------------------------------
    # Weekly sales trend by Store
    st.subheader("Weekly Sales Trend Of Each Store")
    fig_weekly_store_trend = px.line(df, x='Date', y='Weekly_Sales', color='Store', title='Weekly Sales Trend Of Each Store')
    st.plotly_chart(fig_weekly_store_trend, use_container_width=True)

    # ----------------------------------------------------------------------------------

    # Correlation Matrix - Sales vs Other Features
    st.subheader("Correlation Matrix - Sales vs Other Features")
    correlation_features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    corr_matrix = df[correlation_features].corr()

    fig_corr = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=correlation_features,
    y=correlation_features,
    colorscale='Viridis',
    showscale=True
    )
    fig_corr.update_layout(title="Correlation Matrix - Sales vs Other Features")
    st.plotly_chart(fig_corr, use_container_width=True)
    # ----------------------------------------------------------------------------------

    #3. Impact of Temperature on Sales 
    fig_temp_sale = px.scatter(
        df, x='Temperature', y='Weekly_Sales',
        color='Holiday_Flag',  # to see if there's a holiday effect too
        title='Sales vs Temperature Extremes',
        labels={'Temperature': 'Temperature (°F)', 'Weekly_Sales': 'Weekly Sales'}
    )
    st.plotly_chart(fig_temp_sale, use_container_width=True)

    # Impact of Fuel Price on Sales
    fig_fuel_sale = px.scatter(
        df, x='Fuel_Price', y='Weekly_Sales',
        color='Holiday_Flag',  # to see if there's a holiday effect too
        title='Sales vs Fuel Price',
        labels={'Fuel_Price': 'Fuel Price ($)', 'Weekly_Sales': 'Weekly Sales'}
    )
    st.plotly_chart(fig_fuel_sale, use_container_width=True)

    # Impact of CPI on Sales
    fig_cpi_sale = px.scatter(
        df, x='CPI', y='Weekly_Sales',
        color='Holiday_Flag',  # to see if there's a holiday effect too
        title='Sales vs CPI',
        labels={'CPI': 'CPI Index', 'Weekly_Sales': 'Weekly Sales'}
    )
    st.plotly_chart(fig_cpi_sale, use_container_width=True)

    # Impact of Unemployment on Sales
    fig_unemp_sale = px.scatter(
        df, x='Unemployment', y='Weekly_Sales',
        color='Holiday_Flag',  # to see if there's a holiday effect too
        title='Sales vs Unemployment Rate',
        labels={'Unemployment': 'Unemployment Rate (%)', 'Weekly_Sales': 'Weekly Sales'}
    )
    st.plotly_chart(fig_unemp_sale, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # Seasonality Analysis
    st.subheader("Seasonality Analysis")
    fig_season = px.box(
    df, x='Season', y='Weekly_Sales',
    title='Sales Distribution by Season',
    color='Season',
    labels={'Weekly_Sales': 'Weekly Sales'}
    )
    st.plotly_chart(fig_season, use_container_width=True)

    monthly_sales = df.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
    monthly_sales['Month_Year'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))

    fig = px.line(monthly_sales, x='Month_Year', y='Weekly_Sales',
                title='Monthly Sales Trend Over Time',
                labels={'Month_Year': 'Month-Year', 'Weekly_Sales': 'Total Sales'})
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    fig.show()
    st.plotly_chart(fig, use_container_width=True)



    
