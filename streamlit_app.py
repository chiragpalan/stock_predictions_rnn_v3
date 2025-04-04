import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, time

# Set page configuration for a better layout
st.set_page_config(layout="wide")

# Function to fetch table names from the database
def fetch_table_names(db_path):
    conn = sqlite3.connect(db_path)
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    conn.close()
    return tables

# Fetch available tables from both actual and predicted databases
actual_db_path = 'nifty50_data_v1.db'
pred_db_path = 'predictions/predictions.db'
actual_tables = fetch_table_names(actual_db_path)
pred_tables = fetch_table_names(pred_db_path)

# Combine actual and predicted table names for selection
table_options = list(set(actual_tables) & set([t.replace('_predictions', '') for t in pred_tables]))

# Create the dropdown menu for table selection
selected_table = st.selectbox("Select Table", table_options)

# Function to load the selected table's data and plot the candlestick chart
def load_and_plot_data(selected_table):
    # Connect to the databases
    actual_conn = sqlite3.connect(actual_db_path)
    pred_conn = sqlite3.connect(pred_db_path)

    # Load the actual and predicted data based on selected table
    actual_df = pd.read_sql(f"SELECT * FROM {selected_table} ORDER BY Datetime DESC;", actual_conn)
    pred_df = pd.read_sql(f"SELECT * FROM {selected_table}_predictions ORDER BY Datetime DESC;", pred_conn)

    actual_conn.close()
    pred_conn.close()

    # Convert datetime columns to datetime format and remove timezone information
    actual_df['Datetime'] = pd.to_datetime(actual_df['Datetime'], errors='coerce').dt.tz_localize(None)
    pred_df['Datetime'] = pd.to_datetime(pred_df['Datetime'], errors='coerce').dt.tz_localize(None)

    # Drop duplicate entries in the 'Datetime' column by keeping the last entry
    actual_df = actual_df.drop_duplicates(subset=['Datetime'], keep='last')
    pred_df = pred_df.drop_duplicates(subset=['Datetime'], keep='last')

    # Filter data to only include stock market open hours
    market_open = actual_df['Datetime'].dt.time >= pd.to_datetime('09:15').time()
    market_close = actual_df['Datetime'].dt.time <= pd.to_datetime('15:30').time()
    actual_df = actual_df[market_open & market_close]

    pred_open = pred_df['Datetime'].dt.time >= pd.to_datetime('09:15').time()
    pred_close = pred_df['Datetime'].dt.time <= pd.to_datetime('15:30').time()
    pred_df = pred_df[pred_open & pred_close]

    # Combine actual and predicted data for x-axis range slider
    combined_df = pd.concat([actual_df, pred_df])
    combined_df.sort_values(by='Datetime', inplace=True)

    # Convert Timestamps to datetime objects for Streamlit slider
    min_date = combined_df['Datetime'].dt.date.min()
    max_date = combined_df['Datetime'].dt.date.max()
    min_time = time(9, 15)
    max_time = time(15, 30)

    # Streamlit sliders for date and time range selection
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    time_range = st.slider(
        "Select Time Range",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format="HH:mm"
    )

    # Combine date and time range into a single datetime range
    start_datetime = datetime.combine(date_range[0], time_range[0])
    end_datetime = datetime.combine(date_range[1], time_range[1])

    # Filter data based on the selected datetime range from the sliders
    filtered_actual_df = actual_df[
        (actual_df['Datetime'] >= start_datetime) & (actual_df['Datetime'] <= end_datetime)
    ]
    filtered_pred_df = pred_df[
        (pred_df['Datetime'] >= start_datetime) & (pred_df['Datetime'] <= end_datetime)
    ]

    # Plot the candlestick chart using Plotly
    fig = go.Figure()

    # Add actual data to the chart
    fig.add_trace(go.Candlestick(
        x=filtered_actual_df['Datetime'],
        open=filtered_actual_df['Open'],
        high=filtered_actual_df['High'],
        low=filtered_actual_df['Low'],
        close=filtered_actual_df['Close'],
        name='Actual Data',
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='rgba(0,255,0,0.2)',
        decreasing_fillcolor='rgba(255,0,0,0.2)',
    ))

    # Add predicted data to the chart
    fig.add_trace(go.Candlestick(
        x=filtered_pred_df['Datetime'],
        open=filtered_pred_df['Predicted_Open'],
        high=filtered_pred_df['Predicted_High'],
        low=filtered_pred_df['Predicted_Low'],
        close=filtered_pred_df['Predicted_Close'],
        name='Predicted Data',
        increasing_line_color='blue',
        decreasing_line_color='orange',
        increasing_fillcolor='rgba(0,0,255,0.2)',
        decreasing_fillcolor='rgba(255,165,0,0.2)',
    ))

    # Update layout for better visuals
    fig.update_layout(
        title=f"Candlestick Chart for {selected_table}",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M',
            tickmode='auto',
            showgrid=True,
            type='date'
        ),
        yaxis=dict(
            showgrid=True
        ),
        width=1200,  # Set width to fit 12 inches on your screen
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)  # Use container width to ensure the chart uses full width

# Load and display the data when a table is selected
if selected_table:
    load_and_plot_data(selected_table)
