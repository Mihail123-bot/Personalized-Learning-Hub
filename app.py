import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import altair as alt
from PIL import Image
import io
import requests
from streamlit_option_menu import option_menu
import time

# Configure page settings
st.set_page_config(
    page_title="Advanced Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stDownloadButton > button {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Dashboard",
            "Data Analysis",
            "Predictive Analytics",
            "Real-time Monitoring",
            "Settings"
        ],
        icons=[
            'house',
            'graph-up',
            'gear',
            'activity',
            'sliders'
        ],
        menu_icon="cast",
        default_index=0,
    )

# Helper functions
def load_financial_data(symbol, period='1y'):
    """Load financial data using yfinance"""
    data = yf.download(symbol, period=period)
    return data

def create_forecast_model(data):
    """Create and train a RandomForest model for forecasting"""
    df = data.copy()
    df['Target'] = df['Close'].shift(-1)
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df = df.dropna()
    
    features = ['Close', 'MA7', 'MA21', 'RSI']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def calculate_rsi(data, periods=14):
    """Calculate RSI technical indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Main application logic
if selected == "Dashboard":
    # Dashboard layout
    st.title("📊 Advanced Analytics Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Users",
            value="1,234",
            delta="12%"
        )
    
    with col2:
        st.metric(
            label="Revenue",
            value="$50,432",
            delta="-8%"
        )
    
    with col3:
        st.metric(
            label="Conversion Rate",
            value="3.2%",
            delta="0.5%"
        )
    
    with col4:
        st.metric(
            label="Avg. Session Duration",
            value="5m 23s",
            delta="1m 12s"
        )
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Market Overview")
        symbol = st.selectbox(
            "Select Stock Symbol",
            options=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            key='stock_selector'
        )
        
        data = load_financial_data(symbol)
        fig = px.line(
            data,
            y='Close',
            title=f'{symbol} Stock Price'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Real-time Performance")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Server Load', 'Response Time', 'Error Rate']
        )
        
        fig = px.line(
            chart_data,
            title="System Performance Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected == "Data Analysis":
    st.title("🔍 Data Analysis")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv']
    )
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        
        # Data overview
        st.subheader("Data Overview")
        st.write(data.head())
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.write(data.describe())
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = data.corr()
        fig = px.imshow(
            corr,
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Column analysis
        st.subheader("Column Analysis")
        selected_column = st.selectbox(
            "Select column to analyze",
            options=data.columns
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                data,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                data,
                y=selected_column,
                title=f"Box Plot of {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)

elif selected == "Predictive Analytics":
    st.title("🔮 Predictive Analytics")
    
    if st.session_state.data is not None:
        st.subheader("Model Training")
        
        target_column = st.selectbox(
            "Select target column",
            options=st.session_state.data.columns
        )
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Simple example using RandomForest
                X = st.session_state.data.drop(target_column, axis=1)
                y = st.session_state.data[target_column]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                st.session_state.model = model
                st.session_state.predictions = model.predict(X_test)
                
                st.success("Model trained successfully!")
                
                # Model performance visualization
                fig = px.scatter(
                    x=y_test,
                    y=st.session_state.predictions,
                    title="Predicted vs Actual Values"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data in the Data Analysis section first.")

elif selected == "Real-time Monitoring":
    st.title("📈 Real-time Monitoring")
    
    # Simulated real-time data
    chart = st.empty()
    
    if st.button("Start Monitoring"):
        for i in range(100):
            data = pd.DataFrame({
                'time': [datetime.now() - timedelta(seconds=x) for x in range(20)],
                'value': np.random.randn(20).cumsum()
            })
            
            fig = px.line(
                data,
                x='time',
                y='value',
                title="Real-time System Metrics"
            )
            
            chart.plotly_chart(fig, use_container_width=True)
            time.sleep(1)

elif selected == "Settings":
    st.title("⚙️ Settings")
    
    # Theme settings
    st.subheader("Theme Settings")
    theme = st.selectbox(
        "Select theme",
        options=['Light', 'Dark', 'Custom']
    )
    
    if theme == 'Custom':
        primary_color = st.color_picker("Primary Color", "#1f77b4")
        secondary_color = st.color_picker("Secondary Color", "#ff7f0e")
    
    # Notification settings
    st.subheader("Notification Settings")
    email_notifications = st.toggle("Email Notifications")
    push_notifications = st.toggle("Push Notifications")
    
    if email_notifications:
        email = st.text_input("Email Address")
    
    # Data refresh settings
    st.subheader("Data Refresh Settings")
    refresh_interval = st.slider(
        "Data refresh interval (minutes)",
        min_value=1,
        max_value=60,
        value=5
    )
    
    # Export settings
    st.subheader("Export Settings")
    export_format = st.radio(
        "Default export format",
        options=['CSV', 'Excel', 'JSON']
    )
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p>Advanced Analytics Dashboard v1.0</p>
        <p>Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
