import streamlit as st
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import random
import smtplib
import traceback
import joblib
import feedparser
import time
import sqlite3
import streamlit.components.v1 as components
from bs4 import BeautifulSoup 
import plotly.graph_objs as go
from PIL import Image
from pandas.tseries.offsets import BDay
from datetime import timedelta
from email.message import EmailMessage
from db import update_user_password, set_reset_code, verify_reset_code
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from db import add_user, get_user_by_email, verify_user, authenticate_user, get_all_users, delete_user_by_email
from streamlit_autorefresh import st_autorefresh



#--------------------------------------------------------------------------------------------------------#

st.set_page_config(page_title="StockEye - Home", layout="wide")

#--------------------------------------------------------------------------------------------------------#

# Your FMP API key
API_KEY = 'uh4TbAlyMVjqUvM88DzpXEwEiBsHch6N'

# Email credentials for sending verification emails
EMAIL_ADDRESS = 'stockeye.predictor543@gmail.com'      # Replace with your email
EMAIL_PASSWORD = 'yeqgjpyaoaxyoemj'      # Replace with your app password or email password

#--------------------------------------------------------------------------------------------------------#

# Send verification email with code
def send_verification_email(to_email, code):
    msg = EmailMessage()
    msg['Subject'] = 'Your Verification Code'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(f"Your verification code is: {code}")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)


def send_password_reset_email(to_email, reset_code):
    msg = EmailMessage()
    msg['Subject'] = 'Password Reset Code'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(f"Your password reset code is: {reset_code}")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
   

#--------------------------------------------------------------------------------------------------------#

#---------------------------------- FETCH DATA (latest 100) ----------------------------------#

@st.cache_data(ttl=3600)  # Cache data for 1 hour (3600 seconds)
def get_stock_data(symbol, api_key):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=candle&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()

        if "historical" not in data:
            return None

        df = pd.DataFrame(data["historical"])
        df = df.rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date", ascending=True)
        return df.tail(250).reset_index(drop=True)  # Updated to 250 rows

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None




#---------------------------------- MOVING AVERAGES & RSI ----------------------------------#

def calculate_moving_averages(df):
    df['100-day MA'] = df['close'].rolling(window=100).mean()
    df['200-day MA'] = df['close'].rolling(window=200).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df['close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate exponential moving averages of gains and losses
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi

    return df


#---------------------------------- LSTM MODEL ----------------------------------#

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_for_lstm(df, time_step=30):
    data = df[['close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])
        y.append(data_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

@st.cache_resource
def train_lstm_model(df, time_step=30):
    X, y, scaler = prepare_data_for_lstm(df, time_step)
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model, scaler

def predict_lstm(model, scaler, df, time_step=30):
    last_data = df[['close']].values[-time_step:]
    last_data_scaled = scaler.transform(last_data)
    last_data_scaled = np.reshape(last_data_scaled, (1, time_step, 1))
    predicted_price_scaled = model.predict(last_data_scaled)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    return predicted_price[0][0]

#---------------------------------- RANDOM FOREST ----------------------------------#

@st.cache_resource
def train_random_forest(df):
    X = df[['high', 'low', 'close']].values[:-1]
    y = df['close'].shift(-1).dropna().values
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

def predict_random_forest(model, df):
    X_test = df[['high', 'low', 'close']].values[-1].reshape(1, -1)
    return model.predict(X_test)[0]

#---------------------------------- ARIMA MODEL ----------------------------------#

@st.cache_resource
def train_arima(df):
    series = df['close'].dropna()
    if len(series) < 10:
        raise ValueError("Not enough data for ARIMA.")
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def predict_arima(model_fit):
    forecast = model_fit.forecast(steps=1)
    return forecast.iloc[0]


#---------------------------------- MAIN FUNCTION TO USE ----------------------------------#

def run_predictions(symbol):
    api_key = "3KSL9wxMf55F9wrNF4XkAGYGIF9d6fm2"
    df = get_stock_data(symbol, api_key)

    if df is None or df.empty:
        st.error("No data available for the given symbol.")
        return

    # Basic calculations
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
     
     
    st.write("Latest 100 days of stock data:")
    st.dataframe(df.tail())


    # Train & Predict LSTM
    model_lstm, scaler_lstm = train_lstm_model(df)
    pred_lstm = predict_lstm(model_lstm, scaler_lstm, df)

    # Train & Predict Random Forest
    model_rf = train_random_forest(df)
    pred_rf = predict_random_forest(model_rf, df)

    # Train & Predict ARIMA
    model_arima = train_arima(df)
    pred_arima = predict_arima(model_arima)

    # Results
    st.subheader(f"📈 Next Day Prediction for {symbol}")
    st.success(f"LSTM Prediction: {pred_lstm:.2f}")
    st.success(f"Random Forest Prediction: {pred_rf:.2f}")
    st.success(f"ARIMA Prediction: {pred_arima:.2f}")


#----------------------------------------------------------------------------------------------------------------#
            
def login():
    st.title("🔐 Login")

    email = st.text_input("Email") 
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate_user(email, password)
        if user:
            st.session_state['authenticated'] = True
            st.session_state['email'] = user['email']
            st.session_state['role'] = user['role']  
            st.session_state['page'] = 'Home'  
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials or email not verified.")

    if st.button("Forgot Password?"):
        st.session_state['show_forgot_password'] = True
        st.rerun()    
    
    if st.session_state.get('show_forgot_password'):
        forgot_password()
        st.stop()

        

def signup():
    st.title("📝 Signup with Email")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Signup"):
        if get_user_by_email(email):
            st.error("Email already registered.")
        else:
            otp = str(random.randint(100000, 999999))

            # Decide role based on email
            if email == 'stockeye.predictor543@gmail.com':
                role = 'admin'
            else:
                role = 'user'

            added = add_user(email, password, otp, role)  # Pass otp here

            if added:
                try:
                    send_verification_email(email, otp)
                    st.success("Signup successful! Check your email for the verification code.")
                    st.session_state['signup_email'] = email
                    st.session_state['signup_password'] = password
                    st.session_state['awaiting_verification'] = True
                except Exception as e:
                    st.error(f"Failed to send verification email: {e}")
            else:
                st.error("Failed to register user. Try again.")



def forgot_password():
    st.title("🔐 Forgot Password")

    email = st.text_input("Enter your registered email")

    if st.button("Send Reset Code"):
        user = get_user_by_email(email)
        if user:
            reset_code = str(random.randint(100000, 999999))
            st.session_state['reset_code'] = reset_code
            st.session_state['reset_email'] = email
            try:
                send_password_reset_email(email, reset_code)
                st.success("Reset code sent to your email.")
                st.session_state['show_reset_verification'] = True
            except Exception as e:
                st.error(f"Failed to send email. Error: {e}")
        else:
            st.error("Email not found.")

    if st.session_state.get('show_reset_verification'):
        code = st.text_input("Enter the reset code")
        new_password = st.text_input("Enter your new password", type="password")

        if st.button("Reset Password"):
            if code == st.session_state.get('reset_code'):
                from db import update_user_password  # You’ll need to create this function if not present
                if update_user_password(st.session_state['reset_email'], new_password):
                    st.success("Password reset successfully! Please login.")
                    # Cleanup
                    st.session_state.pop('show_forgot_password', None)
                    st.session_state.pop('show_reset_verification', None)
                    st.session_state.pop('reset_code', None)
                    st.session_state.pop('reset_email', None)
                else:
                    st.error("Password update failed.")
            else:
                st.error("Incorrect reset code.")


def verify():
    st.title("✅ Verify Your Email")

    email = st.session_state.get('signup_email', '')
    code = st.text_input(f"Enter the verification code sent to {email}")

    if st.button("Verify"):
        if verify_user(email, code):
            st.success("Email verified! You can now login.")
            st.session_state['awaiting_verification'] = False
            # Optionally auto-login user here or redirect to login
        else:
            st.error("Incorrect verification code.")


#-----------------------------------------------------------------------------------------------------------#            

def show_homepage():
    # Header section with logo and title inside smaller container
    st.markdown(
        """
        <style>
            /* Target the whole app background */
            .stApp {
                background-image: url('https://i.postimg.cc/TPrBbVdB/background-4.jpg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }

            /* Container for header section */
            .header-container {
                background: rgba(255, 255, 255, 0.85) !important;
                border-radius: 12px;
                padding: 20px 15px;
                max-width: 500px;
                margin: 40px auto 30px auto;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            }

            .header {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                margin-bottom: 10px;
            }

            .header img {
                width: 70px;
                height: 70px;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 114, 177, 0.3);
            }

            .header h1 {
                color: #0072B1;
                font-size: 48px;
                font-weight: 700;
                margin: 0;
                user-select: none;
            }

            .title-group {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }

            .title-group h1 {
                margin-bottom: 0;
            }

            .tagline {
                font-size: 14px;
                font-style: italic;
                color: grey;  /* soft violet or choose any */
                margin-top: -20px;
                margin-left: 3px;
                letter-spacing: 0px;
            }


            .subtitle {
                color: #555;
                font-size: 22px;
                text-align: center;
                margin-bottom: 40px;
                font-weight: 500;
            }

            /* New container for features section */
            .features-container {
                background: linear-gradient(135deg, #e0ccf5, #c7aef0); 
                border-radius: 10px;
                padding: 20px 25px;
                max-width: 700px;
                margin: 30px auto 50px auto;
                box-shadow: 0 10px 30px rgba(90, 50, 163, 0.2);
                border-left: 6px solid #5a32a3;
                backdrop-filter: blur(5px);
                transition: all 0.3s ease-in-out;
            }


            /* Style for features heading */
            .features-heading {
                text-align: center;
                font-size: 28px;
                font-weight: 600;
                color: #2e2e2e; 
                margin-bottom: 10px;
                user-select: none;
            }
        </style>

        <div class="header-container">
            <div class="header">
                <img src="https://i.postimg.cc/CMngmqxr/stockeye-logo.png" alt="StockEye Logo" />
                <div class="title-group">
                    <h1>StockEye</h1>
                    <div class="tagline">THE WATCHER</div>
                </div>
            </div>
            <div class="subtitle">Your Personal AI-Powered Stock Market Assistant</div>
        </div>

        """,
        unsafe_allow_html=True,
    )

    # Features overview inside a wider separate container
    st.markdown(
        """
        <div class="features-container">
            <div class="features-heading">🚀 What You Can Do</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use columns with a container approach
    # Wrap the 4 columns inside the same width container by manual padding hack
    col1, col2, col3, col4 = st.columns([1,1,1,1], gap="medium")

    with col1:
        st.info("📉 Predict Stock Prices using ARIMA, LSTM, and Random Forest.")
    with col2:
        st.info("🔍 Analyze stocks with RSI, MA100, MA200 and other indicators.")
    with col3:
        st.info("🔔 Get real-time news updates and plan your trades smartly.")
    with col4:
        st.info("🤖 Ask our AI Chatbot for market insights, tips, or guidance.")


    # Dashboard Navigation Button
    st.markdown("---")
    st.markdown("### 📊 Ready to Explore?")
    if st.button("Go to Dashboard"):
            st.session_state.page = "Dashboard"
            return 
    

    # Footer
    st.markdown("""
        <div class="footer">
            StockEye © 2025 | Version 1.0.0
        </div>
    """, unsafe_allow_html=True)


#-----------------------------------------------------------------------------------------------------------#            


def show_dashboard():
    st.markdown("""
        <style>
        /* Container */
        .dashboard-wrapper {
            max-width: 900px;
            margin: 1rem auto 3rem auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1a202c;
        }
                
        .top-header-box {
            background-color:#f3e8ff;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            padding: 1.5rem 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            max-width: 400px;
            margin: 1rem auto 2rem auto;
            height: 100px;
        }

        .logo-img {
            height: 50px;
            margin-right: 1rem;
        }

        .header-text-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .header-text {
            font-size: 2rem;
            font-weight: 800;
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .subheader-text {
            font-size: 1rem;
            color: #777777;
            font-style: italic;
            margin-top: 4px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        

        /* Headings */
        .dashboard-wrapper h2, .dashboard-wrapper h3 {
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: #2c3e50;
        }

        /* Welcome text */
        .welcome-text {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: #2b6cb0;
            text-align: center;
        }

        /* Input */
        .ticker-input {
            max-width: 300px;
            margin: 0 auto 2rem auto;
            display: block;
        }

        /* Dataframe container */
        .dataframe-container {
            background: #f7fafc;
            padding: 0.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgb(0 0 0 / 0.05);
            margin-bottom: 2rem;
        }

        /* Charts container */
        .chart-container {
            background: #ffffff;
            padding: 0.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgb(0 0 0 / 0.05);
            margin-bottom: 2.5rem;
        }

        /* Predictions container */
        .predictions-wrapper {
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            margin-bottom: 2.5rem;
            flex-wrap: wrap;
        }

        .prediction-card {
            flex: 1 1 250px;
            background: #eaf4ff;
            border-radius: 14px;
            box-shadow: 0 6px 15px rgb(43 108 176 / 0.15);
            padding: 1.2rem 1.5rem;
            transition: transform 0.2s ease-in-out;
            text-align: center;
            cursor: default;
        }

        .prediction-card:hover {
            transform: translateY(-6px);
        }

        .prediction-title {
            font-weight: 700;
            font-size: 1.2rem;
            color: #2b6cb0;
            margin-bottom: 0.7rem;
        }

        .prediction-info {
            font-size: 0.9rem;
            color: #2c3e50;
            margin-bottom: 1rem;
            font-style: italic;
        }

        /* Buttons inside cards */
        .prediction-card button {
            background-color: #2b6cb0;
            color: white;
            border: none;
            padding: 0.45rem 1.2rem;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease-in-out;
            margin-top: 0.8rem;
        }

        .prediction-card button:hover {
            background-color: #1a4280;
        }

        /* News Button */
        .news-button {
            display: block;
            width: 180px;
            margin: 0 auto 2rem auto;
            padding: 0.6rem;
            background-color: #3182ce;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            border: none;
            transition: background-color 0.25s ease-in-out;
            text-align: center;
        }

        .news-button:hover {
            background-color: #225ea8;
        }

        /* Logout Expander */
        .logout-expander .streamlit-expanderHeader {
            font-weight: 700 !important;
            color: #c53030 !important;
            font-size: 1.1rem !important;
        }

        .logout-content {
            background-color: #fed7d7;
            padding: 1.2rem 1rem;
            border-radius: 12px;
            color: #742a2a;
            font-weight: 600;
        }

        .logout-buttons {
            display: flex;
            justify-content: space-around;
            margin-top: 1rem;
        }

        .logout-buttons button {
            width: 110px;
            padding: 0.5rem 0;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            font-size: 1rem;
        }

        .logout-yes {
            background-color: #c53030;
            color: white;
        }

        .logout-yes:hover {
            background-color: #822020;
        }

        .logout-no {
            background-color: #718096;
            color: white;
        }

        .logout-no:hover {
            background-color: #4a5568;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Logo and Headline with subheadline at the top
    st.markdown("""
        <div class="top-header-box">
            <img src="https://i.postimg.cc/CMngmqxr/stockeye-logo.png" class="logo-img" alt="StockEye Logo">
            <div class="header-text-container">
                <span class="header-text">StockEye</span>
                <span class="subheader-text">THE WATCHER</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


    st.markdown(f'<div class="welcome-text">📊 Welcome, {st.session_state["email"]}!</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Go to Home button
    if st.button("Go to Home"):
        st.session_state.page = "Home"
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    ticker = st.text_input(
        "Enter Stock Ticker Symbol",
        "AAPL",
        help="Enter the stock ticker symbol, e.g. AAPL for Apple.",
        key="ticker_input"
    ).strip().upper()

    # 🛡️ Guard clause: Don't proceed if ticker is empty
    if not ticker:
        st.warning("⚠️ Please enter a valid stock ticker symbol.")
        st.stop()

    

    def get_realtime_price(symbol):
        url = f'https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={API_KEY}'
        try:
            response = requests.get(url)
            data = response.json()
            if data and isinstance(data, list):
                return data[0]['price']
        except Exception as e:
            return None
        return None
    
    
    # Initialize session states to store latest price and chart data
    if "latest_price" not in st.session_state:
        st.session_state.latest_price = None

    if "chart_data" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame(columns=['Price'])

    # Button to refresh price
    if st.button("Refresh Price") and ticker:
        new_price = get_realtime_price(ticker)
        if new_price is not None:
            st.session_state.latest_price = new_price
        else:
            st.warning("⚠️ Could not fetch real-time price. Please try again or check the symbol.")

    # Show live price container if price available
    if ticker and st.session_state.latest_price is not None:
        st.markdown(
            f"""
            <div style="
                max-width: 250px;
                margin: 0 0 1.5rem 0;
                background: linear-gradient(135deg, #232526 0%, #414345 100%);
                border-radius: 12px;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                color: #ffffff;
                padding: 1rem 1.2rem;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: left;
            ">
                <div style="font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">
                    Live Price: <strong>{ticker.upper()}</strong>
                </div>
                <div style="font-size: 1.8rem; font-weight: 700;">
                    ${st.session_state.latest_price:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

    # Button to refresh chart
    if st.button("Refresh Chart") and ticker:
        # Fetch 30 price points for the chart (simulate by fetching repeatedly)
        prices = []
        for _ in range(30):
            price = get_realtime_price(ticker)
            if price:
                prices.append(price)
        if prices:
            st.session_state.chart_data = pd.DataFrame({'Price': prices})
        else:
            st.warning("⚠️ Could not fetch chart data. Please try again.")



    if ticker and not st.session_state.chart_data.empty:
        svg_icon = '''
        <svg xmlns="http://www.w3.org/2000/svg" 
            viewBox="0 0 512 512" 
            width="24" height="24" 
            style="vertical-align: middle; margin-right: 8px; fill: #4CAF50;">
            <path d="M64 64c0-17.7-14.3-32-32-32S0 46.3 0 64L0 400c0 44.2 35.8 80 80 80l400 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L80 416c-8.8 0-16-7.2-16-16L64 64zm406.6 86.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L320 210.7l-57.4-57.4c-12.5-12.5-32.8-12.5-45.3 0l-112 112c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L240 221.3l57.4 57.4c12.5 12.5 32.8 12.5 45.3 0l128-128z"/>
        </svg>
        '''
        st.markdown(f'<h3 style="display: flex; align-items: center;">{svg_icon} Live Line Chart for {ticker.upper()}</h3>', unsafe_allow_html=True)


        # Create an interactive Plotly line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.chart_data['Price'],
            mode='lines+markers',
            name='Live Price',
            line=dict(color='cyan', width=2),
            marker=dict(size=5, color='blue', symbol='circle')
        ))

        fig.update_layout(
            template='plotly_dark',
            title=f"Real-Time Price Chart: {ticker.upper()}",
            xaxis_title="Time (ticks)",
            yaxis_title="Price (USD)",
            margin=dict(l=40, r=20, t=40, b=20),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


    # Fetch and store stock_data in session_state if not already fetched
    if ticker:
        if 'stock_data' not in st.session_state or st.session_state.ticker_symbol != ticker:
            try:
                fetched_data = get_stock_data(ticker, API_KEY)
                if fetched_data is None or fetched_data.empty:
                    st.warning("⚠️ No data returned for the given ticker.")
                    return
                st.session_state.stock_data = fetched_data
                st.session_state.ticker_symbol = ticker  # track current ticker
            except ValueError as ve:
                err_msg = str(ve)
                if "No historical data found" in err_msg:
                    st.warning("⚠️ Ticker incorrect or unavailable.")
                    return
                elif "status code 403" in err_msg:
                    st.warning("🚫 Access to this stock's data is restricted.")
                    return
                else:
                    raise ve

        stock_data = st.session_state.stock_data 


     # Ensure historical data is available
    if ticker and not st.session_state.chart_data.empty:
        st.markdown(f'''
            <h3 style="display: flex; align-items: center; gap: 8px;">
                <img src="https://i.postimg.cc/05HBQpPv/candlestick-chart.png" width="40" height="40" style="vertical-align: middle;" />
                Live Candlestick Chart for {ticker.upper()}
            </h3>
        ''', unsafe_allow_html=True)

    if st.button("🔄 Refresh Chart"):
        st.rerun()

    # Convert 'Date' to datetime and rename
    stock_data.rename(columns={'Date': 'date'}, inplace=True)
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # Plot candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['date'],
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    )])

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


    # If no price fetched yet, get initial price on load
    if ticker and st.session_state.latest_price is None:
        initial_price = get_realtime_price(ticker)
        if initial_price is not None:
            st.session_state.latest_price = initial_price


   

    if ticker:
        try:
            try:
                stock_data = get_stock_data(ticker, API_KEY)
            except ValueError as ve:
                err_msg = str(ve)
                if "No historical data found" in err_msg:
                    st.warning("⚠️ The ticker symbol seems incorrect or data is not available. Please verify and try again (e.g., use symbols like AAPL, INFY.BSE, or TATAMOTORS.NSE).")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return
                elif "status code 403" in err_msg:
                    st.warning("🚫 Access to this stock's data is currently restricted or unavailable. We are working on adding support for more stocks soon!")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return
                else:
                    raise ve

            if stock_data is None or stock_data.empty:
                st.warning("⚠️ No data returned for the given ticker. Please check the symbol and try again.")
                st.markdown("</div>", unsafe_allow_html=True)
                return




            # Display raw data
            st.subheader(f"📈 Stock Data for {ticker}")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(stock_data.tail(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Moving Averages chart
            stock_data = calculate_moving_averages(stock_data)
            st.subheader("📊 Moving Averages (100-day & 200-day)")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.line_chart(stock_data[['close', '100-day MA', '200-day MA']])
            st.markdown('</div>', unsafe_allow_html=True)

            # RSI chart
            stock_data = calculate_rsi(stock_data)
            st.subheader("📉 Relative Strength Index (RSI)")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.line_chart(stock_data[['RSI']])
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Predictions Section ---
            st.markdown("---")
            st.subheader("🤖 Predictions")

            st.markdown('<div class="predictions-wrapper">', unsafe_allow_html=True)

            # LSTM card
            with st.container():
                st.markdown("""
                    <div class="prediction-card">
                        <div class="prediction-title">LSTM</div>
                        <div class="prediction-info">Uses sequential past data patterns to predict the next closing price with a deep learning LSTM model.</div>
                """, unsafe_allow_html=True)
                lstm_expanded = st.checkbox("ℹ️ Info - LSTM", key="lstm_info")
                if lstm_expanded:
                    st.info("Uses sequential past data patterns to predict the next closing price with a deep learning LSTM model.")
                if st.button("Run LSTM Prediction"):
                    try:
                        model, scaler = train_lstm_model(stock_data)
                        predicted_price = predict_lstm(model, scaler, stock_data)
                        st.success(f"LSTM Predicted Price: ${predicted_price:.2f}")
                    except Exception:
                        st.error("LSTM Prediction failed.")
                        st.text(traceback.format_exc())
                st.markdown("</div>", unsafe_allow_html=True)

            # Random Forest card
            with st.container():
                st.markdown("""
                    <div class="prediction-card">
                        <div class="prediction-title">Random Forest</div>
                        <div class="prediction-info">Predicts next closing price using high, low, and close features via a Random Forest regression.</div>
                """, unsafe_allow_html=True)
                rf_expanded = st.checkbox("ℹ️ Info - Random Forest", key="rf_info")
                if rf_expanded:
                    st.info("Predicts next closing price using high, low, and close features via a Random Forest regression.")
                if st.button("Run Random Forest Prediction"):
                    try:
                        model_rf = train_random_forest(stock_data)
                        predicted_price_rf = predict_random_forest(model_rf, stock_data)
                        st.success(f"RF Predicted Price: ${predicted_price_rf:.2f}")
                    except Exception:
                        st.error("Random Forest Prediction failed.")
                        st.text(traceback.format_exc())
                st.markdown("</div>", unsafe_allow_html=True)

            # ARIMA card
            with st.container():
                st.markdown("""
                    <div class="prediction-card">
                        <div class="prediction-title">ARIMA</div>
                        <div class="prediction-info">Forecasts the next closing price by modeling trends and seasonality in time series data with ARIMA.</div>
                """, unsafe_allow_html=True)
                arima_expanded = st.checkbox("ℹ️ Info - ARIMA", key="arima_info")
                if arima_expanded:
                    st.info("Forecasts the next closing price by modeling trends and seasonality in time series data with ARIMA.")
                if st.button("Run ARIMA Prediction"):
                    try:
                        model_arima = train_arima(stock_data)
                        predicted_price_arima = predict_arima(model_arima)
                        st.success(f"ARIMA Predicted Price: ${predicted_price_arima:.2f}")
                    except Exception:
                        st.error("ARIMA Prediction failed.")
                        st.text(traceback.format_exc())
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # News button
            st.markdown("---")
            st.markdown("""
                    <h2 style='text-align: center; color: #4B89DC; font-weight: bold;'>
                        Time to review the market analysis
                    </h2>
                    """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3,1,3])  # middle column is wider

            with col2:
                if st.button("Show Latest News"):
                    st.session_state.page = "News"


        except Exception as e:
            st.error(f"Error fetching data: {e}")

    # Logout section with expander and styled buttons
    with st.expander("Logout", expanded=False):
        st.markdown('<div class="logout-content">Are you sure you want to logout?</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            if st.button("Yes, Logout", key="logout_yes"):
                st.session_state.clear()
                st.rerun()
        with cols[1]:
            if st.button("No, Stay Logged In", key="logout_no"):
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

#-----------------------------------------------------------------------------------------------------------------------#            

# Define your tradingview function
def tradingview(symbol: str = "NSE:TCS"):
    widget = f"""
    <style>
      .tradingview-widget-container {{
        max-width: 1000px;
        margin: 20px auto;
        padding: 15px;
        background: linear-gradient(135deg, #1e2a47, #243b66);
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        border: 2px solid #4169e1;
      }}

      /* Responsive width */
      @media (max-width: 1100px) {{
        .tradingview-widget-container {{
          width: 95vw;
          padding: 10px;
        }}
      }}

      /* Optional: Hover effect */
      .tradingview-widget-container:hover {{
        box-shadow: 0 12px 28px rgba(65, 105, 225, 0.7);
        border-color: #6495ed;
        transition: all 0.3s ease-in-out;
      }}
    </style>

    <div class="tradingview-widget-container">
      <div id="tradingview_widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": 500,
          "symbol": "{symbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#1e2a47",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "container_id": "tradingview_widget"
        }});
      </script>
    </div>
    """
    components.html(widget, height=570, width=1000)



#-----------------------------------------------------------------------------------------------------------------------#            


def settings_module():
    # Inject custom CSS styles
    st.markdown(
        """
        <style>
        /* Overall container styling */
        .settings-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 10px 90px;
            background: #2A3439;
            border-radius: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Title styling */
        .settings-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: white;
        }

        /* Section headers */
        .settings-header {
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
            color: white;
            border-bottom: 2px solid #efefef;
            padding-bottom: 0.25rem;
        }

        /* Captions under inputs */
        .caption {
            font-size: 0.85rem;
            color: #777777;
            margin-top: -12px;
            margin-bottom: 15px;
            font-style: italic;
        }

        /* Style disabled inputs */
        input[disabled] {
            background-color: #2C3E50 !important; 
            color: #ECEFF1 !important;  
            border: 1px solid #455A64 !important; 
            border-radius: 6px !important;
            padding: 8px !important;
            }

            
        /* Save button style override */
        div.stButton > button:first-child {
            background-color: #4a90e2;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            transition: background-color 0.3s ease;
            border: none;
            box-shadow: 0 4px 12px rgba(74,144,226,0.4);
        }
        div.stButton > button:first-child:hover {
            background-color: #357ABD;
            box-shadow: 0 6px 16px rgba(53,122,189,0.6);
        }

        /* Radio button horizontal layout */
        .stRadio > div {
            flex-direction: row !important;
            gap: 1.5rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="settings-container">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">⚙️ Settings</div>', unsafe_allow_html=True)

    # Get the logged-in user's email
    email = st.session_state.get("email", "")

    # Fetch current user settings from DB
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, theme, notifications FROM users WHERE email = ?", (email,))
    user_data = cursor.fetchone()
    conn.close()

    current_name = user_data[0] if user_data else "User Name"
    current_theme = user_data[1] if user_data else "Light"
    notifications_enabled = bool(user_data[2]) if user_data else True

    with st.form("settings_form"):
        st.markdown('<div class="settings-header">👤 Profile Information</div>', unsafe_allow_html=True)

        st.text_input("Email", value=email, disabled=True)
        st.markdown('<div class="caption">✉️ Your email address cannot be changed.</div>', unsafe_allow_html=True)

        name = st.text_input("Name", value=current_name, max_chars=30, help="Enter your display name.")

        st.markdown('<div class="settings-header">🎨 Theme Selection</div>', unsafe_allow_html=True)

        theme = st.radio(
            "Choose your theme",
            options=["Light", "Dark"],
            index=0 if current_theme == "Light" else 1,
            horizontal=True,
        )

        st.markdown('<div class="settings-header">🔔 Notifications</div>', unsafe_allow_html=True)

        notifications = st.checkbox("Enable Notifications", value=notifications_enabled)

        st.markdown("")
        submitted = st.form_submit_button("💾 Save Settings")

        if submitted:
            # Update user data in database
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users
                SET name = ?, theme = ?, notifications = ?
                WHERE email = ?
                """,
                (name, theme, int(notifications), email),
            )
            conn.commit()
            conn.close()

            # Update session state
            st.session_state["name"] = name
            st.session_state["theme"] = theme
            st.session_state["notifications"] = notifications

            st.success("✅ Settings saved successfully!")

    st.markdown("</div>", unsafe_allow_html=True)


#-----------------------------------------------------------------------------------------------------------------------#            


def support_module():
    st.markdown("""
    <style>
    .support-container {
        max-width: 750px;
        margin: 20px auto;
        padding: 10px 80px;
        background: #3CB371;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #FFFFFF;
    }
    .support-container h2 {
        text-align: center;
        margin-bottom: 25px;
        font-weight: 700;
        font-size: 2.2rem;
    }
    details {
        background-color: #3A4550;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        cursor: pointer;
    }
    summary {
        font-weight: 600;
        font-size: 1.1rem;
    }
    summary::-webkit-details-marker {
        display: none;
    }
    details[open] {
        background-color: #4A5663;
    }
    .contact-info-box {
        background-color: #2A3439;
        border-left: 5px solid #00BFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #E6E6E6;
        margin-top: 15px;
    }
    .contact-info-box p {
        margin: 10px 0;
        font-size: 15px;
    }
    .contact-info-box strong {
        color: #40E0D0;
    }
    .update-note {
        background-color: #34454D;
        border-left: 5px solid #FF8C00;
        padding: 16px 20px;
        border-radius: 10px;
        margin-top: 30px;
        font-style: italic;
        color: #E0E0E0;
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="support-container">', unsafe_allow_html=True)

    st.markdown("## 🛠️ Support / Help")

    # FAQs Section
    st.markdown("### 📚 Frequently Asked Questions (FAQs)")
    faqs = {
        "How do I reset my password?":
            "Go to the login page and click 'Forgot Password'. Follow the instructions sent to your registered email.",
        "How can I change my email address?":
            "For security reasons, email changes are handled manually. Contact our support team for assistance.",
        "Why am I not receiving notifications?":
            "Ensure notifications are enabled in your Settings and check your Spam/Junk folder. If issues persist, contact support.",
        "How do I delete my account?":
            "Send an account deletion request to [stockeye.predictor543@gmail.com](mailto:stockeye.predictor543@gmail.com).",
        "Can I use this on mobile?":
            "Yes, the web app is optimized for both desktop and mobile devices.",
        "How often is the data updated?":
            "Stock data is fetched in near real-time depending on your internet speed and the data provider's API."
    }

    for question, answer in faqs.items():
        st.markdown(f"<details><summary>{question}</summary><p>{answer}</p></details>", unsafe_allow_html=True)

    st.markdown("---")

    # Contact Information
    st.markdown("### 📞 Contact Our Team")
    st.markdown("""
    <div class="contact-info-box">
        <p><strong>📧 Email:</strong> stockeye.predictor543@gmail.com</p>
        <p><strong>📱 Phone:</strong> +1 (555) 123-4567</p>
        <p><strong>💬 Live Chat:</strong> Available Mon–Fri, 9:00 AM – 6:00 PM (IST)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Troubleshooting Guides
    st.markdown("### 🧩 Troubleshooting Guides")
    troubleshooting_guides = {
        "The app is running slowly": "Clear your browser cache or restart the app. Ensure a stable internet connection.",
        "Data is not updating": "Try refreshing the page. If the problem continues, logout and log back in.",
        "Unable to login": "Check your email and password. Use 'Forgot Password' to reset it if needed.",
        "Charts are not displaying correctly": "Ensure your browser supports JavaScript and you have a good internet connection.",
        "Prediction not showing": "Wait for the model to load and ensure your date range selection is valid."
    }

    for issue, solution in troubleshooting_guides.items():
        st.markdown(f"<details><summary>{issue}</summary><p>{solution}</p></details>", unsafe_allow_html=True)

    st.markdown("---")

    # User Feedback
    st.markdown("### ✍️ Feedback & Suggestions")
    st.markdown("We value your feedback to improve your experience. Please feel free to send us your suggestions or issues at [stockeye.predictor543@gmail.com](mailto:stockeye.predictor543@gmail.com).")

    # Final Note
    st.markdown("""
    <div class="update-note">
        <strong>Note:</strong> Our support resources are actively being improved. More comprehensive guides, real-time chat support, and automated issue tracking will be available in future updates.<br><br>
        🔔 If you're not receiving important alerts, ensure notifications are enabled in your settings and check your spam folder.  
        For unresolved issues, please don't hesitate to reach out to our support team.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


#-----------------------------------------------------------------------------------------------------------------------#            


def fetch_google_news(keyword="stock market"):
    keyword = keyword.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={keyword}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    return feed.entries[:5]

def show_google_news():
    st.markdown("""
        <style>
        .news-wrapper {
            max-width: 500px;
            margin: 0 auto;
            border-radius: 40px;    
            padding: 2rem 1rem;
            background-color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .news-heading {
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }

        .news-card {
            background: #e6ffed;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            transition: transform 0.2s;
        }

        .news-card:hover {
            transform: translateY(-4px);
        }

        .news-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #2b6cb0;
            text-decoration: none;
        }

        .news-title:hover {
            text-decoration: underline;
        }

        .news-time {
            font-size: 0.9rem;
            color: #718096;
            margin-top: 0.3rem;
            margin-bottom: 0.8rem;
        }

        .news-summary {
            font-size: 1rem;
            color: #2d3748;
            line-height: 1.6;
        }

        .news-input {
            text-align: center;
            margin-bottom: 1.5rem;
        }

        </style>

        <div class="news-wrapper">
            <div class="news-heading">🗞️ Latest News Articles</div>
    """, unsafe_allow_html=True)

    keyword = st.text_input("Enter topic (e.g., stock market, Nifty, Reliance)", "stock market", key="news_topic")

    articles = fetch_google_news(keyword)

    if not articles:
        st.info("No news found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for article in articles:
        summary = BeautifulSoup(article.get("summary", ""), "html.parser").get_text()
        st.markdown(f"""
            <div class="news-card">
                <a class="news-title" href="{article.link}" target="_blank">{article.title}</a>
                <div class="news-time">🕒 Published on {article.published}</div>
                <div class="news-summary">{summary}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


#-----------------------------------------------------------------------------------------------------------------------#            

# Simple FAQ dictionary for chatbot responses

def show_chatbot_page():
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 30px;
        color: white;
    '>
        <h2 style='margin: 0; font-size: 32px;'>🤖 StockEye Chatbot</h2>
        <p style='margin-top: 10px; font-size: 16px;'>Chat with me about stock indicators, dashboard usage, or predictions!</p>
    </div>
    """, unsafe_allow_html=True)


    # FAQ Dictionary
    faq = {
        "hello": "Hi there! How can I assist you with your stock market queries today?",
        "how are you?": "I'm doing great and ready to assist you!",
        "what is stockeye?": "StockEye is your smart assistant for stock market analysis and prediction.",
        "how does stockeye work?": "StockEye uses AI models like LSTM, ARIMA, and Random Forest to predict stock trends.",
        "can you predict stock prices?": "Yes, I provide predictions using various models and indicators like RSI, MA100, MA200.",
        "what is rsi?": "RSI (Relative Strength Index) measures the speed and change of price movements.",
        "what are moving averages?": "Moving averages smooth out price data over time. We use 100-day and 200-day MAs.",
        "how to use dashboard?": "Enter a stock ticker, then view charts, indicators, and predictions.",
        "how to login?": "Go to the Login page and enter your email and password.",
        "reset password?": "Click 'Forgot Password' on the login page and follow the instructions.",
        "is this real-time?": "No, it's not real-time tick data, but updated regularly.",
        "thank you": "You're welcome! Feel free to ask more.",
        "bye": "Goodbye! Have a great day!",
        "what is lstm?": "LSTM is a neural network model useful for time series prediction like stock prices.",
        "what is arima?": "ARIMA is a statistical model for time series forecasting.",
        "what is random forest?": "Random Forest builds multiple decision trees to improve prediction accuracy.",
        "how accurate are predictions?": "Predictions are based on historical data and models; they provide guidance but are not guaranteed.",
        "can i trust stockeye for investing?": "StockEye aids analysis but you should always do your own research or consult a financial advisor.",
    }

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # FAQ Section
    with st.expander("💬 What Can I Ask?", expanded=False):
        st.markdown("You can ask me questions like:")

        categories = {
            "📈 StockEye Features": ["stockeye", "real-time", "dashboard", "investing"],
            "🔍 Technical Indicators": ["rsi", "moving averages", "ma100", "ma200", "lstm", "arima", "random forest"],
            "🔐 Login & Access": ["login", "reset password", "forgot password"],
            "🤖 Chat Examples": ["hello", "thank you", "bye", "how are you"]
        }

        for cat, keywords in categories.items():
            st.markdown(f"#### {cat}")
            found = False
            for question in faq.keys():
                q_lower = question.lower()
                if any(kw in q_lower for kw in keywords):
                    st.markdown(f"- {question.capitalize()}")
                    found = True
            if not found:
                st.markdown("- _No questions available in this category._")

    # Enhanced CSS for a more professional look
    st.markdown("""
    <style>
    .chat-container {
        background: linear-gradient(to right, #667eea, #764ba2);
        padding: 20px;
        border-radius: 15px;
        max-height: 420px;
        overflow-y: auto;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.07);
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 30px;
    }
    .user-message, .bot-message {
        display: flex;
        align-items: flex-end;
        margin-bottom: 12px;
    }
    .user-message { justify-content: flex-end; }
    .bot-message { justify-content: flex-start; }
    .user-bubble, .bot-bubble {
        padding: 12px 18px;
        border-radius: 20px;
        max-width: 70%;
        font-size: 15px;
        line-height: 1.5;
    }
    .user-bubble {
        background-color: #2563eb;
        color: white;
        border-radius: 20px 20px 4px 20px;
    }
    .bot-bubble {
        background-color: #e2e8f0;
        color: #1a202c;
        border-radius: 20px 20px 20px 4px;
    }
    .avatar {
        width: 32px;
        height: 32px;
        margin: 0 10px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0 0 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.markdown(
                f'''
                <div class="user-message">
                    <div class="user-bubble">{message}</div>
                    <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png" class="avatar">
                </div>
                ''', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''
                <div class="bot-message">
                    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" class="avatar">
                    <div class="bot-bubble">{message}</div>
                </div>
                ''', unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input("Your message:", key="chat_input", placeholder="Type your question and press Enter")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        # Bot response
        with st.spinner("🤖 StockEye is typing..."):
            time.sleep(1.2)
            lower_input = user_input.lower().strip()
            bot_response = faq.get(lower_input, "🤖 Sorry, I didn’t understand that. Ask about RSI, MA, or predictions.")
            st.session_state.chat_history.append(("bot", bot_response))

        # Clear input
        del st.session_state["chat_input"]
        st.rerun()


#-----------------------------------------------------------------------------------------------------------------------#


def show_logout_confirmation():
    st.markdown(
        """
        <style>
        /* Full screen flex container */
        .logout-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;  /* full viewport height */
            background-color: #fff8e1;  /* subtle light background */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Card container with shadow */
        .logout-card {
            background: #e3f2fd;
            padding: 3rem 4rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 400px;
            text-align: center;
        }

        /* Title styling */
        .logout-card h1 {
            font-weight: 700;
            color: black;
            margin-bottom: 1rem;
            font-size: 2.2rem;
        }

        /* Paragraph styling */
        .logout-card p {
            color: #555;
            margin-bottom: 1.8rem;
            font-size: 1.05rem;
            line-height: 1.5;
        }

        /* Style Streamlit buttons to match */
        div.stButton > button {
            padding: 0.55rem 1.8rem !important;
            font-size: 1rem !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
            border: none !important;
            user-select: none !important;
            transition: background-color 0.25s ease !important;
        }

        /* Confirm button style */
        div.stButton > button[style*="background-color: rgb(37, 166, 91)"] {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        div.stButton > button[style*="background-color: rgb(37, 166, 91)"]:hover {
            background-color: #45a049 !important;
        }

        /* Cancel button style */
        div.stButton > button[style*="background-color: rgb(238, 238, 238)"] {
            background-color: #e0e0e0 !important;
            color: #333 !important;
            margin-left: 1.2rem !important;
        }
        div.stButton > button[style*="background-color: rgb(238, 238, 238)"]:hover {
            background-color: #cacaca !important;
        }

        /* Buttons container (Streamlit columns) inside card, center horizontally */
        .buttons-container {
            display: flex;
            justify-content: center;
            margin-top: -1rem;
        }
        </style>

        <div class="logout-wrapper">
            <div class="logout-card">
                <h1>Logout Confirmation.</h1>
                <p>Are you sure you want to logout?</p>
                <p>We hope this platform helped you analyze the stock market and make better predictions.<br>Thank you for using our prediction tool!</p>
                <p style="color:black; font-size:0.95rem;"><em>Feel free to visit again anytime. You're always welcome!</em></p>
                <div class="buttons-container">
                    <!-- Buttons will be rendered by Streamlit columns below -->
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Render buttons inside the same card visually
    col1, col2 = st.columns([1,1])
    with col1:
        confirm = st.button("✅ Confirm Logout", key="confirm_logout")
    with col2:
        cancel = st.button("❌ Cancel", key="cancel_logout")

    if confirm:
        st.session_state['authenticated'] = False
        st.session_state['email'] = ''
        st.session_state['role'] = 'user'
        st.session_state['page'] = 'Login'
        st.success("You have been logged out successfully. Please refresh the page to login again.")

    elif cancel:
        st.session_state['page'] = 'Dashboard'


#-----------------------------------------------------------------------------------------------------------------------#

import math

def show_admin_panel():

    st.markdown("""
        <style>
            .top-header-wrapper {
                display: flex;
                justify-content: center;
                margin-top: 2px;
                margin-bottom: 5px;
            }
            .top-header-box {
                display: flex;
                align-items: center;
                background-color:  #e8f8f5;
                padding: 30px 80px;
                border-radius: 18px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
            }
            .logo-img {
                height: 70px;
                margin-right: 15px;
            }
            .header-text-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .header-text {
                font-size: 40px;
                font-weight: 700;
                color: black;
            }
            .subheader-text {
                font-size: 12px;
                color: #7f8c8d;
                letter-spacing: 1px;
            }
        </style>
        <div class="top-header-wrapper">
            <div class="top-header-box">
                <img src="https://i.postimg.cc/CMngmqxr/stockeye-logo.png" class="logo-img" alt="StockEye Logo">
                <div class="header-text-container">
                    <span class="header-text">StockEye</span>
                    <span class="subheader-text">THE WATCHER</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Header Section
    st.markdown(f"""
    <div style='background-color: #e0f7fa; padding: 20px; border-radius: 10px; border-left: 6px solid #00796b;'>
        <h2 style='color: #004d40;'>👨‍💼 Admin Dashboard</h2>
        <p style='color: #311b92;'>Welcome Admin: <strong>{st.session_state.get('email', 'Admin')}</strong></p>
    </div>
    """, unsafe_allow_html=True)


    users = get_all_users()
    if not users:
        st.warning("No users found.")
        return

    # Statistics Section
    st.markdown("## 📊 User Statistics")
    total_users = len(users)
    verified_users = sum(1 for u in users if u[1] == 1)
    unverified_users = total_users - verified_users
    admin_users = sum(1 for u in users if u[2] == 'admin')
    normal_users = total_users - admin_users

    stat_style = """
    <style>
        .metric-box {
            background-color: #e3f2fd; 
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(30, 136, 229, 0.15);
            border-left: 5px solid #1e88e5;
        }
        .metric-label {
            font-size: 14px;
            color: black; 
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: black; 
        }
    </style>
"""
    st.markdown(stat_style, unsafe_allow_html=True)


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-box'><div class='metric-label'>Total Users</div><div class='metric-value'>{}</div></div>".format(total_users), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'><div class='metric-label'>Verified</div><div class='metric-value'>{}</div></div>".format(verified_users), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'><div class='metric-label'>Unverified</div><div class='metric-value'>{}</div></div>".format(unverified_users), unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-box'><div class='metric-label'>Admins</div><div class='metric-value'>{}</div></div>".format(admin_users), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 👥 User Management")

    view_mode = st.radio("View Mode", ["Card View", "Table View"], horizontal=True)
    search_email = st.text_input("🔍 Search user by email")

    # Custom dark-themed card styling
    user_card_style = """
    <style>
    .user-card {
        background-color: #1f1f2e;
        border-left: 5px solid #00bcd4;
        padding: 15px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 188, 212, 0.15);
    }
    .user-email {
        font-weight: 600;
        color: #00e6e6;
        font-size: 16px;
    }
    .user-meta {
        color: #cfcfcf;
        font-size: 14px;
    }
    .role-admin {
        background-color: #e74c3c;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        display: inline-block;
    }
    .role-user {
        background-color: #2980b9;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        display: inline-block;
    }
    </style>
    """
    st.markdown(user_card_style, unsafe_allow_html=True)

    filtered_users = [u for u in users if search_email.lower() in u[0].lower()] if search_email else users

    if not filtered_users:
      st.warning("No matching users found.")
    else:
     if view_mode == "Card View":
        import math

        users_per_page = 20
        total_pages = math.ceil(len(filtered_users) / users_per_page)

        if "page" not in st.session_state:
            st.session_state.page = 1
        else:
            # Ensure page is int
            try:
                st.session_state.page = int(st.session_state.page)
            except Exception:
                st.session_state.page = 1

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous") and st.session_state.page > 1:
                st.session_state.page -= 1
        with col2:
            st.markdown(f"<p style='text-align:center; font-weight:bold;'>Page {st.session_state.page} of {total_pages}</p>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️") and st.session_state.page < total_pages:
                st.session_state.page += 1

        start_idx = (st.session_state.page - 1) * users_per_page
        end_idx = start_idx + users_per_page
        current_users = filtered_users[start_idx:end_idx]

        for i, user in enumerate(current_users):
            email, verified, role = user

            with st.container():
                st.markdown("<div class='user-card'>", unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                with col1:
                    st.markdown(f"<div class='user-email'>{email}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("✅" if verified else "❌")
                with col3:
                    role_class = "role-admin" if role.lower() == "admin" else "role-user"
                    st.markdown(f"<span class='{role_class}'>{role.capitalize()}</span>", unsafe_allow_html=True)

                with col4:
                    if st.button("🗑️ Delete", key=f"del_{i}_{st.session_state.page}"):
                        delete_user_by_email(email)
                        st.success(f"Deleted user: {email}")
                        st.rerun()

                with col5.expander("Details"):
                    st.write(f"**Email**: {email}")
                    st.write(f"**Verified**: {'Yes' if verified else 'No'}")
                    st.write(f"**Role**: {role.capitalize()}")

                st.markdown("</div>", unsafe_allow_html=True)

     elif view_mode == "Table View":
        user_df = pd.DataFrame(filtered_users, columns=["Email", "Verified", "Role"])
        user_df["Verified"] = user_df["Verified"].apply(lambda x: "Yes" if x == 1 else "No")
        user_df["Role"] = user_df["Role"].str.capitalize()
        st.dataframe(user_df, use_container_width=True)

    


    st.markdown("---")
    st.caption("🛡️ Admin Panel | StockEye - The Watcher")


#-----------------------------------------------------------------------------------------------------------------------#            


def main():
    # Initialize session state variables if not already set
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['email'] = ''
        st.session_state['role'] = 'user'
        st.session_state['awaiting_verification'] = False
        st.session_state['page'] = 'Login'

    # Determine sidebar menu based on auth & role
    if st.session_state['authenticated']:
        if st.session_state['role'] == 'admin':
            menu = ["Home", "Admin Panel", "Dashboard", "News","Settings", "Chatbot","Support", "Logout"]
        else:
            menu = ["Home", "Dashboard", "News","Candlechart", "Settings", "Chatbot","Support", "Logout"]
    elif st.session_state.get('awaiting_verification', False):
        menu = ["Verify Email"]
    else:
        menu = ["Login", "Signup"]

    # Set default index for selectbox based on current page
    if st.session_state.page in menu:
        default_index = menu.index(st.session_state.page)
    else:
        default_index = 0
        st.session_state.page = menu[0]

    # Sidebar menu selectbox (syncs with st.session_state.page)
    choice = st.sidebar.selectbox("📂 Menu", menu, index=default_index)

    # Update page only if changed via sidebar
    if choice != st.session_state.page:
        st.session_state.page = choice

    # Routing logic
    if st.session_state.page == "Login":
        login()
    elif st.session_state.page == "Signup":
        signup()
    elif st.session_state.page == "Verify Email":
        verify()
    elif st.session_state.page == "Home":
        show_homepage()
    elif st.session_state.page == "Dashboard":
        show_dashboard()
    elif st.session_state.page == "News":
        show_google_news()
    elif st.session_state.page == "Candlechart":
        st.title("📊 Live Candle Chart")

        symbol = st.selectbox("Select a stock symbol", [
            "NSE:TCS", "NSE:INFY", "NSE:RELIANCE", "NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL"
        ], index=0)

        tradingview(symbol)
    
    elif st.session_state.page == "Settings":    
        settings_module()
    elif st.session_state.page == "Chatbot":
        show_chatbot_page()     
    elif st.session_state.page == "Admin Panel":
        show_admin_panel()
    elif st.session_state.page == "Support":    
        support_module()
    elif st.session_state.page == "Logout":
        show_logout_confirmation()

#-----------------------------------------------------------------------------------------------------------------------#            

if __name__ == '__main__':
    main()

#-----------------------------------------------------------------------------------------------------------------------#  