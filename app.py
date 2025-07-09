# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import ta
import warnings
warnings.filterwarnings('ignore')

ALPHA_VANTAGE_KEY = "HNU1UUHL9351CEWZ"

# ------------------ UTILS ------------------
def flatten_series(series_like):
    """Safely flatten series-like objects to 1D pandas Series"""
    if isinstance(series_like, pd.DataFrame):
        if series_like.shape[1] == 1:
            return series_like.iloc[:, 0]
        else:
            return series_like.squeeze()
    elif isinstance(series_like, np.ndarray):
        return pd.Series(series_like.flatten())
    elif isinstance(series_like, pd.Series):
        return series_like
    else:
        return pd.Series(series_like)

def make_recommendation(current_price, predicted_price):
    change = ((predicted_price - current_price) / current_price) * 100
    if change > 5:
        return "🔥 Strong Buy", change, "High confidence in growth. You can consider buying."
    elif change > 2:
        return "🟢 Buy", change, "The model suggests the stock may rise. You can consider buying."
    elif change < -5:
        return "🚨 Strong Sell", change, "Sharp decline expected. Avoid or exit if holding."
    elif change < -2:
        return "🔴 Sell", change, "Price might fall. Better avoid or sell if holding."
    else:
        return "⚪ Hold", change, "Not much change expected. Buying isn't risky, but may not be rewarding either."

def r2_interpretation(score):
    if score <= 0:
        return "❌ Poor"
    elif score <= 0.3:
        return "⚠️ Weak"
    elif score <= 0.5:
        return "😐 Moderate"
    elif score <= 0.7:
        return "👍 Decent"
    elif score <= 0.9:
        return "✅ Good"
    elif score < 1.0:
        return "🔑 Excellent"
    else:
        return "🚀 Perfect (Possible Overfit)"

def get_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.2:
            return "🟢 Positive"
        elif polarity < -0.2:
            return "🔴 Negative"
        else:
            return "🟡 Neutral"
    except:
        return "🟡 Neutral"

def fetch_news(company_name):
    try:
        url = f"https://news.google.com/rss/search?q={company_name.replace(' ', '%20')}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        return feedparser.parse(url).entries[:5]
    except:
        return []

def fetch_alpha_vantage_volume(symbol_raw):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol_raw}.NS&outputsize=full&apikey={ALPHA_VANTAGE_KEY}&datatype=csv"
        df = pd.read_csv(url)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df["volume"]
    except:
        return pd.Series(dtype="float64")

def safe_technical_indicator(close_series, indicator_func, *args, **kwargs):
    """Safely calculate technical indicators with proper error handling"""
    try:
        # Ensure we have a proper 1D pandas Series
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()
        
        # Convert to pandas Series if it's not already
        if not isinstance(close_series, pd.Series):
            close_series = pd.Series(close_series)
        
        # Ensure numeric data
        close_series = pd.to_numeric(close_series, errors='coerce')
        
        # Drop NaN values
        close_series = close_series.dropna()
        
        # Check if we have enough data
        if len(close_series) < 20:
            return pd.Series(index=close_series.index, dtype=float)
        
        # Call the indicator function
        return indicator_func(close_series, *args, **kwargs)
    
    except Exception as e:
        st.warning(f"Error calculating indicator: {e}")
        return pd.Series(index=close_series.index if hasattr(close_series, 'index') else range(len(close_series)), dtype=float)

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="ISA Forecast", layout="wide")
st.title("📈 ISA Stock Forecasting")

@st.cache_data
def fetch_static_stocks():
    try:
        df = pd.read_csv("nse_stocks.csv")
        return df[["SYMBOL", "NAME OF COMPANY"]].dropna()
    except:
        # Fallback with some common NSE stocks
        fallback_data = {
            "SYMBOL": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC"],
            "NAME OF COMPANY": ["Reliance Industries Limited", "Tata Consultancy Services Limited", "HDFC Bank Limited", 
                               "Infosys Limited", "ICICI Bank Limited", "Hindustan Unilever Limited", 
                               "State Bank of India", "Bharti Airtel Limited", "Kotak Mahindra Bank Limited", "ITC Limited"]
        }
        return pd.DataFrame(fallback_data)

stock_df = fetch_static_stocks()
if stock_df.empty:
    st.error("❌ Could not load NSE stock list.")
    st.stop()

company_options = stock_df[["NAME OF COMPANY", "SYMBOL"]].values.tolist()
selected_company = st.selectbox("Choose a Company", company_options, format_func=lambda x: x[0])
symbol = selected_company[1] + ".NS"
symbol_raw = selected_company[1]

start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

st.markdown(f"**Selected Range:** `{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}`")

if st.button("🔍 Analyze"):
    progress_bar = st.progress(0)
    status = st.empty()

    try:
        # Step 1: Download
        status.info("⏳ Step 1: Downloading stock data...")
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            st.error("⚠️ No data found for the selected symbol and date range.")
            st.stop()
        
        # Filter out weekends
        df = df[df.index.dayofweek < 5]
        
        # Ensure we have the basic columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'Volume':
                    df[col] = 0
                else:
                    st.error(f"Missing required column: {col}")
                    st.stop()
        
        progress_bar.progress(20)

        # Step 2: Handle Volume data
        status.info("📊 Step 2: Processing volume data...")
        volume_series = df["Volume"]
        
        # Check if volume data is problematic
        volume_issues = (
            volume_series.isnull().all() or 
            (volume_series == 0).all() or 
            volume_series.empty
        )
        
        if volume_issues:
            status.warning("⚠️ Volume missing in yfinance. Fetching from Alpha Vantage...")
            fallback_volume = fetch_alpha_vantage_volume(symbol_raw)
            if not fallback_volume.empty:
                df = df.join(fallback_volume.rename("Volume_AV"), how="left")
                df["Volume"] = df["Volume"].fillna(df["Volume_AV"])
                df.drop(columns=["Volume_AV"], inplace=True, errors="ignore")
                st.success("✅ Volume filled from Alpha Vantage.")
            else:
                df["Volume"] = 1000  # Default volume
        
        # Apply log transformation to volume
        df["Volume"] = np.log1p(df["Volume"].fillna(0))
        progress_bar.progress(40)

        # Step 3: Calculate Technical Indicators
        status.info("📊 Step 3: Calculating technical indicators...")
        
        # Get close prices as a clean 1D Series
        close_series = df["Close"].copy()
        
        # Ensure we have enough data
        if len(close_series.dropna()) < 50:
            st.error("🚨 Not enough data to calculate indicators. Need at least 50 data points.")
            st.stop()
        
        # Calculate indicators safely
        try:
            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close=close_series, window=14)
            df["RSI"] = rsi_indicator.rsi()
            
            # MACD
            macd_indicator = ta.trend.MACD(close=close_series)
            df["MACD"] = macd_indicator.macd()
            df["MACD_Signal"] = macd_indicator.macd_signal()
            
            # Price change percentage
            df["Change %"] = close_series.pct_change() * 100
            
            # Moving averages
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                if len(close_series) >= period:
                    df[f"SMA_{period}"] = ta.trend.sma_indicator(close_series, window=period)
                    df[f"EMA_{period}"] = ta.trend.ema_indicator(close_series, window=period)
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(close=close_series)
            df["BB_Upper"] = bb_indicator.bollinger_hband()
            df["BB_Lower"] = bb_indicator.bollinger_lband()
            
        except Exception as e:
            st.warning(f"Some technical indicators could not be calculated: {e}")
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < 30:
            st.error("🚨 Not enough clean data after calculating indicators.")
            st.stop()
        
        progress_bar.progress(60)

        # Step 4: Prepare data for ML
        status.info("💡 Step 4: Preparing data for machine learning...")
        
        # Select feature columns (exclude Adj Close to avoid data leakage)
        feature_cols = [col for col in df_clean.columns if col != "Adj Close"]
        
        # Get the index of Close column for prediction
        close_index = feature_cols.index("Close")
        
        # Prepare the data
        data = df_clean[feature_cols].values
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        progress_bar.progress(70)

        # Step 5: Train-test split
        status.info("📂 Step 5: Creating train-test split...")
        
        # Create sequences for time series prediction
        X = scaled_data[:-1]  # All but last row
        y = scaled_data[1:, close_index]  # Close price of next day
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        progress_bar.progress(80)

        # Step 6: Train models
        status.info("🧠 Step 6: Training machine learning models...")
        
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVM (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        }
        
        best_score = -np.inf
        best_model_name = None
        best_model = None
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2 = r2_score(y_test, pred)
                
                predictions[name] = pred
                scores[name] = r2
                
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    best_model = model
            except Exception as e:
                st.warning(f"Error training {name}: {e}")
                scores[name] = -1
        
        if best_model is None:
            st.error("❌ No models could be trained successfully.")
            st.stop()
        
        progress_bar.progress(90)

        # Step 7: Make prediction
        status.info("📈 Step 7: Making final prediction...")
        
        # Use the last data point to predict next day
        last_day = scaled_data[-1].reshape(1, -1)
        predicted_scaled = best_model.predict(last_day)[0]
        
        # Create full prediction array for inverse transform
        predicted_full = last_day.copy()
        predicted_full[0, close_index] = predicted_scaled
        
        # Inverse transform to get actual price
        predicted_price = scaler.inverse_transform(predicted_full)[0, close_index]
        current_price = float(df_clean["Close"].iloc[-1])
        
        # Generate recommendation
        recommendation, change_pct, suggestion_text = make_recommendation(current_price, predicted_price)
        
        progress_bar.progress(100)
        status.success("✅ Analysis complete!")

        # Display results
        st.subheader("📊 Model Performance")
        
        # Plot actual vs predicted
        if len(predictions) > 0:
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                y=y_test,
                mode="lines",
                name="Actual",
                line=dict(color="blue", width=2)
            ))
            
            # Add predictions
            colors = ["red", "green", "orange", "purple", "brown"]
            for i, (name, pred) in enumerate(predictions.items()):
                if name in scores and scores[name] > -1:
                    fig.add_trace(go.Scatter(
                        y=pred,
                        mode="lines",
                        name=f"{name} (R²: {scores[name]:.3f})",
                        line=dict(color=colors[i % len(colors)], width=1)
                    ))
            
            fig.update_layout(
                title="Actual vs Predicted Prices (Test Set)",
                xaxis_title="Time",
                yaxis_title="Scaled Price",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Model scores
        st.subheader("📋 Model Scores")
        for name, score in scores.items():
            if score > -1:
                st.markdown(f"**{name}**: R² = `{score:.3f}` → {r2_interpretation(score)}")

        # Final recommendation
        st.subheader("🔍 Final Recommendation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Price", f"₹{current_price:.2f}")
            st.metric("Predicted Price", f"₹{predicted_price:.2f}")
        
        with col2:
            st.metric("Expected Change", f"{change_pct:.2f}%")
            st.markdown(f"**Best Model**: {best_model_name}")
        
        st.markdown(f"**Recommendation**: {recommendation}")
        st.info(f"💬 {suggestion_text}")

        # News sentiment
        st.subheader("📰 Recent News Sentiment")
        try:
            news_articles = fetch_news(selected_company[0])
            if news_articles:
                for article in news_articles[:3]:
                    sentiment = get_sentiment(article.title)
                    st.markdown(f"**{sentiment}** [{article.title}]({article.link})")
                    st.caption(f"📅 {getattr(article, 'published', 'Unknown date')}")
            else:
                st.warning("No recent news found.")
        except Exception as e:
            st.warning(f"Could not fetch news: {e}")

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {e}")
        st.error("Please try with a different stock or date range.")