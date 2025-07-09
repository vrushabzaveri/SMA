# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import ta
import warnings
warnings.filterwarnings('ignore')

ALPHA_VANTAGE_KEY = "HNU1UUHL9351CEWZ"

# ------------------ UTILS ------------------
def flatten_series(series_like):
    try:
        if isinstance(series_like, pd.DataFrame) and series_like.shape[1] == 1:
            return series_like.iloc[:, 0]
        elif isinstance(series_like, pd.DataFrame):
            return series_like.squeeze()
        elif isinstance(series_like, np.ndarray):
            return pd.Series(series_like.flatten())
        elif isinstance(series_like, pd.Series):
            return series_like
        else:
            return pd.Series(series_like)
    except Exception:
        return pd.Series(dtype="float64")

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

# ------------------ STREAMLIT CONFIG ------------------
st.set_page_config(page_title="ISA Forecast", layout="wide")
st.title("📈 ISA Stock Forecasting")

@st.cache_data
def fetch_static_stocks():
    try:
        df = pd.read_csv("nse_stocks.csv")
        return df[["SYMBOL", "NAME OF COMPANY"]].dropna()
    except:
        return pd.DataFrame({
            "SYMBOL": ["RELIANCE", "TCS"],
            "NAME OF COMPANY": ["Reliance Industries", "Tata Consultancy Services"]
        })

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
        status.info("⏳ Downloading stock data...")
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            st.error("⚠️ No data found.")
            st.stop()

        df = df[df.index.dayofweek < 5]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df:
                df[col] = 0

        volume_series = df['Volume']
        volume_missing = volume_series.isnull().all()
        volume_zero = (volume_series == 0).all()

        if bool(volume_missing) or bool(volume_zero):
            fallback_volume = fetch_alpha_vantage_volume(symbol_raw)
            if not fallback_volume.empty:
                df = df.join(fallback_volume.rename("Volume_AV"), how="left")
                df["Volume"] = df["Volume"].fillna(df["Volume_AV"])
                df.drop(columns=["Volume_AV"], inplace=True, errors="ignore")
            else:
                df["Volume"] = 1000

        df["Volume"] = np.log1p(df["Volume"].fillna(0))
        progress_bar.progress(20)

        status.info("📊 Calculating technical indicators...")
        close_series = df["Close"].copy()
        if len(close_series.dropna()) < 50:
            st.error("🚨 Not enough data for indicators.")
            st.stop()

        df["RSI"] = ta.momentum.RSIIndicator(close=close_series).rsi()
        macd = ta.trend.MACD(close=close_series)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["Change %"] = close_series.pct_change() * 100
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close_series) >= period:
                df[f"SMA_{period}"] = ta.trend.sma_indicator(close_series, window=period)
                df[f"EMA_{period}"] = ta.trend.ema_indicator(close_series, window=period)

        bb = ta.volatility.BollingerBands(close=close_series)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()

        df.dropna(inplace=True)
        if len(df) < 30:
            st.error("🚨 Not enough clean data after indicators.")
            st.stop()
        progress_bar.progress(50)

        status.info("💡 Preparing data...")
        feature_cols = [col for col in df.columns if col != "Adj Close"]
        close_index = feature_cols.index("Close")
        data = df[feature_cols].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        progress_bar.progress(70)

        X = scaled_data[:-1]
        y = scaled_data[1:, close_index]
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        progress_bar.progress(80)

        status.info("🧐 Training models...")
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVM (RBF)": SVR(kernel="rbf")
        }
        best_score = -np.inf
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
                    best_model = model
                    best_name = name
            except Exception as e:
                st.warning(f"{name} failed: {e}")
                scores[name] = -1

        if best_model is None:
            st.error("❌ No model could be trained.")
            st.stop()
        progress_bar.progress(90)

        status.info("📈 Predicting...")
        last_day = scaled_data[-1].reshape(1, -1)
        pred_scaled = best_model.predict(last_day)[0]
        last_day[0, close_index] = pred_scaled
        pred_price = scaler.inverse_transform(last_day)[0, close_index]
        curr_price = float(df["Close"].iloc[-1])
        rec, pct, text = make_recommendation(curr_price, pred_price)
        progress_bar.progress(100)

        st.metric("Current", f"₹{curr_price:.2f}")
        st.metric("Predicted", f"₹{pred_price:.2f}")
        st.metric("Change %", f"{pct:.2f}%")
        st.markdown(f"**{rec}** — {text}")

        st.subheader("📅 News Sentiment")
        for article in fetch_news(selected_company[0]):
            st.markdown(f"**{get_sentiment(article.title)}** [{article.title}]({article.link})")
            st.caption(article.get('published', 'Unknown'))

    except Exception as e:
        st.error(f"Unexpected error: {e}")
