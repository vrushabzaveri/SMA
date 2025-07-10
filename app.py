# 🧠 Imports
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
warnings.filterwarnings("ignore")

# 🔑 API Key
ALPHA_VANTAGE_KEY = "HNU1UUHL9351CEWZ"

# 💡 Recommendation Engine
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

# 🧠 News Sentiment
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

# 📰 News Fetching
def fetch_news(company_name):
    try:
        url = f"https://news.google.com/rss/search?q={company_name.replace(' ', '%20')}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        return feedparser.parse(url).entries[:5]
    except:
        return []

# 📦 Volume fallback
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

# 🔧 Streamlit setup
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

# 🚦 Company selection
stock_df = fetch_static_stocks()
if stock_df.empty:
    st.error("❌ Could not load NSE stock list.")
    st.stop()

company_options = stock_df[["NAME OF COMPANY", "SYMBOL"]].values.tolist()
selected_company = st.selectbox("Choose a Company", company_options, format_func=lambda x: x[0])
symbol = selected_company[1] + ".NS"
symbol_raw = selected_company[1]

start_date = st.date_input("Start Date", pd.to_datetime("2000-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
st.markdown(f"**Selected Range:** `{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}`")

# 🔍 Analysis
if st.button("🔍 Analyze"):
    progress_bar = st.progress(0)
    status = st.empty()

    try:
        status.info("⏳ Downloading stock data...")
        df = yf.download(symbol, start=start_date, end=end_date)

        # 🛠 Fix MultiIndex if it exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, map(str, col))) for col in df.columns]

        if df.empty:
            st.error("⚠️ No data found.")
            st.stop()

        # Identify close column reliably
        close_col_candidates = [col for col in df.columns if "close" in col.lower()]
        if not close_col_candidates:
            st.error("❌ Could not identify Close column.")
            st.stop()

        actual_close_col = close_col_candidates[0]
        close_series = pd.Series(df[actual_close_col].values.flatten(), index=df.index, name="Close")

        # Handle Volume robustly
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df["Volume"] = df["Volume"].replace(0, np.nan)

        fallback_volume = fetch_alpha_vantage_volume(symbol_raw)
        if not fallback_volume.empty:
            df = df.join(fallback_volume.rename("Volume_AV"), how="left")
            df["Volume"] = df["Volume"].fillna(df["Volume_AV"])
            df.drop(columns=["Volume_AV"], inplace=True)

        vol = df["Volume"]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        elif isinstance(vol, np.ndarray) and vol.ndim > 1:
            vol = vol.ravel()
            vol = pd.Series(vol, index=df.index)

        vol = pd.to_numeric(vol, errors="coerce")
        if vol.isnull().values.all():
            vol = pd.Series([1000.0] * len(df), index=df.index)
        df["Volume"] = np.log1p(vol.fillna(0))

        progress_bar.progress(20)

        status.info("📊 Calculating indicators...")
        if close_series.dropna().shape[0] < 50:
            st.error("🚨 Not enough data for indicators.")
            st.stop()

        df["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()
        macd = ta.trend.MACD(close_series)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["Change %"] = close_series.pct_change() * 100

        for p in [5, 10, 20, 50, 100, 200]:
            if len(close_series) >= p:
                df[f"SMA_{p}"] = ta.trend.sma_indicator(close_series, window=p)
                df[f"EMA_{p}"] = ta.trend.ema_indicator(close_series, window=p)

        bb = ta.volatility.BollingerBands(close_series)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()

        df["Close"] = pd.to_numeric(close_series, errors="coerce")
        df.dropna(subset=df.columns.difference(["Close"]), inplace=True)

        if "Close" not in df.columns or df["Close"].dropna().shape[0] < 30:
            st.error("❌ Not enough valid 'Close' data after cleaning.")
            st.stop()

        progress_bar.progress(50)
        status.info("💡 Preparing data...")

        df.dropna(inplace=True)

        feature_cols = [col for col in df.columns if col not in ["Adj Close"]]
        if "Close" not in feature_cols:
            feature_cols.append("Close")
        close_index = feature_cols.index("Close")

        data = df[feature_cols].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X = scaled_data[:-1]
        y = scaled_data[1:, close_index].flatten()

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        progress_bar.progress(70)
        status.info("🧠 Training models...")

        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Linear Regression": LinearRegression(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVM (RBF)": SVR(kernel="rbf")
        }

        best_model, best_score = None, -np.inf
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                if r2 > best_score:
                    best_score = r2
                    best_model = model
            except Exception as e:
                st.warning(f"{name} failed: {e}")

        if best_model is None:
            st.error("❌ No model could be trained.")
            st.stop()

        progress_bar.progress(90)
        status.info("📈 Predicting...")

        last_day = scaled_data[-1].reshape(1, -1)
        pred_scaled = best_model.predict(last_day).flatten()[0]
        last_day[0, close_index] = pred_scaled
        pred_price = scaler.inverse_transform(last_day)[0, close_index]
        curr_price = df["Close"].iloc[-1]

        rec, pct, msg = make_recommendation(curr_price, pred_price)
        progress_bar.progress(100)

        st.metric("Current", f"₹{curr_price:.2f}")
        st.metric("Predicted", f"₹{pred_price:.2f}")
        st.metric("Change %", f"{pct:.2f}%")
        st.markdown(f"**{rec}** — {msg}")

        st.subheader("🗾 News Sentiment")
        for article in fetch_news(selected_company[0]):
            st.markdown(f"**{get_sentiment(article.title)}** [{article.title}]({article.link})")
            st.caption(article.get("published", "Unknown"))

    except Exception as e:
        import traceback
        st.error(f"❌ Error: {e}")
        st.code(traceback.format_exc(), language="python")
