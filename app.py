# isa_forecast_app.py
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

ALPHA_VANTAGE_KEY = "HNU1UUHL9351CEWZ"

# ------------------ UTILS ------------------
def flatten_series(series_like):
    if isinstance(series_like, pd.DataFrame):
        return series_like.iloc[:, 0]
    elif isinstance(series_like, np.ndarray):
        return pd.Series(series_like.ravel())
    return pd.Series(series_like).squeeze()

def make_recommendation(current_price, predicted_price):
    change = ((predicted_price - current_price) / current_price) * 100
    if change > 5:
        return " Strong Buy", change, "High confidence in growth. You can consider buying."
    elif change > 2:
        return " Buy", change, "The model suggests the stock may rise. You can consider buying."
    elif change < -5:
        return "Strong Sell", change, "Sharp decline expected. Avoid or exit if holding."
    elif change < -2:
        return " Sell", change, "Price might fall. Better avoid or sell if holding."
    else:
        return " Hold", change, "Not much change expected. Buying isn’t risky, but may not be rewarding either."

def r2_interpretation(score):
    if score <= 0:
        return "\u274c Poor"
    elif score <= 0.3:
        return "\u26a0\ufe0f Weak"
    elif score <= 0.5:
        return "\ud83d\ude10 Moderate"
    elif score <= 0.7:
        return "\ud83d\udc4d Decent"
    elif score <= 0.9:
        return "\u2705 Good"
    elif score < 1.0:
        return "\ud83d\udd11 Excellent"
    else:
        return "\ud83d\ude80 Perfect (Possible Overfit)"

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "\ud83d\udfe2 Positive"
    elif polarity < -0.2:
        return "\ud83d\udd34 Negative"
    else:
        return "\ud83d\udfe1 Neutral"

def fetch_news(company_name):
    url = f"https://news.google.com/rss/search?q={company_name.replace(' ', '%20')}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    return feedparser.parse(url).entries[:5]

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
st.title("ISA Stock Forecasting")

@st.cache_data
def fetch_static_stocks():
    try:
        df = pd.read_csv("nse_stocks.csv")
        return df[["SYMBOL", "NAME OF COMPANY"]].dropna()
    except:
        return pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY"])

stock_df = fetch_static_stocks()
if stock_df.empty:
    st.error("\u274c Could not load NSE stock list.")
    st.stop()

company_options = stock_df[["NAME OF COMPANY", "SYMBOL"]].values.tolist()
selected_company = st.selectbox("Choose a Company", company_options, format_func=lambda x: x[0])
symbol = selected_company[1] + ".NS"
symbol_raw = selected_company[1]

start_date = st.date_input("Start Date", pd.to_datetime("2000-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

st.markdown(f"**Selected Range:** `{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}`")

if st.button(" Analyze"):
    progress_bar = st.progress(0)
    status = st.empty()

    # Step 1: Download
    status.info("\u23f3 Step 1: Downloading stock data...")
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("\u26a0\ufe0f No data found.")
        st.stop()
    df = df[df.index.dayofweek < 5]
    df.rename(columns={"Close": "Close", "Volume": "Volume"}, inplace=True)

    if "Volume" not in df.columns or df["Volume"].isnull().all() or (df["Volume"] == 0).all():
        status.warning("\u26a0\ufe0f Volume missing in yfinance. Fetching from Alpha Vantage...")
        fallback_volume = fetch_alpha_vantage_volume(symbol_raw)
        if not fallback_volume.empty:
            df = df.join(fallback_volume.rename("Volume"), how="left", rsuffix="_av")
            df["Volume"] = df["Volume"].fillna(df.get("Volume_av", 0))
            df.drop(columns=["Volume_av"], inplace=True, errors="ignore")
            st.success("\u2705 Volume filled from Alpha Vantage.")
        else:
            df["Volume"] = 0
    df["Volume"] = np.log1p(flatten_series(df["Volume"]))
    progress_bar.progress(20)

    # Step 2: Indicators
    status.info("\ud83d\udcca Step 2: Calculating indicators...")
    close_series = df["Close"].copy()
    if close_series.isnull().sum() > 0 or len(close_series.dropna()) < 20:
        st.error("\ud83d\udea8 Not enough valid Close data to calculate indicators.")
        st.stop()
    df["RSI"] = ta.momentum.RSIIndicator(close=close_series).rsi()
    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["Change %"] = close_series.pct_change() * 100
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{period}"] = ta.trend.sma_indicator(close_series, window=period)
        df[f"EMA_{period}"] = ta.trend.ema_indicator(close_series, window=period)
    df.dropna(inplace=True)
    progress_bar.progress(40)

    # Step 3: Scaling
    status.info("\ud83d\udca1 Step 3: Scaling features...")
    feature_cols = [col for col in df.columns if col != "Adj Close"]
    close_index = feature_cols.index("Close")
    data = df[feature_cols]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    progress_bar.progress(60)

    # Step 4: Train-test split
    status.info("\ud83d\udcc2 Step 4: Train-test split...")
    X = scaled_data[:-1]
    y = scaled_data[1:, close_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    progress_bar.progress(70)

    # Step 5: Modeling
    status.info("\ud83e\udde0 Step 5: Training models...")
    models = {
        "SVM (Linear)": SVR(kernel="linear"),
        "SVM (RBF)": SVR(kernel="rbf"),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "ANN (MLP)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500),
    }
    best_score, best_model_name, best_model = -np.inf, None, None
    predictions, scores = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        predictions[name] = pred
        scores[name] = r2
        if r2 > best_score:
            best_score, best_model_name, best_model = r2, name, model
    progress_bar.progress(90)

    # Step 6: Prediction
    status.info("\ud83d\udcc8 Final prediction...")
    last_day = scaled_data[-1].reshape(1, -1)
    predicted_scaled = best_model.predict(last_day)[0]
    predicted_full = last_day.copy()
    predicted_full[0, close_index] = predicted_scaled
    predicted_price = scaler.inverse_transform(predicted_full)[0, close_index]
    current_price = float(df["Close"].iloc[-1])
    recommendation, change_pct, suggestion_text = make_recommendation(current_price, predicted_price)
    progress_bar.progress(100)
    status.success("\u2705 Done!")

    st.subheader("\ud83d\udcca Actual vs Predicted (Test Set)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode="lines", name="Actual"))
    for name, pred in predictions.items():
        fig.add_trace(go.Scatter(y=pred, mode="lines", name=f"{name} ({scores[name]:.2f})"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("\ud83d\udccb Model Scores")
    for name, score in scores.items():
        st.markdown(f"**{name}** → R²: `{score:.2f}` → {r2_interpretation(score)}")

    st.subheader("\ud83d\udd0d Final Recommendation")
    st.markdown(f"**Best Model**: `{best_model_name}`")
    st.markdown(f"**Current Close Price**: `{current_price:.2f}`")
    st.markdown(f"**Predicted Next Close**: `{predicted_price:.2f}`")
    st.markdown(f"**Expected Change**: `{change_pct:.2f}%`")
    st.markdown(f"**Action**: {recommendation}")
    st.info(f"\ud83d\udcac {suggestion_text}")

    st.subheader("\ud83d\udcf0 News Sentiment")
    try:
        news_articles = fetch_news(selected_company[0])
        if not news_articles:
            st.warning("No recent news found.")
        else:
            for article in news_articles:
                sentiment = get_sentiment(article.title)
                st.markdown(f"**{sentiment}** [{article.title}]({article.link})")
                st.caption(f"\ud83d\uddd3 {getattr(article, 'published', 'Unknown')}")
    except Exception as e:
        st.error(f"\u274c Failed to fetch news: {e}")