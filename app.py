import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime
from textblob import TextBlob
import feedparser
import ta
import holidays


# --------- Helpers ---------
def flatten_series(series_like):
    if isinstance(series_like, pd.DataFrame):
        return series_like.iloc[:, 0]
    elif isinstance(series_like, np.ndarray):
        return pd.Series(series_like.ravel())
    return pd.Series(series_like).squeeze()


def get_close_column(df, symbol):
    possible_cols = [col for col in df.columns if f"Close_{symbol}" in str(col)]
    if not possible_cols:
        st.error(f"❌ Could not find a Close column for {symbol}. Found columns: {', '.join(map(str, df.columns))}")
        st.stop()
    return possible_cols[0]


def make_recommendation(current_price, predicted_price):
    change = ((predicted_price - current_price) / current_price) * 100
    if change > 5:
        return "📈 Strong Buy", change, "High confidence in growth. You can consider buying."
    elif change > 2:
        return "📈 Buy", change, "The model suggests the stock may rise. You can consider buying."
    elif change < -5:
        return "🔥 Strong Sell", change, "Sharp decline expected. Avoid or exit if holding."
    elif change < -2:
        return "📉 Sell", change, "Price might fall. Better avoid or sell if holding."
    else:
        return "🤝 Hold", change, "Not much change expected. Buying isn’t risky, but may not be rewarding either."


def r2_interpretation(score):
    if score <= 0:
        return "❌ Poor (Worse than random)"
    elif score <= 0.3:
        return "⚠️ Weak Prediction"
    elif score <= 0.5:
        return "😐 Moderate"
    elif score <= 0.7:
        return "👍 Decent"
    elif score <= 0.9:
        return "✅ Good"
    elif score < 1.0:
        return "🔏 Excellent"
    else:
        return "🚀 Perfect (Possible overfit)"


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "🟢 Positive"
    elif polarity < -0.2:
        return "🔴 Negative"
    else:
        return "🟡 Neutral"


def fetch_news(company_name):
    url = f"https://news.google.com/rss/search?q={company_name.replace(' ', '%20')}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    return feedparser.parse(url).entries[:5]


# --------- Config ---------
st.set_page_config(page_title="ISA", layout="wide")
st.title("ISA Forecast")
SCALER_PATH = "scaler.gz"
CSV_PATH = "nse_stocks.csv"


@st.cache_data
def fetch_static_stocks():
    try:
        df = pd.read_csv(CSV_PATH)
        return df[["SYMBOL", "NAME OF COMPANY"]].dropna()
    except Exception as e:
        st.error(f"❌ Failed to load nse_stocks.csv: {e}")
        return pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY"])


# --------- UI ---------
stock_df = fetch_static_stocks()
if stock_df.empty:
    st.stop()

company_options = stock_df[["NAME OF COMPANY", "SYMBOL"]].values.tolist()
selected_company = st.selectbox("Choose a Company", company_options, format_func=lambda x: x[0])
symbol = selected_company[1] + ".NS"
symbol_raw = selected_company[1]

# Always reset the scaler cache on new stock selection
if os.path.exists(SCALER_PATH):
    os.remove(SCALER_PATH)

start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
st.markdown(f"**Selected Range:** `{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}`")

if st.button("🗑️ Reset Scaler Cache"):
    if os.path.exists(SCALER_PATH):
        os.remove(SCALER_PATH)
    st.success("✅ Scaler cache cleared!")

# --------- Analyze Button ---------
if st.button("🔍 Analyze"):
    start_time = time.time()
    progress_bar = st.progress(0)
    status = st.empty()

    # Step 1: Download
    status.info("⏳ Step 1: Downloading stock data...")
    df = yf.download(symbol, start=start_date, end=end_date)
    df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df = df[df.index.dayofweek < 5]
    if df.empty:
        st.error("⚠️ No data available.")
        st.stop()
    progress_bar.progress(15)

    # Step 2: Indicators
    status.info("📊 Step 2: Calculating indicators...")
    close_col = get_close_column(df, symbol)
    close_series = flatten_series(df[close_col])
    df["Close"] = close_series  # for simplicity downstream

    df["RSI"] = ta.momentum.RSIIndicator(close=close_series).rsi()
    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd().squeeze()
    df["MACD_Signal"] = macd.macd_signal().squeeze()
    df["Change %"] = close_series.pct_change() * 100
    df["Volume"] = np.log1p(flatten_series(df.get(f"Volume_{symbol_raw}.NS", df["Volume"])))

    for period in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{period}"] = ta.trend.sma_indicator(close_series, window=period).squeeze()
        df[f"EMA_{period}"] = ta.trend.ema_indicator(close_series, window=period).squeeze()

    df.dropna(inplace=True)
    progress_bar.progress(30)

    # Step 3: Scaling
    status.info("💡 Step 3: Scaling & preparing data...")
    feature_cols = [col for col in df.columns if col not in ["Adj Close"]]
    close_index = feature_cols.index("Close")
    data = df[feature_cols]

    # Validate data before scaling
    if data is None or data.empty:
        st.error("🚨 Data is empty. Cannot scale.")
        st.stop()
    if data.isnull().values.any():
        st.error("🚨 Data contains NaNs. Cannot scale.")
        st.stop()
    if not np.isfinite(data.to_numpy()).all():
        st.error("🚨 Data contains infinite values. Cannot scale.")
        st.stop()

    try:
        scaler = MinMaxScaler()
        scaler.fit(data)
        joblib.dump(scaler, SCALER_PATH)
        scaled_data = scaler.transform(data)
    except Exception as e:
        st.error(f"❌ Failed to scale data: {e}")
        st.stop()
    progress_bar.progress(45)

    # Step 4: Split
    status.info("📂 Step 4: Train-Test Split")
    X = scaled_data[:-1]
    y = scaled_data[1:, close_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    progress_bar.progress(60)

    # Step 5: Train
    status.info("🧠 Step 5: Training models...")
    models = {
        "SVM (Linear)": SVR(kernel="linear"),
        "SVM (RBF)": SVR(kernel="rbf"),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "ANN (MLP)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500),
    }

    predictions, scores = {}, {}
    best_score, best_model_name, best_model = -np.inf, None, None
    for i, (name, model) in enumerate(models.items(), start=1):
        status.info(f"🚀 Training `{name}` ({i}/{len(models)})")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        predictions[name] = pred
        scores[name] = r2
        if r2 > best_score:
            best_score, best_model_name, best_model = r2, name, model

    progress_bar.progress(90)

    # Step 6: Final Prediction
    status.info("📈 Final predictions...")
    last_day = scaled_data[-1].reshape(1, -1)
    predicted_scaled = best_model.predict(last_day)[0]
    predicted_full = last_day.copy()
    predicted_full[0, close_index] = predicted_scaled
    predicted_price = scaler.inverse_transform(predicted_full)[0, close_index]
    current_price = float(df["Close"].iloc[-1])
    recommendation, change_pct, suggestion_text = make_recommendation(current_price, predicted_price)
    progress_bar.progress(100)
    status.success("✅ Analysis complete!")

    # Charts
    st.subheader("📊 Actual vs Predicted Close Price (Test Set)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode="lines", name="Actual"))
    for name, pred in predictions.items():
        fig.add_trace(go.Scatter(y=pred, mode="lines", name=f"{name} ({scores[name]:.2f})"))
    fig.update_layout(title="Actual vs Predicted", xaxis_title="Time", yaxis_title="Scaled Close", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Model Accuracy Summary")
    for name, score in scores.items():
        st.markdown(f"**{name}** → R² Score: `{score:.2f}` → {r2_interpretation(score)}")

    with st.expander("📘 R² Score Legend"):
        st.markdown("""
        **❌ Poor**: `0 or below`  
        **⚠️ Weak**: `0.0 – 0.3`  
        **😐 Moderate**: `0.3 – 0.5`  
        **👍 Decent**: `0.5 – 0.7`  
        **✅ Good**: `0.7 – 0.9`  
        **🔏 Excellent**: `0.9 – 0.99`  
        **🚀 Perfect**: `1.0`  
        """)

    # Final recommendation
    st.subheader("🔍 Final Recommendation")
    st.markdown(f"**Best Model**: `{best_model_name}`")
    st.markdown(f"**Current Close Price**: `{current_price:.2f}`")
    st.markdown(f"**Predicted Next Close**: `{predicted_price:.2f}`")
    st.markdown(f"**Expected Change**: `{change_pct:.2f}%`")
    st.markdown(f"**Action**: {recommendation}")
    st.info(f"💬 {suggestion_text}")

    # News
    st.subheader("📰 News Sentiment Analysis")
    try:
        news_articles = fetch_news(selected_company[0])
        if not news_articles:
            st.warning("No recent news found.")
        else:
            for article in news_articles:
                title = article.title
                link = article.link
                published = article.published if "published" in article else "Unknown"
                sentiment = get_sentiment(title)
                st.markdown(f"**{sentiment}** [{title}]({link})")
                st.caption(f"🗓 {published}")
    except Exception as e:
        st.error(f"🤨 News fetch failed: {e}")
