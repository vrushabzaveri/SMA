import streamlit as st
import bcrypt
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

ALPHA_VANTAGE_KEY = "HNU1UUHL9351CEWZ"
st.set_page_config(page_title="ISA Forecast", layout="wide")

# --- Login Logic ---
def check_password(username, password):
    if username in st.secrets["users"]:
        stored_hash = st.secrets["users"][username].encode()
        return bcrypt.checkpw(password.encode(), stored_hash)
    return False

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

if not st.session_state.authenticated:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_password(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

# --- Authenticated Area ---
st.title("📈 Stock Forecasting App")
st.sidebar.success(f"👋 Logged in as **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()

# 
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

def make_recommendation(current_price, predicted_price, r2_score=None):
    change = ((predicted_price - current_price) / current_price) * 100
    confidence = ""
    if r2_score is not None:
        if r2_score > 0.85: confidence = "🔒 Very High Confidence"
        elif r2_score > 0.70: confidence = "🔐 High Confidence"
        elif r2_score > 0.50: confidence = "🔓 Moderate Confidence"
        else: confidence = "⚠️ Low Confidence"

    if change > 5:
        rec = "🔥 Strong Buy"
        msg = "High upside potential. Consider entering now and monitoring closely."
    elif change > 2:
        rec = "🟢 Buy"
        msg = "Positive momentum expected. A good entry point for short to mid-term."
    elif change < -5:
        rec = "🚨 Strong Sell"
        msg = "Significant downside likely. Consider exiting or hedging your position."
    elif change < -2:
        rec = "🔴 Sell"
        msg = "Potential decline ahead. Monitor carefully or consider selling partial holdings."
    else:
        rec = "⚪ Hold"
        msg = "Sideways movement expected. Best to wait for stronger signals."

    return rec, change, f"{msg} ({confidence})"

def get_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.2: return "🟢 Positive"
        elif polarity < -0.2: return "🔴 Negative"
        else: return "🟡 Neutral"
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

# --- User Input ---
stock_df = fetch_static_stocks()
if stock_df.empty:
    st.error("❌ Could not load NSE stock list.")
    st.stop()

company_options = stock_df[["NAME OF COMPANY", "SYMBOL"]].values.tolist()
selected_company = st.selectbox("Choose a Company", company_options, format_func=lambda x: x[0])
symbol = selected_company[1] + ".NS"
symbol_raw = selected_company[1]

start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
st.markdown(f"**Selected Range:** `{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}`")

if st.button("🔍 Analyze"):
    progress_bar = st.progress(0)
    status = st.empty()

    try:
        status.info("⏳ Fetching stock data...")
        df = yf.download(symbol, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, map(str, col))) for col in df.columns]

        if df.empty:
            st.error("⚠️ No data found.")
            st.stop()

        close_col_candidates = [col for col in df.columns if "close" in col.lower()]
        df['Close'] = df[close_col_candidates[0]]
        close_series = df['Close']

        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df["Volume"] = df["Volume"].replace(0, np.nan)

        fallback_volume = fetch_alpha_vantage_volume(symbol_raw)
        if not fallback_volume.empty:
            df = df.join(fallback_volume.rename("Volume_AV"), how="left")
            df["Volume"] = df["Volume"].fillna(df["Volume_AV"])
            df.drop(columns=["Volume_AV"], inplace=True)

        if df["Volume"].isnull().all():
            df["Volume"] = 1000.0

        df["Volume"] = np.log1p(pd.to_numeric(df["Volume"], errors="coerce"))

        progress_bar.progress(20)
        status.info("📊 Calculating indicators...")

        df["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()
        macd = ta.trend.MACD(close_series)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["Change %"] = close_series.pct_change() * 100
        bb = ta.volatility.BollingerBands(close_series)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df.dropna(inplace=True)

        if len(df) < 30:
            st.error("📉 Not enough data after cleaning.")
            st.stop()

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        close_index = df.columns.get_loc("Close")
        X = scaled_data[:-1]
        y = scaled_data[1:, close_index]

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

        model_results = {}
        best_model, best_score = None, -np.inf

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            model_results[name] = {"model": model, "r2": r2, "preds": preds, "errors": y_test - preds}
            if r2 > best_score:
                best_score = r2
                best_model = model

        st.subheader("🧠 Combined Model Prediction vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name="Actual", line=dict(color="white", width=2)))
        for i, (name, result) in enumerate(model_results.items()):
            fig.add_trace(go.Scatter(y=result["preds"], name=name, line=dict(width=2)))
        fig.update_layout(height=500, title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Model Evaluation Results")
        latest_price = df["Close"].iloc[-2]
        actual_price = df["Close"].iloc[-1]
        best_final_rec, best_final_msg, best_final_model = "", "", ""

        for name, result in model_results.items():
            r2 = result["r2"]
            pred_scaled = result["preds"][-1]
            try:
                test_sample = X_test[-1].copy()
                test_sample[close_index] = pred_scaled
                predicted_price = scaler.inverse_transform([test_sample])[0][close_index]
            except:
                predicted_price = actual_price

            rec, change, msg = make_recommendation(latest_price, predicted_price, r2)
            if r2 == best_score:
                best_final_rec = rec
                best_final_msg = msg
                best_final_model = name

            st.markdown(f"""
            #### 🧠 {name}
            - **R² Score:** `{r2:.4f}`
            - **Predicted Price:** ₹`{predicted_price:.2f}`
            - **Recommendation:** {rec} — *{msg}*
            """)

        st.subheader("✅ Final Decision")
        st.markdown(f"""
        Based on the best performing model (**{best_final_model}**), the final recommendation is:

        ### {best_final_rec}
        > _{best_final_msg}_
        """)

        st.subheader("📰 Latest News")
        news = fetch_news(selected_company[0])
        if not news:
            st.info("No recent news found.")
        else:
            for article in news:
                sentiment = get_sentiment(article.title)
                st.markdown(f"""
                - [{article.title}]({article.link})  
                  <small><i>{sentiment}</i> — {article.published}</small>
                """, unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error(f"❌ Error: {e}")
        st.code(traceback.format_exc(), language="python")
