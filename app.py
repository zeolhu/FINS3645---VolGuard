import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VolGuard Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path, index_col=None):
    """Loads a CSV file from a given path, returns an empty DataFrame if file not found."""
    try:
        return pd.read_csv(file_path, index_col=index_col)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please check your folder structure.")
        return pd.DataFrame()


def get_rebalancing_recommendation(sentiment_score):
    """
    Provides a portfolio allocation recommendation based on a sentiment score.
    This function contains the core optimization logic.
    """
    if sentiment_score < 0.40:
        profile = "Defensive Stance"
        recommendation = {
            "Bitcoin (BTC)": 25,
            "Ethereum (ETH)": 25,
            "Altcoins": 10,
            "Stablecoins (USDC/USDT)": 40
        }
        explanation = "Sentiment is low, indicating market fear or uncertainty. The portfolio shifts to a defensive position with a higher allocation to stablecoins to minimize potential drawdowns."
    elif 0.40 <= sentiment_score < 0.65:
        profile = "Balanced Growth"
        recommendation = {
            "Bitcoin (BTC)": 40,
            "Ethereum (ETH)": 40,
            "Altcoins": 15,
            "Stablecoins (USDC/USDT)": 5
        }
        explanation = "Sentiment is neutral to positive. The portfolio is balanced to capture upside potential while maintaining a small defensive cash position."
    else:  # sentiment_score >= 0.65
        profile = "Aggressive Growth"
        recommendation = {
            "Bitcoin (BTC)": 45,
            "Ethereum (ETH)": 45,
            "Altcoins": 10,
            "Stablecoins (USDC/USDT)": 0
        }
        explanation = "Sentiment is strongly positive, signaling market confidence. The portfolio maximizes exposure to high-growth assets (BTC and ETH) to capitalize on the upward momentum."

    return profile, recommendation, explanation


def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')


# --- FILE PATHS & DATA LOADING ---
BASE_DIR = Path(__file__).resolve().parent
CRYPTO_RESULTS_DIR = BASE_DIR / "crypto_results"
SENTIMENT_RESULTS_DIR = BASE_DIR / "sentiment_results"

perf_metrics_df = load_data(CRYPTO_RESULTS_DIR / "portfolio_performance_metrics.csv", index_col=0)
baseline_df = load_data(CRYPTO_RESULTS_DIR / "baseline_backtest.csv")
sentiment_df = load_data(SENTIMENT_RESULTS_DIR / "sentiment_backtest.csv")
sentiment_index_df = load_data(SENTIMENT_RESULTS_DIR / "Sentiment_index.csv")

# --- HEADER & PROJECT DESCRIPTION ---
st.title("🛡️ VolGuard: Crypto Portfolio Optimisation Using News Sentiment")
st.markdown("""
This dashboard showcases the VolGuard backtesting engine and provides a **live rebalancing advisor**. By integrating a sentiment index from financial news, VolGuard aims to reduce volatility and improve risk-adjusted returns in digital asset markets.
""")
st.markdown("---")

# --- NEW: VOLGUARD REBALANCING ADVISOR ---
st.header("🤖 VolGuard Rebalancing Advisor")
st.info(
    "This advisor simulates a real-time recommendation using the most recent sentiment data from our historical analysis.")

if not sentiment_index_df.empty:
    try:
        # Get the latest sentiment score from the data
        latest_sentiment_row = sentiment_index_df.iloc[-1]
        latest_date = pd.to_datetime(latest_sentiment_row['date']).strftime('%Y-%m-%d')
        latest_score = latest_sentiment_row['sentiment_index']

        # Get the recommendation from our logic function
        profile, recommendation, explanation = get_rebalancing_recommendation(latest_score)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Current Market Signal")
            st.metric(label=f"Latest Sentiment Score (as of {latest_date})", value=f"{latest_score:.3f}")
            st.markdown(f"**Interpreted Profile: `{profile}`**")
            with st.expander("See Explanation"):
                st.write(explanation)

        with col2:
            st.subheader("Recommended Portfolio Allocation")

            # Create the pie chart
            pie_fig = go.Figure(data=[go.Pie(
                labels=list(recommendation.keys()),
                values=list(recommendation.values()),
                hole=.3,
                pull=[0.05 if asset == "Bitcoin (BTC)" else 0 for asset in recommendation.keys()]  # Highlight BTC
            )])
            pie_fig.update_traces(textinfo='percent+label', textfont_size=14)
            pie_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(pie_fig, use_container_width=True)

    except (KeyError, IndexError) as e:
        st.error(
            f"Could not generate recommendation. Please check that `Sentiment_index.csv` contains 'date' and 'sentiment_index' columns. Error: {e}")
else:
    st.warning("Sentiment Index data not loaded. Advisor cannot generate a recommendation.")

st.markdown("---")

# --- Tabs for Retrospective Analysis ---
st.header(" retrospective Analysis")
tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "💹 Cumulative Returns", "🖼️ Visualisation Gallery"])

with tab1:
    # --- 1. PERFORMANCE METRICS DISPLAY ---
    st.subheader("Key Performance Metrics")
    if not perf_metrics_df.empty:
        st.dataframe(perf_metrics_df, use_container_width=True)
    else:
        st.warning("Could not load performance metrics.")

with tab2:
    # --- 2. CUMULATIVE RETURN PLOT ---
    st.subheader("Cumulative Returns Comparison")
    if not baseline_df.empty and not sentiment_df.empty:
        try:
            baseline_df.rename(columns={'date': 'Date', 'portfolio_value': 'Equal Weight'}, inplace=True)
            sentiment_df.rename(columns={'date': 'Date', 'portfolio_value': 'Sentiment Weighted'}, inplace=True)

            comparison_df = pd.merge(baseline_df[['Date', 'Equal Weight']],
                                     sentiment_df[['Date', 'Sentiment Weighted']], on='Date', how='inner')
            comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Equal Weight'], mode='lines',
                                     name='Equal-Weighted (Baseline)'))
            fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Sentiment Weighted'], mode='lines',
                                     name='Sentiment-Weighted'))
            fig.update_layout(title='Portfolio Growth Over Time', xaxis_title='Date', yaxis_title='Cumulative Return',
                              legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)
        except KeyError as e:
            st.error(f"A column was not found in backtest files. Error: {e}")
    else:
        st.warning("Could not load one or both backtest files to generate the cumulative return plot.")

with tab3:
    # --- 3. VISUALISATION GALLERY ---
    st.subheader("Backtest Performance Comparison")
    try:
        st.image(str(CRYPTO_RESULTS_DIR / "backtest_comparison.png"),
                 caption="Side-by-side performance of both portfolios.", use_column_width=True)
    except Exception:
        st.warning("Image 'backtest_comparison.png' not found.")

    st.subheader("Sentiment Analysis Visuals")
    # ... (rest of the image gallery code remains the same)

# --- KEY INSIGHTS & DOWNLOADS ---
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    st.header("🔑 Key Insights from Backtest")
    st.markdown("""
    - The sentiment-weighted portfolio exhibited **lower volatility and drawdown** than the baseline.
    - Sentiment signals were especially useful during periods of **high market uncertainty**.
    - Positive sentiment spikes led to increased allocation to **BTC/ETH**.
    - Fear periods saw defensive allocation to **stablecoins** and reduced exposure to volatile altcoins.
    """)

with col2:
    st.header("📥 Download Data")
    if not baseline_df.empty:
        st.download_button("Download Baseline Backtest (CSV)", convert_df_to_csv(baseline_df), "baseline_backtest.csv",
                           "text/csv")
    if not sentiment_index_df.empty:
        st.download_button("Download Sentiment Index (CSV)", convert_df_to_csv(sentiment_index_df),
                           "Sentiment_index.csv", "text/csv")