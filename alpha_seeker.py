import streamlit as st
import yfinance as yf
import ollama
import pandas as pd
import json
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(page_title="Alpha Seeker", layout="wide", page_icon="🐂")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔭 Watchlist")
    default_tickers = "BTC-USD, ETH-USD, NVDA, TSLA, COIN"
    user_tickers = st.text_area("Enter Tickers (comma separated):", value=default_tickers, height=100)
    st.caption("Supports Crypto (BTC-USD) and Stocks (AAPL).")

# --- FUNCTIONS ---

def fetch_market_data(ticker):
    """
    Get Price + Latest News from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker.strip())
        
        # 1. Get Price Data
        history = stock.history(period="2d")
        if history.empty:
            return None
            
        current_price = history['Close'].iloc[-1]
        prev_close = history['Close'].iloc[0]
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        # 2. Get News (Limit to top 3 to save AI time)
        news = stock.news[:3] if stock.news else []
        
        return {
            "ticker": ticker.upper(),
            "price": current_price,
            "change": change_pct,
            "news": news
        }
    except Exception as e:
        return None

def analyze_sentiment(ticker, news_items):
    """
    Feeds headlines to Llama 3.2 to judge Bullish/Bearish sentiment.
    """
    if not news_items:
        return {"sentiment": "Neutral", "score": 0, "reason": "No recent news found."}
    
    # Prepare a simple text block for the AI
    headlines = [n.get('title', '') for n in news_items]
    headlines_text = "\n".join([f"- {h}" for h in headlines])
    
    prompt = f"""
    Analyze the market sentiment for {ticker} based on these news headlines:
    {headlines_text}
    
    Return a JSON object with:
    - "sentiment": "Bullish", "Bearish", or "Neutral".
    - "score": A number from -10 (Total Panic) to +10 (Extreme Greed).
    - "reason": A 5-word summary of the key driver.
    
    Respond ONLY with JSON.
    """
    
    try:
        response = ollama.chat(
            model='llama3.2',
            format='json',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return json.loads(response['message']['content'])
    except:
        return {"sentiment": "Error", "score": 0, "reason": "AI Failed"}

# --- MAIN APP ---

st.title("🐂 Alpha Seeker: Market Sentiment Scanner")
st.markdown("Correlating **Price Action** with **AI-Driven News Sentiment**.")

if st.button("🚀 Scan Market"):
    
    tickers = [t.strip() for t in user_tickers.split(',') if t.strip()]
    
    # 1. Fetch Data (Parallel)
    status = st.empty()
    status.info(f"Fetching data for {len(tickers)} assets...")
    
    market_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_market_data, tickers)
        for r in results:
            if r: market_data.append(r)
            
    # 2. Analyze Sentiment (Sequential)
    final_results = []
    progress = st.progress(0)
    
    for i, data in enumerate(market_data):
        ticker = data['ticker']
        status.write(f"🧠 Reading news for **{ticker}**...")
        
        # AI Analysis
        ai_analysis = analyze_sentiment(ticker, data['news'])
        
        final_results.append({
            "Ticker": ticker,
            "Price": data['price'],
            "24h Change": data['change'],
            "Sentiment": ai_analysis.get('sentiment', 'Neutral'),
            "Score": ai_analysis.get('score', 0),
            "Key Driver": ai_analysis.get('reason', 'N/A')
        })
        progress.progress((i+1)/len(market_data))
        
    progress.empty()
    status.success("Scan Complete!")
    
    # 3. Display Dashboard
    if final_results:
        df = pd.DataFrame(final_results)
        
        # Styling function to color code rows
        def color_sentiment(val):
            color = 'white'
            if val == 'Bullish': color = '#d4edda' # Light Green
            elif val == 'Bearish': color = '#f8d7da' # Light Red
            return f'background-color: {color}; color: black'

        # Metrics
        st.subheader("Market Pulse")
        col1, col2, col3 = st.columns(3)
        avg_sentiment = df['Score'].mean()
        
        col1.metric("Market Mood", "Greed" if avg_sentiment > 0 else "Fear", f"{avg_sentiment:.1f}")
        col2.metric("Top Gainer", df.loc[df['24h Change'].idxmax()]['Ticker'])
        col3.metric("Most Bullish", df.loc[df['Score'].idxmax()]['Ticker'])

        # Main Table
        st.subheader("Deep Dive")
        
        # Format for display
        st.dataframe(
            df.style.map(lambda x: "color: green" if x > 0 else "color: red", subset=['24h Change']),
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "24h Change": st.column_config.NumberColumn(format="%.2f%%"),
                "Score": st.column_config.ProgressColumn(
                    "Sentiment Score", 
                    min_value=-10, 
                    max_value=10, 
                    format="%d"
                )
            },
            use_container_width=True
        )