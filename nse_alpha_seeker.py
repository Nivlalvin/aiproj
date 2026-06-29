import streamlit as st
import pandas as pd
import requests
import ollama
import concurrent.futures
from bs4 import BeautifulSoup
import yfinance as yf
import json
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Alpha Seeker (Fixed)", layout="wide")

# Session for faster, persistent connections
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Watchlist")
    st.caption("Use commas to separate tickers.")
    
    st.subheader("🇰🇪 NSE (Kenya)")
    # Note: Using standard NSE tickers
    nse_input = st.text_input("NSE Tickers", value="SCOM, KCB, EQTY, EABL")
    
    st.subheader("🌎 Global (US/Crypto)")
    global_input = st.text_input("Global Tickers", value="BTC-USD, NVDA, TSLA")

# --- ROBUST DATA FETCHING ---

@st.cache_data(ttl=300)
def fetch_nse_robust():
    """
    Scrapes AFX using BeautifulSoup directly (More reliable than pandas.read_html).
    """
    url = "https://afx.kwayisi.org/nse/"
    data_map = {}
    
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main table (usually the one with class 't')
        table = soup.find('table')
        if not table:
            return {}
            
        rows = table.find_all('tr')
        
        for row in rows[1:]: # Skip header
            cols = row.find_all('td')
            if len(cols) >= 4:
                # AFX Structure: [Ticker, Name, Volume, Price, Change, ...]
                ticker = cols[0].get_text(strip=True).upper()
                name = cols[1].get_text(strip=True)
                price_text = cols[3].get_text(strip=True).replace(',', '')
                change_text = cols[4].get_text(strip=True).replace('%', '').replace('+', '')
                
                try:
                    price = float(price_text)
                    change = float(change_text) if change_text not in ['-', ''] else 0.0
                    
                    data_map[ticker] = {
                        "price": price,
                        "change": change,
                        "name": name
                    }
                except:
                    continue
                    
        return data_map
    except Exception as e:
        st.error(f"NSE Scrape Error: {e}")
        return {}

def fetch_google_news_rss(query):
    """
    Fetches latest news from Google News RSS (Works for ANY asset, mostly reliable).
    """
    try:
        # Sanitize query for URL
        clean_query = query.replace(" ", "%20")
        rss_url = f"https://news.google.com/rss/search?q={clean_query}+stock+news&hl=en-KE&gl=KE&ceid=KE:en"
        
        response = session.get(rss_url, timeout=4)
        soup = BeautifulSoup(response.content, 'xml') # XML parser for RSS
        
        items = soup.find_all('item', limit=3)
        news_list = []
        
        for item in items:
            title = item.title.text
            # Clean up title (Google adds " - Source Name" at the end)
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            news_list.append(title)
            
        return news_list
    except:
        return []

def fetch_global_data(tickers):
    """
    Fetches Price from YFinance, but News from Google.
    """
    data_map = {}
    
    for ticker in tickers:
        try:
            # 1. Get Price (YFinance)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[0]
                change = ((curr - prev) / prev) * 100
                
                # 2. Get News (Google RSS fallback)
                # Use the full name if possible for better news results, else ticker
                search_term = ticker
                if ticker == "BTC-USD": search_term = "Bitcoin"
                
                news = fetch_google_news_rss(search_term)
                
                data_map[ticker] = {
                    "price": round(curr, 2),
                    "change": round(change, 2),
                    "news": news
                }
        except:
            continue
    return data_map

# --- AI ANALYSIS ---

def analyze_market_sentiment(ticker, price, change, news_list):
    """
    Analyzes sentiment based on Price Action + News Headlines.
    """
    news_text = " | ".join(news_list) if news_list else "No recent news."
    
    prompt = f"""
    Asset: {ticker}
    Price Change: {change}%
    Headlines: {news_text}
    
    Task: Return JSON.
    - "sentiment": "Bullish", "Bearish", or "Neutral"
    - "action": "Buy", "Sell", or "Hold"
    - "reason": Max 8 words explanation.
    """
    
    try:
        response = ollama.chat(
            model='llama3.2',
            format='json',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return json.loads(response['message']['content'])
    except:
        return {"sentiment": "Neutral", "action": "Hold", "reason": "AI Error"}

# --- MAIN APP UI ---

st.title("🦁 Alpha Seeker: Nairobi & Global")
st.markdown("Live Sentiment Analysis using **Google News** & **Llama 3.2**.")

if st.button("🚀 Start Scan"):
    
    # 1. FETCH DATA
    status = st.empty()
    progress = st.progress(0)
    
    status.info("📡 Scraping Nairobi Securities Exchange...")
    nse_db = fetch_nse_robust()
    
    status.info("📡 Fetching Global Markets...")
    global_tickers = [t.strip().upper() for t in global_input.split(',')]
    global_db = fetch_global_data(global_tickers)
    
    # 2. PREPARE TASKS
    nse_tickers = [t.strip().upper() for t in nse_input.split(',')]
    tasks = []
    
    # Add NSE Tasks
    for t in nse_tickers:
        if t in nse_db:
            # Fetch specific news for this Kenyan stock
            specific_news = fetch_google_news_rss(f"{nse_db[t]['name']} Kenya")
            tasks.append({
                "type": "NSE",
                "ticker": t,
                "price": nse_db[t]['price'],
                "change": nse_db[t]['change'],
                "news": specific_news
            })
    
    # Add Global Tasks
    for t, data in global_db.items():
        tasks.append({
            "type": "Global",
            "ticker": t,
            "price": data['price'],
            "change": data['change'],
            "news": data['news']
        })
        
    # 3. RUN ANALYSIS
    results = []
    total = len(tasks)
    
    for i, task in enumerate(tasks):
        status.write(f"🧠 Analyzing **{task['ticker']}**...")
        
        ai = analyze_market_sentiment(
            task['ticker'], 
            task['price'], 
            task['change'], 
            task['news']
        )
        
        results.append({
            "Market": task['type'],
            "Ticker": task['ticker'],
            "Price": task['price'],
            "Change": f"{task['change']}%",
            "Sentiment": ai.get('sentiment'),
            "Action": ai.get('action'),
            "Reason": ai.get('reason'),
            "News Context": task['news'][0] if task['news'] else "N/A"
        })
        progress.progress((i+1)/total)
        
    progress.empty()
    status.success("Done!")
    
    # 4. DISPLAY
    if results:
        df = pd.DataFrame(results)
        
        # Split Tables
        st.subheader("🇰🇪 Nairobi Stocks")
        st.dataframe(df[df['Market']=="NSE"].drop(columns=['Market']), use_container_width=True)
        
        st.subheader("🌎 Global Markets")
        st.dataframe(df[df['Market']=="Global"].drop(columns=['Market']), use_container_width=True)