import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as gr
from plotly.subplots import make_subplots

# Set up mobile screen layout configuration
st.set_page_config(page_title="AlgoScanner", layout="centered")
st.title("📊 Algorithmic Trading Scanner")

# User Input Array accessible directly from a mobile layout interface
tickers_input = st.text_input("Enter Tickers (comma separated):", "SPY, AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

if st.button("⚡ Run Daily Market Scan"):
    with st.spinner("Downloading Wall Street market data matrices..."):
        # Fetch data blocks using yfinance engine
        raw_data = yf.download(tickers, start="2025-11-01", group_by="ticker")
        
        # Iterative loop to process signals and construct mobile interactive tables
        summary_rows = []
        
        for ticker in tickers:
            try:
                data = raw_data[ticker].dropna().copy()
                if data.empty: continue
                
                # Math Indicators block (Support, Resistance, RSI, ATR)
                data["Support"] = data["Close"].rolling(window=20).min()
                data["Resistance"] = data["Close"].rolling(window=20).max()
                
                delta = data["Close"].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                data["RSI"] = 100 - (100 / (1 + (avg_gain / avg_loss)))
                
                # Anti-Falling Knife Trigger Logic
                latest = data.iloc[-1]
                prev = data.iloc[-2]
                
                is_oversold = (latest["RSI"] < 35) or (latest["Close"] <= latest["Support"] * 1.01)
                has_turned_up = latest["Close"] > prev["Close"]
                is_overbought = (latest["RSI"] > 70) or (latest["Close"] >= latest["Resistance"] * 0.99)
                
                if is_oversold and has_turned_up: recommendation = "🟢 BUY"
                elif is_overbought: recommendation = "🔴 SELL"
                else: recommendation = "⚪ HOLD"
                
                summary_rows.append({
                    "Ticker": ticker, "Price": f"${latest['Close']:.2f}",
                    "RSI": f"{latest['RSI']:.1f}", "Signal": recommendation
                })
                
                # 2. Construct mobile-optimized charts using Plotly
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
                fig.add_trace(gr.Scatter(x=data.index, y=data["Close"], name="Price", line=dict(color="black")), row=1, col=1)
                fig.add_trace(gr.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
                
                # Format visual panels for clean touch-screen navigation
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.subheader(f"📈 Framework Analysis: {ticker}")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error executing ticker data pipeline for {ticker}")
        
        # Display the Summary Scorecard up top
        st.subheader("📋 Execution Scorecard Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)