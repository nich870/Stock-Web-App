import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as gr
from plotly.subplots import make_subplots

# Set up mobile screen layout configuration
st.set_page_config(page_title="AlgoScanner", layout="centered")

# 1. INITIALIZE LOGIN STATE TRACKING
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def check_password():
    """Verifies user inputs against secure dashboard environment secrets."""
    if st.session_state["password_input"] == st.secrets["APP_PASSWORD"]:
        st.session_state["authenticated"] = True
        del st.session_state["password_input"] # Clear password cache
    else:
        st.error("❌ Invalid Access Token. Please try again.")

# 2. RENDER THE LOGIN INTERFACE IF NOT AUTHENTICATED
if not st.session_state["authenticated"]:
    st.title("🔒 Secure Gateway Access")
    st.text_input("Enter Portfolio Password:", type="password", key="password_input", on_change=check_password)
    st.stop() # HALTS SCRIPT EXECUTION

# ==============================================================================
# 3. AUTHENTICATED DASHBOARD AREA 
# ==============================================================================
col_title, col_logout = st.columns([0.8, 0.2])
with col_title:
    st.title("📊 AlgoScanner Pro")
with col_logout:
    st.write("") 
    if st.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

# User Input Array accessible directly from a mobile layout interface
tickers_input = st.text_input("Enter Tickers (comma separated):", "SPY, AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

if st.button("⚡ Run Daily Market Scan"):
    with st.spinner("Downloading and processing market data matrices..."):
        # Fetch an extended timeline (start in 2024) to ensure the 200-day rolling math is fully populated
        raw_data = yf.download(tickers, start="2024-01-01", group_by="ticker")
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
                
                high_low = data["High"] - data["Low"]
                high_close = (data["High"] - data["Close"].shift(1)).abs()
                low_close = (data["Low"] - data["Close"].shift(1)).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data["ATR"] = true_range.rolling(window=14).mean()
                
                # 200-Day Structural Trend Line Math
                data["Long_Trend"] = data["Close"].rolling(window=200).mean()
                
                # Filter down to our graphing presentation window (Nov 2025 to July 2026)
                data = data.loc["2025-11-01":].copy()
                
                # Anti-Falling Knife Trigger Logic & Stop-Loss Simulation Loop
                data["Position"] = 0
                data["Stop_Line"] = np.nan
                current_position = 0
                highest_price = 0.0
                atr_multiplier = 2.5
                
                for i in range(1, len(data)):
                    c_price = data["Close"].iloc[i]
                    p_price = data["Close"].iloc[i - 1]
                    c_rsi = data["RSI"].iloc[i]
                    c_sup = data["Support"].iloc[i]
                    c_res = data["Resistance"].iloc[i]
                    c_atr = data["ATR"].iloc[i]
                    c_trend = data["Long_Trend"].iloc[i]
                    
                    # Safety Filter Check
                    is_trending_bullish = pd.notna(c_trend) and (c_price > c_trend)
                    
                    is_oversold = (c_rsi < 35) or (c_price <= c_sup * 1.01)
                    has_turned_up = c_price > p_price
                    
                    # Long buying signals are strictly forbidden if under the 200d trend line
                    buy_trigger = is_oversold and has_turned_up and is_trending_bullish
                    sell_trigger = (c_rsi > 70) or (c_price >= c_res * 0.99)
                    
                    if current_position == 0:
                        if buy_trigger:
                            current_position = 1
                            highest_price = c_price
                    else:
                        if c_price > highest_price:
                            highest_price = c_price
                        
                        stop_floor = highest_price - (c_atr * atr_multiplier)
                        data.iloc[i, data.columns.get_loc("Stop_Line")] = stop_floor
                        
                        if c_price <= stop_floor or sell_trigger:
                            current_position = 0
                            
                    data.iloc[i, data.columns.get_loc("Position")] = current_position

                # Extract latest parameters for daily tracking display
                latest = data.iloc[-1]
                prev = data.iloc[-2]
                
                # Identify shift coordinates for marker dots
                data["State_Shift"] = data["Position"].diff()
                buy_signals = data[data["State_Shift"] == 1]
                sell_signals = data[(data["State_Shift"] == -1) | ((data["Position"].shift(1) == 1) & (data["Position"] == 0))]

                if latest["Position"] == 1 and prev["Position"] == 0: recommendation = "🟢 BUY"
                elif latest["Position"] == 0 and prev["Position"] == 1: recommendation = "🔴 SELL (Exit)"
                elif latest["Position"] == 1: recommendation = "🔵 HOLD LONG"
                else: recommendation = "⚪ CASH (Wait)"
                
                stop_print = f"${highest_price - (latest['ATR'] * atr_multiplier):.2f}" if latest["Position"] == 1 else "N/A"
                
                # Check current regime status for the display block
                is_bull_market = pd.notna(latest["Long_Trend"]) and (latest["Close"] > latest["Long_Trend"])
                regime_string = "Bull Market (Buying Allowed)" if is_bull_market else "Bear Market (Buying Locked)"
                
                summary_rows.append({
                    "Ticker": ticker, "Price": f"${latest['Close']:.2f}",
                    "RSI": f"{latest['RSI']:.1f}", "Regime": regime_string,
                    "Signal": recommendation, "ATR Stop Target": stop_print
                })
                
                # 4. Render Mobile UI Container blocks
                st.write("---")
                st.subheader(f"📈 Framework Analysis: {ticker}")
                
                # DISPLAY THE DYNAMIC REGIME BANNER
                if is_bull_market:
                    st.success(f"🟢 **Market Regime**: {ticker} is in a long-term **Bull Market**. Dip-buying strategies are active.")
                else:
                    st.error(f"⚠️ **Market Regime Alert**: {ticker} is in a structural **Bear Market**. Long-entry logic has been **paused** to prevent catching a falling knife.")
                
                # Construct mobile-optimized charts using Plotly
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
                
                # Top Subplot: Prices, Channels, Stops, and the 200d Trend Line
                fig.add_trace(gr.Scatter(x=data.index, y=data["Close"], name="Price", line=dict(color="black")), row=1, col=1)
                fig.add_trace(gr.Scatter(x=data.index, y=data["Long_Trend"], name="200d SMA Trend", line=dict(color="blue", width=2)), row=1, col=1)
                fig.add_trace(gr.Scatter(x=data.index, y=data["Support"], name="Support", line=dict(color="green", dash="dash"), opacity=0.15), row=1, col=1)
                fig.add_trace(gr.Scatter(x=data.index, y=data["Resistance"], name="Resistance", line=dict(color="red", dash="dash"), opacity=0.15), row=1, col=1)
                fig.add_trace(gr.Scatter(x=data.index, y=data["Stop_Line"], name="ATR Stop", line=dict(color="orange", width=2)), row=1, col=1)
                
                # Marker overlays
                if not buy_signals.empty:
                    fig.add_trace(gr.Scatter(x=buy_signals.index, y=buy_signals["Close"], mode="markers", name="BUY", marker=dict(color="limegreen", size=10, line=dict(width=1, color="white"))), row=1, col=1)
                if not sell_signals.empty:
                    fig.add_trace(gr.Scatter(x=sell_signals.index, y=sell_signals["Close"], mode="markers", name="SELL", marker=dict(color="crimson", size=10, line=dict(width=1, color="white"))), row=1, col=1)
                
                # Bottom Subplot: Relative Strength Index Panel
                fig.add_trace(gr.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", line_width=1.5, row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", line_width=1.5, row=2, col=1)
                fig.update_yaxes(range=[10, 90], row=2, col=1)
                
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error executing ticker data pipeline for {ticker}")
        
        # Display the Summary Scorecard up top
        st.subheader("📋 Execution Scorecard Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
