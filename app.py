import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as gr
from plotly.subplots import make_subplots

# Calculator for testing strategies
def calculate_strategy_performance(data, initial_investment=1000.0):
    """
    Computes historical percent returns and cash compounding metrics 
    for a baseline capital stake based on algorithmic position flags.
    """
    # 1. Calculate daily percentage change of the underlying stock asset
    data["Market_Returns"] = data["Close"].pct_change()
    
    # 2. Shift the position flags by 1 day (You capture tomorrow's return based on tonight's signal)
    data["Strategy_Returns"] = data["Market_Returns"] * data["Position"].shift(1)
    
    # 3. Mathematically compound the returns over the historical timeline matrix
    data["Cumulative_Strategy"] = (1 + data["Strategy_Returns"].fillna(0)).cumprod()
    
    # 4. Extract the final compound output values
    final_value = initial_investment * data["Cumulative_Strategy"].iloc[-1]
    percent_return = (data["Cumulative_Strategy"].iloc[-1] - 1) * 100
    
    return percent_return, final_value


# Set up mobile screen layout configuration
st.set_page_config(page_title="Gebauer Stock Scanner", layout="centered")

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

with col_logout:
    st.write("") 
    if st.button("🚪 Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

app_mode = st.sidebar.selectbox("Choose Application:", ["Market Scanner", "Strategy Discovery Scout", "Nick's Account Ledger", "Jacob's Account Ledger"])
if app_mode == "Market Scanner":
    # User Input Array accessible directly from a mobile layout interface
    with col_title:
        st.title("📊 Stock MarketScanner")
    tickers_input = st.text_input("Enter Tickers (comma separated):", "VTI, XRP, NVDA, SPY, AAPL, MSFT, GOOGL, AMZN, AMD, MU, TSLA, META, AVGO")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    if st.button("⚡ Run Daily Market Scan"):
        with st.spinner("Downloading and processing market data matrices..."):
            # Fetch an extended timeline (start in 2024) to ensure the 200-day rolling math is fully populated
            raw_data = yf.download(tickers, start="2024-01-01", group_by="ticker")
            summary_rows = []
            
            def optimize_window_parameter(stock_df):
                """
                Simulates performance across windows 30-120 to find the most robust setting 
                based on the preceding 6 months of capital growth data.
                """
                best_window = 65 # Default structural quarter anchor if optimization ties
                max_strategy_return = -999.0
                
                # Isolate a trailing 180-day optimization training slice matrix
                train_df = stock_df.tail(180).copy()
                if len(train_df) < 100:
                    return best_window

                stock_df_copy = stock_df.copy()
                stock_df_copy["Long_Trend"] = stock_df_copy["Close"].rolling(window=200).mean()
                train_df["Long_Trend"] = stock_df_copy["Long_Trend"].loc[train_df.index]

                # Scan the parameter spectrum using a 5-day stride for robustness
                for test_w in range(30, 121, 5):
                    sim = train_df.copy()
                    sim["Support"] = sim["Close"].rolling(window=test_w).min()
                    sim["Resistance"] = sim["Close"].rolling(window=test_w).max()
                    
                    delta = sim["Close"].diff()
                    gain = delta.where(delta > 0, 0.0)
                    loss = -delta.where(delta < 0, 0.0)
                    sim["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
                    
                    sim["Position"] = 0
                    curr_pos = 0
                    fixed_stop = np.nan
                    
                    for i in range(1, len(sim)):
                        c_p = sim["Close"].iloc[i]
                        p_p = sim["Close"].iloc[i - 1]
                        c_rsi = sim["RSI"].iloc[i]
                        c_sup = sim["Support"].iloc[i]
                        c_res = sim["Resistance"].iloc[i]
                        c_trend = sim["Long_Trend"].iloc[i]

                        is_trending_bullish = pd.notna(c_trend) and (c_p > c_trend)
                        
                        buy = (c_rsi < 30 or c_p <= c_sup * 1.01) and is_trending_bullish # and (c_p > p_p)
                        sell = (c_rsi > 70 or c_p >= c_res * 0.99)
                        
                        if curr_pos == 0:
                            if buy:
                                curr_pos = 1
                                fixed_stop = c_sup * 0.98
                        else:
                            if c_p <= fixed_stop or sell:
                                curr_pos = 0
                        sim.iloc[i, sim.columns.get_loc("Position")] = curr_pos
                        
                    sim["M_Ret"] = sim["Close"].pct_change()
                    sim["S_Ret"] = sim["M_Ret"] * sim["Position"].shift(1)
                    cum_prod = (1 + sim["S_Ret"].fillna(0)).cumprod().iloc[-1]
                    
                    if cum_prod > max_strategy_return:
                        max_strategy_return = cum_prod
                        best_window = test_w
                        
                return best_window

            for ticker in tickers:
                try:
                    data = raw_data[ticker].dropna().copy()
                    if data.empty: continue
                    
                    # Execute adaptive walk forward optimization to find the best rolling window parameter for this ticker
                    # optimal_w = optimize_window_parameter(data)
                    # Math Indicators block (Support, Resistance, RSI, ATR)
                    data["Support"] = data["Close"].rolling(window=50).min()
                    data["Resistance"] = data["Close"].rolling(window=50).max()
                    
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
                    # data = data.loc["2024-01-01":].copy()
                    
                    # Anti-Falling Knife Trigger Logic & Stop-Loss Simulation Loop
                    data["Position"] = 0
                    data["Stop_Line"] = np.nan
                    current_position = 0
                    fixed_stop_floor = np.nan
                    last_loss_date = None # Tracks the date of the last loss
                    
                    for i in range(1, len(data)):
                        c_date = data.index[i]
                        c_price = data["Close"].iloc[i]
                        p_price = data["Close"].iloc[i - 1]
                        c_rsi = data["RSI"].iloc[i]
                        c_sup = data["Support"].iloc[i]
                        c_res = data["Resistance"].iloc[i]
                        c_trend = data["Long_Trend"].iloc[i]
                    
                        if last_loss_date is not None:
                            days_since_loss = (c_date - last_loss_date).days
                        else:
                            days_since_loss = 999 # Infinite buffer if there has been no previous loss

                        # Wash-Sale Safety Check: Locks buying if loss within 30 days
                        is_wash_sale_risk = days_since_loss <= 30

                        # Safety Filter Check
                        is_trending_bullish = pd.notna(c_trend) and (c_price > c_trend)
                        
                        is_oversold = (c_rsi < 30) or (c_price <= c_sup * 1.01)
                        has_turned_up = c_price > p_price
                        
                        # Long buying signals are strictly forbidden if under the 200d trend line
                        buy_trigger = is_oversold and is_trending_bullish and not is_wash_sale_risk # and has_turned_up
                        sell_trigger = (c_rsi > 70) or (c_price >= c_res * 0.99)
                        
                        if current_position == 0:
                            if buy_trigger:
                                current_position = 1
                                # Lock the stop floor to the 50 day rolling window minus 2%
                                fixed_stop_floor = c_sup * 0.98
                                entry_price = c_price
                        else:
                            data.iloc[i, data.columns.get_loc("Stop_Line")] = fixed_stop_floor
                            
                            if c_price <= fixed_stop_floor or sell_trigger:
                                current_position = 0
                                if c_price < entry_price:
                                    last_loss_date = c_date
                                
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
                    elif latest["Position"] == 0 and is_wash_sale_risk:
                        recommendation = "🟡 WASH SALE RISK"
                    elif latest["Position"] == 1: recommendation = "🔵 HOLD LONG"
                    else: recommendation = "⚪ CASH (Wait)"
                    
                    stop_print = f"${fixed_stop_floor:.2f}" if latest["Position"] == 1 else "N/A"
                    
                    # Check current regime status for the display block
                    is_bull_market = pd.notna(latest["Long_Trend"]) and (latest["Close"] > latest["Long_Trend"])
                    regime_string = "Bull Market (Buying Allowed)" if is_bull_market else "Bear Market (Buying Locked)"
                    
                    pct_return, final_dollar_worth = calculate_strategy_performance(data, initial_investment=1000.0)
                    
                    summary_rows.append({
                        "Ticker": ticker, "Price": f"${latest['Close']:.2f}",
                        "RSI": f"{latest['RSI']:.1f}", "Regime": regime_string,
                        "Signal": recommendation, "StructuralStop Target": stop_print,
                        "Strategy Return %": f"{pct_return:+.1f}%", # <-- NEW PERFORMANCE COLUMN
                        "$1,000 Growth Value": f"${final_dollar_worth:,.2f}" # <-- NEW PERFORMANCE COLUMN
                    })
                    
                    # 4. Render Mobile UI Container blocks
                    st.write("---")
                    st.subheader(f"📈 Framework Analysis: {ticker}")
                    
                    # DISPLAY THE DYNAMIC REGIME BANNER
                    if is_bull_market:
                        st.success(f"🟢 **Market Regime**: {ticker} is in a long-term **Bull Market**. Dip-buying strategies are active.")
                    else:
                        st.error(f"⚠️ **Market Regime Alert**: {ticker} is in a structural **Bear Market**. Long-entry logic has been **paused** to prevent losses.")
                    
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

                    today_timestamp = data.index[-1]
                    initial_zoom_start = today_timestamp - pd.DateOffset(months=8)  # Maybe change this to 8 months prior to today dynamically in the future
                    fig.update_layout(
                        template="plotly_white",
                        height=450,
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False,

                        # Price Viewport constraints
                        xaxis=dict(
                            range=[initial_zoom_start, today_timestamp], # Forces default zoom to the last 8 months of data
                            rangeslider=dict(
                                visible=True, # Spins up an interactive range slider for mobile users (Maybe try without sometimes)
                                thickness=0.04
                            ),
                            type="date"
                        ),

                        # RSI Panel synchronization
                        xaxis2=dict(
                            range=[initial_zoom_start, today_timestamp],
                            type="date"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error executing ticker data pipeline for {ticker}")
            
            # Display the Summary Scorecard up top
            st.subheader("📋 Execution Scorecard Summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# ==============================================================================
# PANEL MODE: Strategy Discovery Scout
# ==============================================================================
elif app_mode == "Strategy Discovery Scout":
    st.title("Strategy Discovery Scout")
    st.write("This scouting panel systematically screens major market equities to identify high-probability bull market pullbacks for tactical practice.")
    
    # Curated broad market scanning pool array spanning diverse sectors (Entire S&P 500)
    scout_pool = [
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AEM', 'AFT', 'AFRM', 'AGCO', 'AGN', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APTV', 'ARE', 'ARES', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBDC', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCK', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CINT', 'CL', 'CLX', 'CMA', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISH', 'DLR', 'DLTR', 'DOC', 'DOCU', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FI', 'FICO', 'FIS', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWRE', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PCP', 'PDD', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PX', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVMD', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SPLK', 'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TI', 'TJX', 'TMO', 'TMUS', 'TROW', 'TRGP', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VALE', 'VEEV', 'VERU', 'VFC', 'VICI', 'VLO', 'VMC', 'VOD', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WEC', 'WEL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
    ]
    
    if st.button("🚀 Execute Broad Market Pullback Scan", use_container_width=True):
        with st.spinner("Analyzing volume channels, velocity curves, and trend lines..."):
            
            # Fetch a 1-year data chunk to fully populate our 200-day SMA matrices
            raw_scout_data = yf.download(scout_pool, period="1y", group_by="ticker")
            discovered_opportunities = []
            
            for ticker in scout_pool:
                try:
                    data = raw_scout_data[ticker].dropna().copy()
                    if len(data) < 200: continue
                    
                    # 1. Technical Math Core Processing
                    data["200_SMA"] = data["Close"].rolling(window=200).mean()
                    data["20_Min"] = data["Close"].rolling(window=20).min()
                    # Calculate Trading Volume
                    data["Daily_Volume"] = data["Volume"].iloc[-1]
                    
                    delta = data["Close"].diff()
                    gain = delta.where(delta > 0, 0.0)
                    loss = -delta.where(delta < 0, 0.0)
                    data["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
                    
                    # Extract final tracking variables
                    latest = data.iloc[-1]
                    price = latest["Close"]
                    rsi = latest["RSI"]
                    sma_200 = latest["200_SMA"]
                    support_20d = latest["20_Min"]
                    
                    # 2. EVALUATE THE IDEAL PULLBACK MATRIX CONDITIONS
                    is_trading_volume_healthy = latest["Daily_Volume"] >= 1000000  # Minimum liquidity threshold
                    is_bull_market = price > sma_200
                    is_oversold_velocity = rsi <= 40
                    is_near_support = price <= (support_20d * 1.03) # Within 3% of its recent floor
                    
                    # If the stock passes our quality filters, flag it as a premium practice target
                    if is_bull_market and is_oversold_velocity and is_trading_volume_healthy:
                        # Calculate the proximity to the support line to display trade quality
                        discount_pct = ((price - support_20d) / support_20d) * 100
                        
                        discovered_opportunities.append({
                            "Practice Ticker": ticker,
                            "Current Price": f"${price:.2f}",
                            "RSI Value": f"{rsi:.1f}",
                            "Trading Volume": f"{latest['Daily_Volume']:,}",
                            "Proximity to Support": f"+{discount_pct:.1f}%",
                            "Action Strategy": "🟢 ACTIVE DIP - Practice Layering Buy Orders" if is_near_support else "🔵 WATCHING - Awaiting Support Test"
                        })
                except Exception as e:
                    pass
            
            # 3. RENDER RESULTS INTERFACE TO PHONE SCREEN
            st.write("---")
            if discovered_opportunities:
                st.subheader("🎯 Qualified Practice Targets Found")
                st.write("These assets match your setup rules. Use your **Ledger Pro** tab to open fake paper positions in these stocks to practice managing your structural stop-loss floors risk-free.")
                
                # Render responsive data grid layout
                df_scout = pd.DataFrame(discovered_opportunities)
                st.dataframe(df_scout, use_container_width=True)
                
                # Dynamic Touchscreen Card loops
                for opt in discovered_opportunities:
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="background-color:#F0F8FF; border-radius:8px; padding:12px; margin-bottom:12px; border-left: 5px solid #1E90FF;">
                                <h4 style="margin:0; color:#1E3F66;">🎯 Target: {opt['Practice Ticker']} — {opt['Current Price']}</h4>
                                <p style="margin:4px 0 0 0; font-size:14px; color:#333333;">
                                    <b>RSI Velocity</b>: {opt['RSI Value']} | <b>Floor Proximity</b>: {opt['Proximity to Support']}<br>
                                    <b>Strategy Focus</b>: {opt['Action Strategy']}
                                </p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.info("**Market Analysis Complete**: No stocks are currently experiencing an active bull market pullback. The broader market is either too overextended (RSI > 40) or in a structural bear market trend under the 200-day line. Capital protection is active; preserve your uninvested cash.")

elif app_mode == "Nick's Account Ledger":
    # Render Account Ledger interface
    import streamlit as st
    import pandas as pd
    import os
    import plotly.graph_objects as gr
    from datetime import datetime

    # Mobile layout configuration
    st.set_page_config(page_title="AlgoLedger Pro", layout="centered")

    # 1. INITIALIZE DATA STORAGE FILES
    LEDGER_FILE = "trading_ledger.csv"
    BALANCE_FILE = "capital_balance.txt"
    EQUITY_HISTORY_FILE = "equity_history.csv"
    INITIAL_STARTING_CASH = 20.00

    DIVIDEND_DATABASE = {
        "MSFT": 0.91, "AVGO": 0.65, "NVDA": 0.25, "AAPL": 0.25, "META": 0.50, "GOOGL": 0.20, "MU": 0.15
    }

    def load_data():
        if os.path.exists(LEDGER_FILE):
            df = pd.read_csv(LEDGER_FILE)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df
        return pd.DataFrame(columns=["ID", "Date", "Ticker", "Type", "Price", "Capital", "Shares", "PnL", "Status"])

    def load_cash_balance():
        if os.path.exists(BALANCE_FILE):
            with open(BALANCE_FILE, "r") as f:
                return float(f.read().strip())
        return INITIAL_STARTING_CASH

    def load_equity_history():
        if os.path.exists(EQUITY_HISTORY_FILE):
            df = pd.read_csv(EQUITY_HISTORY_FILE)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df
        # Initialize with Day 0 baseline milestone
        return pd.DataFrame([{"Date": datetime.now().date(), "Total_Net_Worth": INITIAL_STARTING_CASH}])

    def save_all(df, cash_balance, equity_history_df):
        df.to_csv(LEDGER_FILE, index=False)
        with open(BALANCE_FILE, "w") as f:
            f.write(f"{cash_balance:.2f}")
        equity_history_df.to_csv(EQUITY_HISTORY_FILE, index=False)

    # Load global variables into active operational cache memory
    df_ledger = load_data()
    current_cash = load_cash_balance()
    df_equity = load_equity_history()

    # ==============================================================================
    # 2. CALCULATE PORTFOLIO FINANCIAL BALANCES
    # ==============================================================================
    st.title("📓 Nick's Private Trading Ledger")

    # ==============================================================================
    # TAX QUARTER DEADLINE MONITOR (IRS Form 1040-ES Framework)
    # ==============================================================================
    current_date = datetime.now().date()
    current_year = current_date.year

    # Official IRS Deadline Arrays
    deadlines = [
        {"Quarter": "Q1", "Date": datetime(current_year, 4, 15).date(), "Voucher": "Voucher 1"},
        {"Quarter": "Q2", "Date": datetime(current_year, 6, 15).date(), "Voucher": "Voucher 2"},
        {"Quarter": "Q3", "Date": datetime(current_year, 9, 15).date(), "Voucher": "Voucher 3"},
        {"Quarter": "Q4", "Date": datetime(current_year + 1, 1, 15).date(), "Voucher": "Voucher 4"}
    ]

    # Identify the upcoming deadline milestone row
    upcoming_deadline = None
    for d in deadlines:
        if current_date <= d["Date"]:
            upcoming_deadline = d
            break

    if upcoming_deadline:
        days_remaining = (upcoming_deadline["Date"] - current_date).days
        deadline_str = upcoming_deadline["Date"].strftime("%B %d, %Y")
        
        # Render dynamic, touchscreen-friendly calendar warning callouts
        if days_remaining <= 14:
            st.error(f"🚨 **🚨 DANGER: IRS TAX DEADLINE CRITICAL** — Your **{upcoming_deadline['Quarter']} ({upcoming_deadline['Voucher']})** estimated tax voucher payment is due in exactly **{days_remaining} days** ({deadline_str}). Move your calculated tax reserve cache out of your broker account immediately.")
        else:
            st.info(f"📅 **Tax Calendar Reminder**: The next estimated IRS payment deadline is for **{upcoming_deadline['Quarter']}** in **{days_remaining} days** ({deadline_str}).")
    
    open_positions = df_ledger[df_ledger["Status"] == "OPEN"]
    total_invested = open_positions["Capital"].sum()
    closed_positions = df_ledger[(df_ledger["Status"] == "CLOSED") & (df_ledger["Type"] == "SELL")]
    net_realized_pnl = closed_positions["PnL"].sum()
    dividend_rows = df_ledger[df_ledger["Type"] == "DIVIDEND"]
    total_dividends_collected = dividend_rows["Capital"].sum()
    total_portfolio_value = current_cash + total_invested

    # Sync performance changes back to the performance chart logging system
    today_date = datetime.now().date()
    if df_equity.empty or df_equity.iloc[-1]["Date"] != today_date:
        new_snapshot = pd.DataFrame([{"Date": today_date, "Total_Net_Worth": total_portfolio_value}])
        df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)
    else:
        # Update today's existing row value to keep the curve pinpoint accurate
        df_equity.at[df_equity.index[-1], "Total_Net_Worth"] = total_portfolio_value
    df_equity.to_csv(EQUITY_HISTORY_FILE, index=False)

    # Mobile-responsive financial scorecards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚪ Uninvested Cash (Core Balance)", f"${current_cash:,.2f}")
        st.metric("🔵 Active Capital Deployed", f"${total_invested:,.2f}")
    with col2:
        st.metric("📈 Net Realized Profit/Loss", f"${net_realized_pnl:,.2f}", delta=f"{net_realized_pnl:+.2f}")
        st.metric("💼 Total Portfolio Net Worth", f"${total_portfolio_value:,.2f}")
    with col3:
        st.metric("💵 Total Dividends Collected", f"${total_dividends_collected:,.2f}", delta=f"+${total_dividends_collected:,.2f}" if total_dividends_collected > 0 else None)
    with col4:
        st.metric("Total Estimated Taxes", f"${net_realized_pnl * 0.16:,.2f}" if net_realized_pnl * 0.16 > 0 else None, help="Using 16% short-term capital gains tax estimate.")
    # ==============================================================================
    # 3. VISUAL PORTFOLIO PERFORMANCE LINE GRAPH CHART
    # ==============================================================================
    st.write("---")
    st.subheader("📈 Capital Growth Performance History")
    if len(df_equity) > 1:
        fig_equity = gr.Figure()
        fig_equity.add_trace(gr.Scatter(
            x=df_equity["Date"], 
            y=df_equity["Total_Net_Worth"], 
            mode="lines+markers", 
            name="Net Worth",
            line=dict(color="limegreen", width=3),
            marker=dict(size=6)
        ))
        fig_equity.update_layout(
            height=280, 
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickprefix="$")
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Performance trend graphics will populate here as trade logs generate data milestones over time.")

    st.write("---")

    # ==============================================================================
    # 4. ORDER ENTRY & LIQUIDATION AND CAPITAL BANK MODIFICATION ENGINE
    # ==============================================================================
    menu_selection = st.selectbox(
        "Choose Ledger Action Interface:", 
        ["🟢 Log New Buy Order", "🔴 Log Sell / Exit Order", "🏦 Account Capital Adjustments"]
    )

    if menu_selection == "🟢 Log New Buy Order":
        st.subheader("Execute Entry Position")
        with st.form("buy_form", clear_on_submit=True):
            b_date = st.date_input("Transaction Date", datetime.now().date())
            b_ticker = st.text_input("Ticker Symbol:").strip().upper()
            b_price = st.number_input("Actual Stock Buy Price ($):", min_value=0.01, step=0.01)
            b_capital = st.number_input("Total Money Invested ($):", min_value=1.00, step=100.00)
            submit_buy = st.form_submit_button("Submit Position Entry")
            
            if submit_buy:
                if not b_ticker:
                    st.error("Please enter a valid ticker symbol.")
                elif b_capital > current_cash:
                    st.error("Insufficient cash balance available in your uninvested core account.")
                else:
                    calculated_shares = b_capital / b_price
                    new_cash = current_cash - b_capital
                    new_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": b_date, "Ticker": b_ticker,
                        "Type": "BUY", "Price": b_price, "Capital": b_capital,
                        "Shares": calculated_shares, "PnL": 0.0, "Status": "OPEN"
                    }])
                    df_ledger = pd.concat([df_ledger, new_row], ignore_index=True)
                    save_all(df_ledger, new_cash, df_equity)
                    st.success(f"Successfully logged: {calculated_shares:.2f} shares of {b_ticker} acquired.")
                    st.rerun()

    elif menu_selection == "🔴 Log Sell / Exit Order":
        st.subheader("Execute Exit Position")
        open_options = df_ledger[df_ledger["Status"] == "OPEN"]
        
        if open_options.empty:
            st.info("You currently hold zero active positions in your open portfolio array.")
        else:
            position_list = open_options.apply(lambda r: f"{r['Ticker']} | Bought on {r['Date']} (${r['Capital']:.2f})", axis=1).tolist()
            selected_pos_str = st.selectbox("Select Position to Liquidate:", position_list)
            selected_idx = open_options.index[position_list.index(selected_pos_str)]
            target_row = df_ledger.loc[selected_idx]
            
            with st.form("sell_form", clear_on_submit=True):
                s_date = st.date_input("Liquidation Date", datetime.now().date())
                s_price = st.number_input("Actual Stock Sell Price ($):", min_value=0.01, step=0.01)

                captured_dividend = st.checkbox("Did you hold this stock past its official Ex-dividend date during this trade?")

                submit_sell = st.form_submit_button("Submit Position Liquidation")
                
                if submit_sell:
                    initial_capital = target_row["Capital"]
                    shares_held = target_row["Shares"]
                    ticker_symbol = target_row["Ticker"]
                    final_liquidation_value = shares_held * s_price
                    trade_pnl = final_liquidation_value - initial_capital
                    dividend_payout = 0.0
                    if ticker_symbol in DIVIDEND_DATABASE and captured_dividend:
                        dividend_payout = shares_held * DIVIDEND_DATABASE[ticker_symbol]


                    new_cash = current_cash + final_liquidation_value + dividend_payout
                    
                    df_ledger.at[selected_idx, "PnL"] = trade_pnl
                    df_ledger.at[selected_idx, "Status"] = "CLOSED"
                    
                    exit_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": s_date, "Ticker": ticker_symbol,
                        "Type": "SELL", "Price": s_price, "Capital": final_liquidation_value,
                        "Shares": shares_held, "PnL": trade_pnl, "Status": "CLOSED"
                    }])
                    df_ledger = pd.concat([df_ledger, exit_row], ignore_index=True)
                    
                    if dividend_payout > 0:
                        dividend_row = pd.DataFrame([{
                            "ID": int(datetime.now().timestamp()) + 1, "Date": s_date, "Ticker": "DIVIDEND",
                            "Type": "DIVIDEND", "Price": DIVIDEND_DATABASE[ticker_symbol], "Capital": dividend_payout,
                            "Shares": shares_held, "PnL": dividend_payout, "Status": "CLOSED"
                        }])
                        df_ledger = pd.concat([df_ledger, dividend_row], ignore_index=True)

                    # Append updated net worth path straight to performance plot array
                    new_snapshot = pd.DataFrame([{"Date": s_date, "Total_Net_Worth": new_cash + (total_invested - initial_capital)}])
                    df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)
                    
                    save_all(df_ledger, new_cash, df_equity)
                    st.success(f"Trade Closed. Performance Matrix Logged: {trade_pnl:+.2f}")
                    st.rerun()

    elif menu_selection == "🏦 Account Capital Adjustments":
        st.subheader("Manual Cash Capital Adjustments")
        st.write("Use this portal to record manual brokerage deposit transfers or external bank account cash withdrawals.")
        
        with st.form("adjustment_form", clear_on_submit=True):
            adj_type = st.radio("Adjustment Vector Type:", ["DEPOSIT (Add External Cash)", "WITHDRAWAL (Extract Bank Cash)"], horizontal=True)
            adj_amount = st.number_input("Transfer Capital Amount ($):", min_value=1.00, step=100.00)
            adj_date = st.date_input("Execution Date", datetime.now().date())
            submit_adj = st.form_submit_button("Finalize Balance Adjustment")
            
            if submit_adj:
                if "WITHDRAWAL" in adj_type and adj_amount > current_cash:
                    st.error("Execution failed: Requested transfer size exceeds current uninvested cash bounds.")
                else:
                    adjustment_mod = adj_amount if "DEPOSIT" in adj_type else -adj_amount
                    new_cash = current_cash + adjustment_mod
                    new_portfolio_value = total_portfolio_value + adjustment_mod
                    
                    # Append adjustments directly into transaction matrices logs
                    adj_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": adj_date, "Ticker": "CASH_ADJ",
                        "Type": "DEPOSIT" if "DEPOSIT" in adj_type else "WITHDRAW", "Price": 1.0, 
                        "Capital": adj_amount, "Shares": 0.0, "PnL": 0.0, "Status": "SYSTEM"
                    }])
                    df_ledger = pd.concat([df_ledger, adj_row], ignore_index=True)
                    # Sync capital shifts into performance graph data matrices tracks
                    new_snapshot = pd.DataFrame([{"Date": adj_date, "Total_Net_Worth": new_portfolio_value}])
                    df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)
                    save_all(df_ledger, new_cash, df_equity)
                    st.success(f"Balance adjustments updated successfully. New uninvested pool: ${new_cash:,.2f}")
                    st.rerun()
    # ==============================================================================
    # 5. VIEW HISTORICAL TRANSACTION BOOK
    # ==============================================================================
    st.write("---")
    st.subheader("🗃️ Historical Transaction Logbook")
    if df_ledger.empty:st.write("No transaction data logged yet.")
    else:
        display_df = df_ledger.copy().sort_values(by="Date", ascending=False)
        st.dataframe(display_df[["Date", "Ticker", "Type", "Price", "Capital", "PnL", "Status"]], use_container_width=True)

    # ==============================================================================
    # 5b. TAX-LOSS HARVESTING & DEDUCTION ANALYTICS
    # ==============================================================================
    st.write("---")
    st.subheader("🏛️ Tax-Loss Harvesting & Write-off Tracker")

    # 1. Isolate verified closed losses (Excluding cash adjustments and dividends)
    closed_stock_trades = df_ledger[(df_ledger["Status"] == "CLOSED") & (df_ledger["Ticker"] != "CASH_ADJ") & (df_ledger["Ticker"] != "DIVIDEND") & (df_ledger["Type"] == "BUY")]
    gross_gains = closed_stock_trades[closed_stock_trades["PnL"] > 0]["PnL"].sum()
    total_harvested_losses = abs(closed_stock_trades[closed_stock_trades["PnL"] < 0]["PnL"].sum())

    # 2. Calculate the IRS writing-off limits
    # The IRS allows you to offset 100% of capital gains, plus up to $3,000 of ordinary income
    if gross_gains > total_harvested_losses:
        taxable_net_gains = max(0.0, gross_gains - total_harvested_losses)
        ordinary_income_offset = 0.0
    else:
        taxable_net_gains = 0.0
        ordinary_income_offset = min(3000.0, total_harvested_losses - gross_gains)

    estimated_tax_savings = total_harvested_losses * 0.16

    # 3. Render mobile-optimized tax accounting grid layout
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.metric("📉 Total Harvested Capital Losses", f"${total_harvested_losses:,.2f}", delta="- Tax Deduction", delta_color="inverse")
        st.metric("🛡️ Ordinary Income Offset Value", f"${ordinary_income_offset:,.2f}", help="$3,000 max that can be written off standard W-2 income.")
    with t_col2:
        st.metric("⚖️ Net Taxable Short-Term Gains", f"${taxable_net_gains:,.2f}")
        st.metric("💵 Estimated Cash Tax Savings", f"${estimated_tax_savings:,.2f}", delta="+ Saved Core Cash")

    # Display a clean, expandable warning table pinpointing exactly which trades generated your tax write-offs
    if not closed_stock_trades.empty:
        with st.expander("📋 Review Active Loss Deductions (Form 1099-B Audit)"):
            st.dataframe(closed_stock_trades[closed_stock_trades["PnL"] < 0][["Date", "Ticker", "PnL"]], use_container_width=True)

    # ==============================================================================
    # 6. SYSTEM MAINTENANCE: SECURE MANAGEMENT UTILITIES
    # ==============================================================================
    st.write("---")
    with st.expander("🛠️ Advanced System Utilities"):
        
        # --- SUBSECTION A: SECURE BACKUP DATA EXTRACTION ---
        st.subheader("📥 Export Financial Records")
        st.write("Download an off-site local backup copy of your transaction logs before performing any system updates.")
        
        if not df_ledger.empty:
            # Convert the active dataframe memory matrix into a clean web string
            csv_download_buffer = df_ledger.to_csv(index=False).encode('utf-8')
            
            # Native interactive mobile download button container
            st.download_button(
                label="💾 Download Ledger Backup (.CSV)",
                data=csv_download_buffer,
                file_name=f"ledger_backup_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No transaction data exists to generate a backup file.")
            
        st.write("---")
        
        # --- SUBSECTION B: LOCKED DESTRUCTIVE ERASE UTILITY ---
        st.subheader("🔥 Hard System Reset")
        st.warning("Deletes all ledger transaction sheets, equity growth graphs, and balance tracking streams.")
        
        # Requirement 1: Admin Master Key Validation Box
        admin_key_input = st.text_input(
            "Enter System Admin Master Key to unlock:", 
            type="password", 
            placeholder="Input developer override key..."
        )
        
        # Pull the true secure secret string safely from your Streamlit Dashboard memory cache
        if admin_key_input == st.secrets["ADMIN_KEY"]:
            st.info("🔓 Admin Credentials Verified. Reset parameters unlocked.")
            
            # Requirement 2: Explicit Touch Safety Checkbox
            confirm_wipe = st.checkbox("I verify that I want to completely delete my entire transaction history and reset my balance back to its original value.")
            
            # Execute action block only if password is matching AND checkbox is checked
            if st.button("🔥 Confirm Complete Data Erasure", disabled=not confirm_wipe, use_container_width=True):
                try:
                    for file_path in [LEDGER_FILE, BALANCE_FILE, EQUITY_HISTORY_FILE]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    st.success("Database wiped successfully. Reloading baseline core portfolio settings...")
                    st.rerun()
                except Exception as e:
                    st.error("System Error: Failed to cleanly decouple and erase infrastructure file logs.")
        elif admin_key_input:
            st.error("❌ Access Denied: Incorrect Admin Master Key.")

elif app_mode == "Jacob's Account Ledger":
    # Render Account Ledger interface
    import streamlit as st
    import pandas as pd
    import os
    import plotly.graph_objects as gr
    from datetime import datetime

    # Mobile layout configuration
    st.set_page_config(page_title="AlgoLedger Pro1", layout="centered")

    # 1. INITIALIZE DATA STORAGE FILES
    LEDGER_FILE = "trading_ledger1.csv"
    BALANCE_FILE = "capital_balance1.txt"
    EQUITY_HISTORY_FILE = "equity_history1.csv"
    INITIAL_STARTING_CASH = 100.00

    DIVIDEND_DATABASE = {
        "MSFT": 0.91, "AVGO": 0.65, "NVDA": 0.25, "AAPL": 0.25, "META": 0.50, "GOOGL": 0.20, "MU": 0.15
    }

    def load_data():
        if os.path.exists(LEDGER_FILE):
            df = pd.read_csv(LEDGER_FILE)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df
        return pd.DataFrame(columns=["ID", "Date", "Ticker", "Type", "Price", "Capital", "Shares", "PnL", "Status"])

    def load_cash_balance():
        if os.path.exists(BALANCE_FILE):
            with open(BALANCE_FILE, "r") as f:
                return float(f.read().strip())
        return INITIAL_STARTING_CASH

    def load_equity_history():
        if os.path.exists(EQUITY_HISTORY_FILE):
            df = pd.read_csv(EQUITY_HISTORY_FILE)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            return df
        # Initialize with Day 0 baseline milestone
        return pd.DataFrame([{"Date": datetime.now().date(), "Total_Net_Worth": INITIAL_STARTING_CASH}])

    def save_all(df, cash_balance, equity_history_df):
        df.to_csv(LEDGER_FILE, index=False)
        with open(BALANCE_FILE, "w") as f:
            f.write(f"{cash_balance:.2f}")
        equity_history_df.to_csv(EQUITY_HISTORY_FILE, index=False)

    # Load global variables into active operational cache memory
    df_ledger = load_data()
    current_cash = load_cash_balance()
    df_equity = load_equity_history()

    # ==============================================================================
    # 2. CALCULATE PORTFOLIO FINANCIAL BALANCES
    # ==============================================================================
    st.title("📓 Jacob's Private Trading Ledger")


    # ==============================================================================
    # TAX QUARTER DEADLINE MONITOR (IRS Form 1040-ES Framework)
    # ==============================================================================
    current_date = datetime.now().date()
    current_year = current_date.year

    # Official IRS Deadline Arrays
    deadlines = [
        {"Quarter": "Q1", "Date": datetime(current_year, 4, 15).date(), "Voucher": "Voucher 1"},
        {"Quarter": "Q2", "Date": datetime(current_year, 6, 15).date(), "Voucher": "Voucher 2"},
        {"Quarter": "Q3", "Date": datetime(current_year, 9, 15).date(), "Voucher": "Voucher 3"},
        {"Quarter": "Q4", "Date": datetime(current_year + 1, 1, 15).date(), "Voucher": "Voucher 4"}
    ]

    # Identify the upcoming deadline milestone row
    upcoming_deadline = None
    for d in deadlines:
        if current_date <= d["Date"]:
            upcoming_deadline = d
            break

    if upcoming_deadline:
        days_remaining = (upcoming_deadline["Date"] - current_date).days
        deadline_str = upcoming_deadline["Date"].strftime("%B %d, %Y")
        
        # Render dynamic, touchscreen-friendly calendar warning callouts
        if days_remaining <= 14:
            st.error(f"🚨 **🚨 DANGER: IRS TAX DEADLINE CRITICAL** — Your **{upcoming_deadline['Quarter']} ({upcoming_deadline['Voucher']})** estimated tax voucher payment is due in exactly **{days_remaining} days** ({deadline_str}). Move your calculated tax reserve cache out of your broker account immediately.")
        else:
            st.info(f"📅 **Tax Calendar Reminder**: The next estimated IRS payment deadline is for **{upcoming_deadline['Quarter']}** in **{days_remaining} days** ({deadline_str}).")


    open_positions = df_ledger[df_ledger["Status"] == "OPEN"]
    total_invested = open_positions["Capital"].sum()
    closed_positions = df_ledger[(df_ledger["Status"] == "CLOSED") & (df_ledger["Type"] == "SELL")]
    net_realized_pnl = closed_positions["PnL"].sum()
    dividend_rows = df_ledger[df_ledger["Type"] == "DIVIDEND"]
    total_dividends_collected = dividend_rows["Capital"].sum()
    total_portfolio_value = current_cash + total_invested

    # Sync performance changes back to the performance chart logging system
    today_date = datetime.now().date()
    if df_equity.empty or df_equity.iloc[-1]["Date"] != today_date:
        new_snapshot = pd.DataFrame([{"Date": today_date, "Total_Net_Worth": total_portfolio_value}])
        df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)
    else:
        # Update today's existing row value to keep the curve pinpoint accurate
        df_equity.at[df_equity.index[-1], "Total_Net_Worth"] = total_portfolio_value
    df_equity.to_csv(EQUITY_HISTORY_FILE, index=False)

    # Mobile-responsive financial scorecards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚪ Uninvested Cash (Core Balance)", f"${current_cash:,.2f}")
        st.metric("🔵 Active Capital Deployed", f"${total_invested:,.2f}")
    with col2:
        st.metric("📈 Net Realized Profit/Loss", f"${net_realized_pnl:,.2f}", delta=f"{net_realized_pnl:+.2f}")
        st.metric("💼 Total Portfolio Net Worth", f"${total_portfolio_value:,.2f}")
    with col3:
        st.metric("💵 Total Dividends Collected", f"${total_dividends_collected:,.2f}", delta=f"+${total_dividends_collected:,.2f}" if total_dividends_collected > 0 else None)
    with col4:
        st.metric("Total Estimated Taxes", f"${net_realized_pnl * 0.16:,.2f}" if net_realized_pnl * 0.16 > 0 else None, help="Using 16% short-term capital gains tax estimate.")
    # ==============================================================================
    # 3. VISUAL PORTFOLIO PERFORMANCE LINE GRAPH CHART
    # ==============================================================================
    st.write("---")
    st.subheader("📈 Capital Growth Performance History")
    if len(df_equity) > 1:
        fig_equity = gr.Figure()
        fig_equity.add_trace(gr.Scatter(
            x=df_equity["Date"], 
            y=df_equity["Total_Net_Worth"], 
            mode="lines+markers", 
            name="Net Worth",
            line=dict(color="limegreen", width=3),
            marker=dict(size=6)
        ))
        fig_equity.update_layout(
            height=280, 
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickprefix="$")
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Performance trend graphics will populate here as trade logs generate data milestones over time.")

    st.write("---")

    # ==============================================================================
    # 4. ORDER ENTRY & LIQUIDATION AND CAPITAL BANK MODIFICATION ENGINE
    # ==============================================================================
    menu_selection = st.selectbox(
        "Choose Ledger Action Interface:", 
        ["🟢 Log New Buy Order", "🔴 Log Sell / Exit Order", "🏦 Account Capital Adjustments"]
    )

    if menu_selection == "🟢 Log New Buy Order":
        st.subheader("Execute Entry Position")
        with st.form("buy_form", clear_on_submit=True):
            b_date = st.date_input("Transaction Date", datetime.now().date())
            b_ticker = st.text_input("Ticker Symbol:").strip().upper()
            b_price = st.number_input("Actual Stock Buy Price ($):", min_value=0.01, step=0.01)
            b_capital = st.number_input("Total Money Invested ($):", min_value=1.00, step=100.00)
            submit_buy = st.form_submit_button("Submit Position Entry")
            
            if submit_buy:
                if not b_ticker:
                    st.error("Please enter a valid ticker symbol.")
                elif b_capital > current_cash:
                    st.error("Insufficient cash balance available in your uninvested core account.")
                else:
                    calculated_shares = b_capital / b_price
                    new_cash = current_cash - b_capital
                    new_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": b_date, "Ticker": b_ticker,
                        "Type": "BUY", "Price": b_price, "Capital": b_capital,
                        "Shares": calculated_shares, "PnL": 0.0, "Status": "OPEN"
                    }])
                    df_ledger = pd.concat([df_ledger, new_row], ignore_index=True)
                    save_all(df_ledger, new_cash, df_equity)
                    st.success(f"Successfully logged: {calculated_shares:.2f} shares of {b_ticker} acquired.")
                    st.rerun()

    elif menu_selection == "🔴 Log Sell / Exit Order":
        st.subheader("Execute Exit Position")
        open_options = df_ledger[df_ledger["Status"] == "OPEN"]
        
        if open_options.empty:
            st.info("You currently hold zero active positions in your open portfolio array.")
        else:
            position_list = open_options.apply(lambda r: f"{r['Ticker']} | Bought on {r['Date']} (${r['Capital']:.2f})", axis=1).tolist()
            selected_pos_str = st.selectbox("Select Position to Liquidate:", position_list)
            selected_idx = open_options.index[position_list.index(selected_pos_str)]
            target_row = df_ledger.loc[selected_idx]
            
            with st.form("sell_form", clear_on_submit=True):
                s_date = st.date_input("Liquidation Date", datetime.now().date())
                s_price = st.number_input("Actual Stock Sell Price ($):", min_value=0.01, step=0.01)

                captured_dividend = st.checkbox("Did you hold this stock past its official Ex-dividend date during this trade?")

                submit_sell = st.form_submit_button("Submit Position Liquidation")
                
                if submit_sell:
                    initial_capital = target_row["Capital"]
                    shares_held = target_row["Shares"]
                    ticker_symbol = target_row["Ticker"]
                    final_liquidation_value = shares_held * s_price
                    trade_pnl = final_liquidation_value - initial_capital
                    dividend_payout = 0.0
                    if ticker_symbol in DIVIDEND_DATABASE and captured_dividend:
                        dividend_payout = shares_held * DIVIDEND_DATABASE[ticker_symbol]


                    new_cash = current_cash + final_liquidation_value + dividend_payout
                    
                    df_ledger.at[selected_idx, "PnL"] = trade_pnl
                    df_ledger.at[selected_idx, "Status"] = "CLOSED"
                    
                    exit_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": s_date, "Ticker": ticker_symbol,
                        "Type": "SELL", "Price": s_price, "Capital": final_liquidation_value,
                        "Shares": shares_held, "PnL": trade_pnl, "Status": "CLOSED"
                    }])
                    df_ledger = pd.concat([df_ledger, exit_row], ignore_index=True)
                    
                    if dividend_payout > 0:
                        dividend_row = pd.DataFrame([{
                            "ID": int(datetime.now().timestamp()) + 1, "Date": s_date, "Ticker": "DIVIDEND",
                            "Type": "DIVIDEND", "Price": DIVIDEND_DATABASE[ticker_symbol], "Capital": dividend_payout,
                            "Shares": shares_held, "PnL": dividend_payout, "Status": "CLOSED"
                        }])
                        df_ledger = pd.concat([df_ledger, dividend_row], ignore_index=True)

                    # Append updated net worth path straight to performance plot array
                    new_snapshot = pd.DataFrame([{"Date": s_date, "Total_Net_Worth": new_cash + (total_invested - initial_capital)}])
                    df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)
                    
                    save_all(df_ledger, new_cash, df_equity)
                    st.success(f"Trade Closed. Performance Matrix Logged: {trade_pnl:+.2f}")
                    st.rerun()

    elif menu_selection == "🏦 Account Capital Adjustments":
        st.subheader("Manual Cash Capital Adjustments")
        st.write("Use this portal to record manual brokerage deposit transfers or external bank account cash withdrawals.")
        
        with st.form("adjustment_form", clear_on_submit=True):
            adj_type = st.radio("Adjustment Vector Type:", ["DEPOSIT (Add External Cash)", "WITHDRAWAL (Extract Bank Cash)"], horizontal=True)
            adj_amount = st.number_input("Transfer Capital Amount ($):", min_value=1.00, step=100.00)
            adj_date = st.date_input("Execution Date", datetime.now().date())
            submit_adj = st.form_submit_button("Finalize Balance Adjustment")
            
            if submit_adj:
                if "WITHDRAWAL" in adj_type and adj_amount > current_cash:
                    st.error("Execution failed: Requested transfer size exceeds current uninvested cash bounds.")
                else:
                    adjustment_mod = adj_amount if "DEPOSIT" in adj_type else -adj_amount
                    new_cash = current_cash + adjustment_mod
                    new_portfolio_value = total_portfolio_value + adjustment_mod
                    
                    # Append adjustments directly into transaction matrices logs
                    adj_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": adj_date, "Ticker": "CASH_ADJ",
                        "Type": "DEPOSIT" if "DEPOSIT" in adj_type else "WITHDRAW", "Price": 1.0, 
                        "Capital": adj_amount, "Shares": 0.0, "PnL": 0.0, "Status": "SYSTEM"
                    }])
                    df_ledger = pd.concat([df_ledger, adj_row], ignore_index=True)
                    # Sync capital shifts into performance graph data matrices tracks
                    new_snapshot = pd.DataFrame([{"Date": adj_date, "Total_Net_Worth": new_portfolio_value}])
                    df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)
                    save_all(df_ledger, new_cash, df_equity)
                    st.success(f"Balance adjustments updated successfully. New uninvested pool: ${new_cash:,.2f}")
                    st.rerun()
    # ==============================================================================
    # 5. VIEW HISTORICAL TRANSACTION BOOK
    # ==============================================================================
    st.write("---")
    st.subheader("🗃️ Historical Transaction Logbook")
    if df_ledger.empty:st.write("No transaction data logged yet.")
    else:
        display_df = df_ledger.copy().sort_values(by="Date", ascending=False)
        st.dataframe(display_df[["Date", "Ticker", "Type", "Price", "Capital", "PnL", "Status"]], use_container_width=True)

    # ==============================================================================
    # 5b. TAX-LOSS HARVESTING & DEDUCTION ANALYTICS
    # ==============================================================================
    st.write("---")
    st.subheader("🏛️ Tax-Loss Harvesting & Write-off Tracker")

    # 1. Isolate verified closed losses (Excluding cash adjustments and dividends)
    closed_stock_trades = df_ledger[(df_ledger["Status"] == "CLOSED") & (df_ledger["Ticker"] != "CASH_ADJ") & (df_ledger["Ticker"] != "DIVIDEND") & (df_ledger["Type"] == "BUY")]
    gross_gains = closed_stock_trades[closed_stock_trades["PnL"] > 0]["PnL"].sum()
    total_harvested_losses = abs(closed_stock_trades[closed_stock_trades["PnL"] < 0]["PnL"].sum())

    # 2. Calculate the IRS writing-off limits
    # The IRS allows you to offset 100% of capital gains, plus up to $3,000 of ordinary income
    if gross_gains > total_harvested_losses:
        taxable_net_gains = max(0.0, gross_gains - total_harvested_losses)
        ordinary_income_offset = 0.0
    else:
        taxable_net_gains = 0.0
        ordinary_income_offset = min(3000.0, total_harvested_losses - gross_gains)

    estimated_tax_savings = total_harvested_losses * 0.16

    # 3. Render mobile-optimized tax accounting grid layout
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.metric("📉 Total Harvested Capital Losses", f"${total_harvested_losses:,.2f}", delta="- Tax Deduction", delta_color="inverse")
        st.metric("🛡️ Ordinary Income Offset Value", f"${ordinary_income_offset:,.2f}", help="$3,000 max that can be written off standard W-2 income.")
    with t_col2:
        st.metric("⚖️ Net Taxable Short-Term Gains", f"${taxable_net_gains:,.2f}")
        st.metric("💵 Estimated Cash Tax Savings", f"${estimated_tax_savings:,.2f}", delta="+ Saved Core Cash")

    # Display a clean, expandable warning table pinpointing exactly which trades generated your tax write-offs
    if not closed_stock_trades.empty:
        with st.expander("📋 Review Active Loss Deductions (Form 1099-B Audit)"):
            st.dataframe(closed_stock_trades[closed_stock_trades["PnL"] < 0][["Date", "Ticker", "PnL"]], use_container_width=True)


    # ==============================================================================
    # 6. SYSTEM MAINTENANCE: SECURE MANAGEMENT UTILITIES
    # ==============================================================================
    st.write("---")
    with st.expander("🛠️ Advanced System Utilities"):
        
        # --- SUBSECTION A: SECURE BACKUP DATA EXTRACTION ---
        st.subheader("📥 Export Financial Records")
        st.write("Download an off-site local backup copy of your transaction logs before performing any system updates.")
        
        if not df_ledger.empty:
            # Convert the active dataframe memory matrix into a clean web string
            csv_download_buffer = df_ledger.to_csv(index=False).encode('utf-8')
            
            # Native interactive mobile download button container
            st.download_button(
                label="💾 Download Ledger Backup (.CSV)",
                data=csv_download_buffer,
                file_name=f"ledger_backup_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No transaction data exists to generate a backup file.")
            
        st.write("---")
        
        # --- SUBSECTION B: LOCKED DESTRUCTIVE ERASE UTILITY ---
        st.subheader("🔥 Hard System Reset")
        st.warning("Deletes all ledger transaction sheets, equity growth graphs, and balance tracking streams.")
        
        # Requirement 1: Admin Master Key Validation Box
        admin_key_input = st.text_input(
            "Enter System Admin Master Key to unlock:", 
            type="password", 
            placeholder="Input developer override key..."
        )
        
        # Pull the true secure secret string safely from your Streamlit Dashboard memory cache
        if admin_key_input == st.secrets["ADMIN_KEY1"]:
            st.info("🔓 Admin Credentials Verified. Reset parameters unlocked.")
            
            # Requirement 2: Explicit Touch Safety Checkbox
            confirm_wipe = st.checkbox("I verify that I want to completely delete my entire transaction history and reset my balance back to its original value.")
            
            # Execute action block only if password is matching AND checkbox is checked
            if st.button("🔥 Confirm Complete Data Erasure", disabled=not confirm_wipe, use_container_width=True):
                try:
                    for file_path in [LEDGER_FILE, BALANCE_FILE, EQUITY_HISTORY_FILE]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    st.success("Database wiped successfully. Reloading baseline core portfolio settings...")
                    st.rerun()
                except Exception as e:
                    st.error("System Error: Failed to cleanly decouple and erase infrastructure file logs.")
        elif admin_key_input:
            st.error("❌ Access Denied: Incorrect Admin Master Key.")