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
INITIAL_STARTING_CASH = 50000.00

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
st.title("📓 Private Trading Ledger Pro")

open_positions = df_ledger[df_ledger["Status"] == "OPEN"]
total_invested = open_positions["Capital"].sum()
closed_positions = df_ledger[df_ledger["Status"] == "CLOSED"]
net_realized_pnl = closed_positions["PnL"].sum()
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
col1, col2 = st.columns(2)
with col1:
    st.metric("⚪ Uninvested Cash (Core Balance)", f"${current_cash:,.2f}")
    st.metric("🔵 Active Capital Deployed", f"${total_invested:,.2f}")
with col2:
    st.metric("📈 Net Realized Profit/Loss", f"${net_realized_pnl:,.2f}", delta=f"{net_realized_pnl:+.2f}")
    st.metric("💼 Total Portfolio Net Worth", f"${total_portfolio_value:,.2f}")

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
        submit_buy = st.form_submit_with_button("Submit Position Entry")
        
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
            submit_sell = st.form_submit_with_button("Submit Position Liquidation")
            
            if submit_sell:
                initial_capital = target_row["Capital"]
                shares_held = target_row["Shares"]
                final_liquidation_value = shares_held * s_price
                trade_pnl = final_liquidation_value - initial_capital
                new_cash = current_cash + final_liquidation_value
                
                df_ledger.at[selected_idx, "PnL"] = trade_pnl
                df_ledger.at[selected_idx, "Status"] = "CLOSED"
                
                exit_row = pd.DataFrame([{
                    "ID": int(datetime.now().timestamp()), "Date": s_date, "Ticker": target_row["Ticker"],
                    "Type": "SELL", "Price": s_price, "Capital": final_liquidation_value,
                    "Shares": shares_held, "PnL": trade_pnl, "Status": "CLOSED"
                }])
                df_ledger = pd.concat([df_ledger, exit_row], ignore_index=True)
                
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
        submit_adj = st.form_submit_with_button("Finalize Balance Adjustment")
        
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
df_ledger = pd.concat([df_ledger, adj_row], ignore_index=True)# Sync capital shifts into performance graph data matrices tracksnew_snapshot = pd.DataFrame([{"Date": adj_date, "Total_Net_Worth": new_portfolio_value}])df_equity = pd.concat([df_equity, new_snapshot], ignore_index=True)save_all(df_ledger, new_cash, df_equity)st.success(f"Balance adjustments updated successfully. New uninvested pool: ${new_cash:,.2f}")st.rerun()==============================================================================5. VIEW HISTORICAL TRANSACTION BOOK==============================================================================st.write("---")st.subheader("🗃️ Historical Transaction Logbook")if df_ledger.empty:st.write("No transaction data logged yet.")else:display_df = df_ledger.copy().sort_values(by="Date", ascending=False)st.dataframe(display_df[["Date", "Ticker", "Type", "Price", "Capital", "PnL", "Status"]], use_container_width=True)