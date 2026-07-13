    import streamlit as st
    import pandas as pd
    import os
    from datetime import datetime

    # Mobile layout configuration
    st.set_page_config(page_title="AlgoLedger", layout="centered")

    # 1. INITIALIZE LEDGER DATA STORAGE (Local Excel or CSV Matrix file)
    LEDGER_FILE = "trading_ledger.csv"
    BALANCE_FILE = "capital_balance.txt"
    INITIAL_STARTING_CASH = 5000.00 # Change this to your actual starting account size

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

    def save_all(df, cash_balance):
        df.to_csv(LEDGER_FILE, index=False)
        with open(BALANCE_FILE, "w") as f:
            f.write(f"{cash_balance:.2f}")

    # Load active assets and balance matrices into active cache memory
    df_ledger = load_data()
    current_cash = load_cash_balance()

    # ==============================================================================
    # 2. RENDER TOP METRICS PANEL
    # ==============================================================================
    st.title("📓 Private Trading Ledger")

    # Calculate metrics summary
    open_positions = df_ledger[df_ledger["Status"] == "OPEN"]
    total_invested = open_positions["Capital"].sum()
    closed_positions = df_ledger[df_ledger["Status"] == "CLOSED"]
    net_realized_pnl = closed_positions["PnL"].sum()
    total_portfolio_value = current_cash + total_invested

    # Mobile-responsive financial scorecards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚪ Uninvested Cash (Core Balance)", f"${current_cash:,.2f}")
        st.metric("🔵 Active Capital Deployed", f"${total_invested:,.2f}")
    with col2:
        st.metric("📈 Net Realized Profit/Loss", f"${net_realized_pnl:,.2f}", delta=f"{net_realized_pnl:+.2f}")
        st.metric("💼 Total Portfolio Net Worth", f"${total_portfolio_value:,.2f}")

    st.write("---")

    # ==============================================================================
    # 3. INTERACTIVE ORDER LOGGING SYSTEM
    # ==============================================================================
    menu_selection = st.radio("Select Action:", ["🟢 Log New Buy Order", "🔴 Log Sell / Exit Order"], horizontal=True)

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
                    # Math processing for equity position
                    calculated_shares = b_capital / b_price
                    new_cash = current_cash - b_capital
                    
                    new_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": b_date, "Ticker": b_ticker,
                        "Type": "BUY", "Price": b_price, "Capital": b_capital,
                        "Shares": calculated_shares, "PnL": 0.0, "Status": "OPEN"
                    }])
                    
                    df_ledger = pd.concat([df_ledger, new_row], ignore_index=True)
                    save_all(df_ledger, new_cash)
                    st.success(f"Successfully logged long position: {calculated_shares:.2f} shares of {b_ticker} acquired.")
                    st.rerun()

    elif menu_selection == "🔴 Log Sell / Exit Order":
        st.subheader("Execute Exit Position")
        
        # Isolate active entries that can be liquidated
        open_options = df_ledger[df_ledger["Status"] == "OPEN"]
        
        if open_options.empty:
            st.info("You currently hold zero active positions in your open portfolio array.")
        else:
            # Create user selection list combining ticker and date for accuracy
            position_list = open_options.apply(lambda r: f"{r['Ticker']} | Bought on {r['Date']} (${r['Capital']:.2f})", axis=1).tolist()
            selected_pos_str = st.selectbox("Select Position to Liquidate:", position_list)
            
            # Extract selected row parameters
            selected_idx = open_options.index[position_list.index(selected_pos_str)]
            target_row = df_ledger.loc[selected_idx]
            
            with st.form("sell_form", clear_on_submit=True):
                s_date = st.date_input("Liquidation Date", datetime.now().date())
                s_price = st.number_input("Actual Stock Sell Price ($):", min_value=0.01, step=0.01)
                
                submit_sell = st.form_submit_button("Submit Position Liquidation")
                
                if submit_sell:
                    # Calculate capital returns and trade mathematics
                    initial_capital = target_row["Capital"]
                    shares_held = target_row["Shares"]
                    
                    final_liquidation_value = shares_held * s_price
                    trade_pnl = final_liquidation_value - initial_capital
                    new_cash = current_cash + final_liquidation_value
                    
                    # Update row parameters in the data spreadsheet matrix
                    df_ledger.at[selected_idx, "PnL"] = trade_pnl
                    df_ledger.at[selected_idx, "Status"] = "CLOSED"
                    
                    # Append corresponding cash release trail row
                    exit_row = pd.DataFrame([{
                        "ID": int(datetime.now().timestamp()), "Date": s_date, "Ticker": target_row["Ticker"],
                        "Type": "SELL", "Price": s_price, "Capital": final_liquidation_value,
                        "Shares": shares_held, "PnL": trade_pnl, "Status": "CLOSED"
                    }])
                    
                    df_ledger = pd.concat([df_ledger, exit_row], ignore_index=True)
                    save_all(df_ledger, new_cash)
                    
                    if trade_pnl >= 0:
                        st.success(f"Trade Closed. Profit Realized: **+${trade_pnl:.2f}**")
                    else:
                        st.error(f"Trade Closed. Loss Realized: **${trade_pnl:.2f}**")
                    st.rerun()

    # ==============================================================================
    # 4. VIEW HISTORICAL TRANSACTION BOOK
    # ==============================================================================
    st.write("---")
    st.subheader("🗃️ Historical Transaction Logbook")
    if df_ledger.empty:
        st.write("No transaction data logged yet.")
    else:
        # Clean display formatting mapping for tables
        display_df = df_ledger.copy()
        display_df = display_df.sort_values(by="Date", ascending=False)
        st.dataframe(display_df[["Date", "Ticker", "Type", "Price", "Capital", "PnL", "Status"]], use_container_width=True)