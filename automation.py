import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as gr
from plotly.subplots import make_subplots

# Imports for creating the PDF document structure
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# 1. INITIALIZE DATA MATRIX & CONFIGURATIONS
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AMD", "MU", "TSLA", "META", "AVGO"]
summary_rows = []
generated_images = []

# Fetch extended history to clear 200-day rolling calculations cushion
raw_data = yf.download(tickers, start="2024-01-01", group_by="ticker")

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
        
        optimal_w = optimize_window_parameter(data)
        # Core Algorithmic Indicators block
        data["Support"] = data["Close"].rolling(window=optimal_w).min()
        data["Resistance"] = data["Close"].rolling(window=optimal_w).max()
        data["Long_Trend"] = data["Close"].rolling(window=200).mean()
        
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        data["RSI"] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
        
        # Align time matrices windows for presentation
        data = data.loc["2025-11-01":].copy()
        
        # Initialize simulation registers for state evaluation
        data["Position"] = 0
        data["Stop_Line"] = np.nan
        current_position = 0
        fixed_stop_floor = np.nan
        last_loss_date = None # Tracks date of last loss
        
        for i in range(1, len(data)):
            c_date = data.index[i]
            c_price = data["Close"].iloc[i]
            p_price = data["Close"].iloc[i - 1]
            c_rsi = data["RSI"].iloc[i]
            c_sup = data["Support"].iloc[i]
            c_res = data["Resistance"].iloc[i]
            c_trend = data["Long_Trend"].iloc[i]
            
            if last_loss_date is not None:
                days_since_loss =(c_date - last_loss_date).days
            else:
                days_since_loss = 999 # Infinite Buffer if no prior loss

            is_wash_sale_risk = days_since_loss <= 30

            is_trending = (c_price > c_trend)
            turned_up = (c_price > p_price)
            is_oversold = (c_rsi < 30) or (c_price <= c_sup * 1.01)
            
            buy_trigger = is_oversold and is_trending and not is_wash_sale_risk # and turned_up 
            sell_trigger = (c_rsi > 70) or (c_price >= c_res * 0.99)
            
            if current_position == 0:
                if buy_trigger:
                    current_position = 1
                    fixed_stop_floor = c_sup * 0.98
                    entry_price = c_price
            else:
                data.iloc[i, data.columns.get_loc("Stop_Line")] = fixed_stop_floor
                if c_price <= fixed_stop_floor or sell_trigger:
                    current_position = 0
                    if c_price < entry_price:
                        last_loss_date = c_date
                    
            data.iloc[i, data.columns.get_loc("Position")] = current_position

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        data["State_Shift"] = data["Position"].diff()
        buy_signals = data[data["State_Shift"] == 1]
        sell_signals = data[(data["State_Shift"] == -1) | ((data["Position"].shift(1) == 1) & (data["Position"] == 0))]

        if latest["Position"] == 1 and prev["Position"] == 0:
            recommendation = '<font color="limegreen"><b>BUY SIGNAL</b></font>'
        elif latest["Position"] == 0 and prev["Position"] == 1:
            recommendation = '<font color="crimson"><b>SELL TRIGGER</b></font>'
        elif latest["Position"] == 0 and is_wash_sale_risk:
            recommendation = '<font color="orange"><b>WASH SALE RISK</b></font>'
        elif latest["Position"] == 1:
            recommendation = '<font color="darkblue"><b>HOLD LONG</b></font>'
        else:
            recommendation = '<font color="gray"><b>CASH STATUS</b></font>'

        summary_rows.append(f"<b>{ticker}</b>: {recommendation} (Price: ${latest['Close']:.2f} | RSI: {latest['RSI']:.1f})")
        
        # 2. GENERATE HEADLESS STATIC IMAGES OF CHARTS
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
        fig.add_trace(gr.Scatter(x=data.index, y=data["Close"], name="Price", line=dict(color="black")), row=1, col=1)
        fig.add_trace(gr.Scatter(x=data.index, y=data["Long_Trend"], name="200d SMA", line=dict(color="blue", width=1.5)), row=1, col=1)
        fig.add_trace(gr.Scatter(x=data.index, y=data["Stop_Line"], name="Stop Floor", line=dict(color="orange", width=2, dash="dot")), row=1, col=1)
        fig.add_trace(gr.Scatter(x=data.index, y=data["Support"], name="Support", line=dict(color="green", dash="dash"), opacity=0.15), row=1, col=1)
        fig.add_trace(gr.Scatter(x=data.index, y=data["Resistance"], name="Resistance", line=dict(color="red", dash="dash"), opacity=0.15), row=1, col=1)

        if not buy_signals.empty:
            fig.add_trace(gr.Scatter(x=buy_signals.index, y=buy_signals["Close"], mode="markers", name="BUY", marker=dict(color="limegreen", size=10)), row=1, col=1)
        if not sell_signals.empty:
            fig.add_trace(gr.Scatter(x=sell_signals.index, y=sell_signals["Close"], mode="markers", name="SELL", marker=dict(color="crimson", size=10)), row=1, col=1)
        
        fig.add_trace(gr.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

        fig.update_layout(height=350, width=600, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        
        # Export chart image file locally inside the server instance sandbox
        img_filename = f"{ticker}_chart.png"
        fig.write_image(img_filename, engine="kaleido")
        generated_images.append((ticker, img_filename))
        
    except Exception as e:
        print(f"Error processing pipeline for {ticker}: {e}")

# ==============================================================================
# 3. CONSTRUCT THE MULTI-PAGE PDF DOCUMENT
# ==============================================================================
pdf_filename = "Daily_Stock_Report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=10, bottomMargin=10)
styles = getSampleStyleSheet()

# Create clean corporate layout typographic sheets
title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=22, spaceAfter=15, textColor='#1A1A1A')
body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=8)

story = []
today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
story.append(Paragraph(f"📊 Quantitative Portfolio Analysis Report", title_style))
story.append(Paragraph(f"<b>Execution Date</b>: {today_str} | System State: Optimized Sandbox Matrix Framework", body_style))
story.append(Spacer(1, 15))

story.append(Paragraph("<b>📋 Daily Execution Watchlist Scorecard:</b>", styles['Heading2']))
for row in summary_rows:
    story.append(Paragraph(row, body_style))
story.append(Spacer(1, 20))

# Sequentially weave the exported chart images down the document layout
story.append(Paragraph("<b>📈 Technical Structure Matrix Charts:</b>", styles['Heading2']))
for ticker, img_path in generated_images:
    story.append(Paragraph(f"<b>Asset Framework Analysis: {ticker}</b>", styles['Heading3']))
    story.append(Image(img_path, width=500, height=290))
    story.append(Spacer(1, 15))

doc.build(story)

# ==============================================================================
# 4. PACK ATTACHMENT & DISPATCH EMAIL REPORT
# ==============================================================================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = RECEIVER_EMAIL
msg['Subject'] = f"📁 Stock Scanner PDF Report — {today_str}"

msg.attach(MIMEText(f"Greetings Nicholas,\n\nThe automated background data-science engine has completed your dual-daily market scan for your tracked asset portfolio array.\n\nPlease find your comprehensive technical document and high-resolution chart diagrams attached inside the compiled PDF file below.\n\nBest regards,\nStock Scanner Terminal Bot", 'plain'))

# Prepare binary stream attachment mapping for the PDF document
with open(pdf_filename, "rb") as attachment:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={pdf_filename}")
    msg.attach(part)

try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    server.quit()
    print("Full visual PDF briefing attachment transmitted successfully.")
except Exception as e:
    print(f"SMTP Attachment Pipeline Error: {e}")
    raise e

# Clean up temporary server files
for _, img_path in generated_images:
    if os.path.exists(img_path): os.remove(img_path)
if os.path.exists(pdf_filename): os.remove(pdf_filename)