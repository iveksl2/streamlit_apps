import pandas as pd
import yfinance as yf
import streamlit as st

# Read the content of the file directly into a pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt", header=None, names=["Symbol"])

def get_financial_metrics(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info

    metrics = {
        "Revenue (TTM) ": info.get("totalRevenue"),
        "Price to Sales (TTM) ": info.get("priceToSalesTrailing12Months"),
        "Price to Earnings": info.get("trailingPE"),
        "Earnings Quarterly Growth": info.get("earningsQuarterlyGrowth"),
        "YoY Sales": info.get("revenueGrowth"),
        "Volume": info.get("volume"),
        "marketCap": info.get("marketCap"),
        "enterpriseToRevenue": info.get("enterpriseToRevenue"),
        "industry": info.get("industry"),
        "quickRatio": info.get("quickRatio"),
        "ebitda": info.get("ebitda"),
        "debtToEquity": info.get("debtToEquity"),
        "revenuePerShare": info.get("revenuePerShare"),
        "returnOnAssets": info.get("returnOnAssets"),
        "returnOnEquity": info.get("returnOnEquity"),
        "freeCashflow": info.get("freeCashflow"),
        "operatingCashflow": info.get("operatingCashflow"),
        "earningsGrowth": info.get("earningsGrowth"),
        "grossMargins": info.get("grossMargins"),
        "ebitdaMargins": info.get("ebitdaMargins"),
        "operatingMargins": info.get("operatingMargins")  
    }

    return metrics

# Get unique symbols from the DataFrame
symbols = df["Symbol"].unique()

# Create a Streamlit dropdown widget for selecting symbols
selected_symbol = st.selectbox("Select a symbol:", symbols)

# st.button('Get Financial Metrics'): -> Can do it this way 
if selected_symbol: 
    financial_metrics = get_financial_metrics(selected_symbol)
    st.write(f"Financial Metrics for {selected_symbol}:")
    st.write(financial_metrics)
