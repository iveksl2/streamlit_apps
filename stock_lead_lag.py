import streamlit as st


from pandas_datareader import data # Not working
import yfinance as yf #Using this as a tmp alternative
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from typing import *
import plotly.express as px

def generate_lags(df: pd.DataFrame, lag_column: str, num_lags:int=7):
  df = df.copy()
  for i in range(1, num_lags+1):
    df[f'{lag_column}_lag_{i}'] = df[lag_column].shift(i)
  return df

def gen_lag_heatmap(lag_df: pd.DataFrame, lead_sym: str):
  fig, ax = plt.subplots(figsize=(10,2))
  sns.heatmap(lag_df.corr().loc[[lead_sym]], annot=True)

def get_stock_data(sym_1: str, sym_2: str, start_date: str):
  sym_1_df = yf.download(sym_1,  start=start_date)[['Adj Close']]
  sym_2_df =  yf.download(sym_2, start=start_date)[['Adj Close']]

  dual_df = pd.merge(sym_1_df, 
                     sym_2_df,
                     left_index=True, 
                     right_index=True, 
                     suffixes=('_'+sym_1,'_'+sym_2))
  
  dual_df.columns = dual_df.columns.str.replace('Adj Close_', '')
  return dual_df

dual_df = get_stock_data("btc-usd","COIN", start_date='2021-01-01')

st.dataframe(dual_df)
plt.style.use('default')

level_of_aggregation = st.selectbox("Select your level of aggregation",
                             options=["Daily", "Weekly"])

if level_of_aggregation == 'Weekly':
  dual_df = dual_df.resample('w').mean()

st.write("Relative performance of both Assets")
yo = dual_df.div(dual_df.iloc[0])
st.line_chart(yo)


pct_returns = dual_df.pct_change(periods=1).mul(100).dropna()

fig = px.bar(pct_returns, barmode='group',title='Percent Returns')
st.plotly_chart(fig)

st.write("Granger Causality Test:")
gc_test = grangercausalitytests(pct_returns, 5)
st.write('P value for a Granger Causality test is', gc_test[1][0]['ssr_ftest'][1])


