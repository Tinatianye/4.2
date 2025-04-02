import streamlit as st

# Import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic

import plotly.express as px 

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime  
from dateutil.relativedelta import relativedelta  

st.set_page_config(    
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

col1, col2 = st.columns([1, 3])
col_controls = col2.columns(3)

# Load default file and process
df = pd.read_csv('wo_na.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")
final_df = df.dropna()
final_df.set_index('Date', inplace=True)

# Controls
maxlags = col_controls[0].number_input("**Maxlags**", value=12, min_value=1, max_value=100)
months = col_controls[1].number_input(f"**Months ahead (Started in {final_df.index.tolist()[-1].strftime('%Y-%m-%d')})**", value=17, min_value=1, max_value=50)
country = col_controls[2].multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

# VAR Model
final_df_differenced = final_df.diff().dropna()
model = VAR(final_df_differenced)
model_fitted = model.fit(maxlags)
lag_order = model_fitted.k_ar
forecast_input = final_df_differenced.values[-lag_order:]

fc = model_fitted.forecast(y=forecast_input, steps=months)
fc_period = pd.date_range(start=final_df.index.tolist()[-1]+relativedelta(months=1), 
                          end=final_df.index.tolist()[-1]+relativedelta(months=months), freq='MS')
df_forecast = pd.DataFrame(fc, index=fc_period, columns=final_df.columns + '_1d')

def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_forecast_processed = invert_transformation(final_df, df_forecast)

# Forecast Plot
def fun(x):
    d1 = final_df[[f'{x} HRC (FOB, $/t)']].loc["2024-03-01":"2025-12-31"].copy()
    d1.columns = ["HRC (FOB, $/t)"]
    d1["t"] = f"{x} HRC (FOB, $/t)"
    d1 = d1.reset_index()

    d2 = df_forecast_processed[[f"{x} HRC (FOB, $/t)_forecast"]].copy()
    d2.columns = ["HRC (FOB, $/t)"]
    d2["t"] = f"{x} HRC (FOB, $/t)_forecast"
    d2 = d2.reset_index()
    if 'Date' in d2.columns:
        d2 = d2[d2['Date'] <= '2025-12-31']

    d = pd.concat([d1, d2])
    return d

d = [fun(i) for i in country]
d3 = pd.concat(d)

required_cols = {"Date", "HRC (FOB, $/t)", "t"}
if not required_cols.issubset(d3.columns):
    st.error("Error: One or more required columns missing in forecast dataframe.")
else:
    fig = px.line(d3, x="Date", y="HRC (FOB, $/t)", color="t",
                  markers=True, line_shape="linear",
                  color_discrete_sequence=['#0E549B', 'red', '#FFCE44', 'violet'])
    fig.update_traces(hovertemplate='%{y}')
    fig.update_layout(title = {'text': "/".join(country)+" Forecasting HRC prices", 'x': 0.5, 'y': 0.96, 'xanchor': 'center'},
                      margin = dict(t=30), height = 500,
                      legend=dict(title="", yanchor="top", y=0.99, xanchor="center", x=0.5, orientation="h"),
                      xaxis_title="Date",
                      yaxis_title="HRC (FOB, $/t)",
                      paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    col2.plotly_chart(fig, use_container_width=True, height=400)
