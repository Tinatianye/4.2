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

@st.cache_data
def get_dd():
    fp = 'wo_na.csv'
    return pd.read_csv(fp)


st.set_page_config(    
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

st.markdown(f'''
    <h1 style="font-size: 25px; color: white; text-align: center; background: #0E539A; border-radius: .5rem; margin-bottom: 1rem;">
    HRC Price Predict Model Dashboard
    </h1>''', unsafe_allow_html=True)



col = st.columns([1.2, 3])
col1 = col[1].columns(3)
col0 = col[0].columns(2)
# uploaded_file = col[0].file_uploader("Choose a file", type=["csv"])

df = get_dd()

# ff = False
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     if str(df.columns.tolist())==str(dd.columns.tolist()):
#         ff = True
#     else:
#         # file_path = 'wo_na.csv'
#         df = dd
#         col[0].warning("Columns error, please keep the headers consistent!")
# else:
#     # file_path = 'wo_na.csv'
#     df = dd
#     col[0].warning("No file be upload!")

sea_freight = col0[0].number_input("**Sea Freight**",value=10)
exchange_rate = col0[1].number_input("**Exchange Rate (Rs/USD)**",value=0.1)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

final_df = df.copy()
final_df = df.dropna()

final_df.set_index('Date', inplace=True)

maxlags = col1[0].number_input("**Maxlags**", value=12, min_value=1, max_value=100)
months = col1[1].number_input(f"**Months ahead (Started in {final_df.index.tolist()[-1].strftime('%Y-%m-%d')})**", value=17, min_value=1, max_value=50)
country = col1[2].multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

final_df_differenced = final_df.diff().dropna()

model = VAR(final_df_differenced)
x = model.select_order(maxlags=maxlags)

model_fitted = model.fit(maxlags)

lag_order = model_fitted.k_ar
forecast_input = final_df_differenced.values[-lag_order:]


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

st.markdown('''
    <style>
    [data-testid="block-container"] {
        padding: 60px;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .stPlotlyChart {
        border: 1px solid gray;
        border-radius: 0.5rem;
    }
    [data-testid="stImage"] img {
        border-radius: 0.5rem;
    }
    </style>''', unsafe_allow_html=True)

fc = model_fitted.forecast(y=forecast_input, steps=months)
print('?????')
print(final_df.index.tolist()[-1])
fc_period = pd.date_range(
    start=final_df.index.tolist()[-1],
    end=final_df.index.tolist()[-1]+relativedelta(months=months-1), freq='MS')
df_forecast = pd.DataFrame(fc, index=fc_period, columns=final_df.columns + '_1d')

df_forecast_processed = invert_transformation(final_df, df_forecast)



df_res = df_forecast_processed[["China HRC (FOB, $/t)_forecast", "Japan HRC (FOB, $/t)_forecast"]]
df_res = df_res.reset_index()
last_row = df_res.iloc[-1]


month = last_row[0]
china_hrc= last_row[1]
japan_hrc =  last_row[2]

def fun(x):
    d1 = final_df[[f'{x} HRC (FOB, $/t)']]
    d1.columns = [f"HRC (FOB, $/t)"]
    d1["t"] = f"{x} HRC (FOB, $/t)"
    d2 = df_forecast_processed[[f"{x} HRC (FOB, $/t)_forecast"]]
    d2.columns = [f"HRC (FOB, $/t)"]
    d2["t"] = f"{x} HRC (FOB, $/t)_forecast"
    d = pd.concat([d1, d2])
    return d

d = []
for i in country:
    d.append(fun(i))

d3 = pd.concat(d)

d3 = d3[d3.index > '2023-01-01']

# 创建折线图  
fig = px.line(d3, x=d3.index, y="HRC (FOB, $/t)", color="t", 
                markers=False, color_discrete_sequence=['#0E549B', 'red', '#FFCE44', 'violet'])

# 自定义 tooltip 内容  
fig.update_traces(hovertemplate='%{y}')

fig.update_layout(
    title = {
        'text': "/".join(country)+" Forecasting HRC prices",
        'x': 0.5,  # 标题居中
        'y': 0.96,
        'xanchor': 'center'
    },
    margin = dict(t=30),
    height = 500,
    #width = "auto",
    legend=dict(
        title="",
        yanchor="top",
        y=0.99,
        xanchor="center",
        x=0.5,
        orientation="h"
    ),
    xaxis={
        "title":"Date"
    },
    paper_bgcolor='rgba(0, 0, 0, 0)',  # 图表外部背景透明  
    plot_bgcolor='rgba(0, 0, 0, 0)'    # 图表内部背景透明
)

col[1].plotly_chart(fig, use_container_width=True, height=400)

china_columns  =[ 'HRC FOB China',
                'Sea Freight',
                'Basic Customs Duty (%)',
                'LC charges & Port Charges',
                'Exchange Rate (Rs/USD)',
                'Freight from port to city'
               ]
china_data  =[china_hrc,
              sea_freight,
                7.5,
                10,
                exchange_rate,
               500
               ]
china_df = pd.DataFrame(china_data,index=china_columns)
china_df = china_df.reset_index()
china_df.columns=['Factors','Value']


japan_columns  =[ 'HRC FOB Japan',
                'Sea Freight',
                'Basic Customs Duty (%)',
                'LC charges & Port Charges',
                'Exchange Rate (Rs/USD)',
                'Freight from port to city'
               ]
japan_data  =[japan_hrc,
              sea_freight,
                0,
                10,
                exchange_rate,
               500
               ]
japan_df = pd.DataFrame(japan_data,index=china_columns)
japan_df = japan_df.reset_index()
japan_df.columns=['Factors','Value']

china_land_price = exchange_rate*(10+1.01*(china_hrc+sea_freight)*(1+1.1*7.5))+500
japan_land_price = exchange_rate*(10+1.01*(japan_hrc+sea_freight)*(1+1.1*0))+500
if 'China' in country:
    col[0].write("**China**")
    col[0].dataframe(china_df,hide_index=True)
    col[0].write('China land price is: '+str(round(china_land_price)))
if 'Japan' in country:
    col[0].write("**Japan**")
    col[0].dataframe(japan_df,hide_index=True)
    col[0].write('Japan land price is: '+str(round(japan_land_price)))
