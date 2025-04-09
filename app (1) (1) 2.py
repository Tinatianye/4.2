import streamlit as st
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
import warnings
from statsmodels.tsa.api import VAR
from datetime import datetime  
from dateutil.relativedelta import relativedelta  

warnings.filterwarnings("ignore")

st.set_page_config(    
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

@st.cache_data
def get_dd():
    return pd.read_csv('hrc_price_CN_JP.csv')

st.markdown(f'''
    <h1 style="font-size: 25px; color: white; text-align: center; background: #0E539A; border-radius: .5rem; margin-bottom: 1rem;">
    HRC Price Predict Model Dashboard
    </h1>''', unsafe_allow_html=True)

# === Layout setup ===
left_col, right_col = st.columns([1, 3])

with left_col:
    st.subheader("Model Parameters")
    sea_freight = st.number_input("Sea Freight", value=10)
    exchange_rate = st.number_input("Exchange Rate (Rs/USD)", value=0.1)
    upside_pct = st.number_input("Upside (%)", value=10)
    downside_pct = st.number_input("Downside (%)", value=10)
    maxlags = st.number_input("Maxlags", value=12, min_value=1, max_value=100)

    df = get_dd()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    final_df = df.dropna()
    final_df.set_index('Date', inplace=True)

    months = st.number_input(
        f"Months ahead (Started in {final_df.index[-1].strftime('%Y-%m-%d')})",
        value=17, min_value=1, max_value=50)
    country = st.multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

with right_col:
    # Modeling
    final_df_differenced = final_df.diff().dropna()
    model = VAR(final_df_differenced)
    model_fitted = model.fit(maxlags)
    lag_order = model_fitted.k_ar
    forecast_input = final_df_differenced.values[-lag_order:]

    fc = model_fitted.forecast(y=forecast_input, steps=months)
    fc_period = pd.date_range(start=final_df.index[-1] + relativedelta(months=1), periods=months, freq='MS')
    df_forecast = pd.DataFrame(fc, index=fc_period, columns=final_df.columns)

    def invert_transformation(df_train, df_forecast):
        df_fc = pd.DataFrame(index=df_forecast.index)
        for i, col in enumerate(df_train.columns):
            last_value = df_train[col].iloc[-1]
            df_fc[f"{col}_forecast"] = last_value + df_forecast.iloc[:, i].cumsum()
        return df_fc

    df_forecast_processed = invert_transformation(final_df, df_forecast)
    df_res = df_forecast_processed[[col for col in df_forecast_processed.columns if col.endswith('_forecast')]].copy()
    df_res["Date"] = df_res.index

    def apply_upside_downside(df, column, up_pct, down_pct):
        df[f'{column}_upside'] = df[column] * (1 + up_pct / 100)
        df[f'{column}_downside'] = df[column] * (1 - down_pct / 100)
        return df

    for country_name in ["China", "Japan"]:
        colname = f"{country_name} HRC (FOB, $/t)_forecast"
        if colname in df_res.columns:
            df_res = apply_upside_downside(df_res, colname, upside_pct, downside_pct)

    history = final_df[[f"{c} HRC (FOB, $/t)" for c in country]].copy()
    history = history[history.index >= pd.to_datetime("2022-03-01")]
    history = history.reset_index()
    history["type"] = "Historical"

    forecast_melted = pd.melt(df_res, id_vars=["Date"], value_vars=[f"{c} HRC (FOB, $/t)_forecast" for c in country], var_name="series", value_name="value")
    forecast_melted["type"] = "Forecast"
    forecast_melted["series"] = forecast_melted["series"].str.replace("_forecast", "")

    history_melted = pd.melt(history, id_vars=["Date"], var_name="series", value_name="value")
    history_melted["series"] = history_melted["series"].str.replace(" HRC \(FOB, \$/t\)", "")
    full_data = pd.concat([history_melted.assign(type="Historical"), forecast_melted])

    fig = go.Figure()
    for s in full_data["series"].unique():
        for t in ["Historical", "Forecast"]:
            subset = full_data[(full_data["series"] == s) & (full_data["type"] == t)]
            fig.add_trace(go.Scatter(
                x=subset["Date"],
                y=subset["value"],
                name=f"{s} {t}",
                mode="lines",
                line=dict(dash="solid" if t == "Forecast" else "dot")
            ))

    for c in country:
        forecast_col = f"{c} HRC (FOB, $/t)_forecast"
        upper = f"{forecast_col}_upside"
        lower = f"{forecast_col}_downside"
        if upper in df_res.columns and lower in df_res.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df_res["Date"], df_res["Date"][::-1]]),
                y=pd.concat([df_res[upper], df_res[lower][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)' if c == "China" else 'rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"{c} Upside/Downside"
            ))



    fig.update_layout(
        title=dict(
            text="Forecasting HRC Prices<br><sub>with Historical Data + Upside/Downside</sub>",
            x=0.5,
            xanchor='center',
            y=0.9,
            yanchor='top',
            font=dict(size=22, color='#222', family='Arial Black')
        ),
        xaxis_title="Date",
        height=500,
        legend=dict(orientation="h", x=0.5, xanchor="center"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Show tables below chart ---
    last_row = df_res.iloc[-1]
    china_hrc = last_row.get("China HRC (FOB, $/t)_forecast", np.nan)
    japan_hrc = last_row.get("Japan HRC (FOB, $/t)_forecast", np.nan)

    china_columns = [
        'HRC FOB China', 'Sea Freight', 'Basic Customs Duty (%)',
        'LC charges & Port Charges', 'Exchange Rate (Rs/USD)', 'Freight from port to city'
    ]
    china_data = [china_hrc, sea_freight, 7.5, 10, exchange_rate, 500]
    china_df = pd.DataFrame(china_data, index=china_columns).reset_index()
    china_df.columns = ['Factors', 'Value']

    japan_columns = [
        'HRC FOB Japan', 'Sea Freight', 'Basic Customs Duty (%)',
        'LC charges & Port Charges', 'Exchange Rate (Rs/USD)', 'Freight from port to city'
    ]
    japan_data = [japan_hrc, sea_freight, 0, 10, exchange_rate, 500]
    japan_df = pd.DataFrame(japan_data, index=japan_columns).reset_index()
    japan_df.columns = ['Factors', 'Value']

    china_land_price = exchange_rate * (10 + 1.01 * (china_hrc + sea_freight) * (1 + 1.1 * 7.5)) + 500
    japan_land_price = exchange_rate * (10 + 1.01 * (japan_hrc + sea_freight) * (1 + 1.1 * 0)) + 500

    if 'China' in country and not np.isnan(china_hrc):
        st.markdown("### China Price Breakdown")
        st.dataframe(china_df, hide_index=True)
        st.markdown(f'<span style="color:#0E549B;font-weight:bold;">China landed price is: {round(china_land_price)} Rs/t</span>', unsafe_allow_html=True)

    if 'Japan' in country and not np.isnan(japan_hrc):
        st.markdown("### Japan Price Breakdown")
        st.dataframe(japan_df, hide_index=True)
        st.markdown(f'<span style="color:#C93B3B;font-weight:bold;">Japan landed price is: {round(japan_land_price)} Rs/t</span>', unsafe_allow_html=True)

    # --- Add download button ---
    with st.expander("⬇️ Download Forecast Data"):
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name='hrc_forecast_data.csv',
            mime='text/csv'
        )
