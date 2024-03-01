import pickle
import xgboost as xgb


from ta import add_all_ta_features

# In[2]:


import pandas as pd 
import datetime
import numpy as np
#import matplotlib.pyplot as plt

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode

df = pd.read_excel("df.xlsx")
# In[3]:

option = st.selectbox(
    'Select A store',
    df["StoreNumber"].unique().tolist())

st.write('You selected:', option)

option2 = st.selectbox(
    'Select A Product',
    df["ProductDescription"].unique().tolist())

st.write('You selected:', option2)



import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

def plot_sales_with_logistic_forecast_and_bounded_ci_plotly(df, store_number, product_description):
    # Filter and prepare data
    specific_store_item = df[
        (df['StoreNumber'] == store_number) & 
        (df['ProductDescription'] == product_description)
    ]
    
    specific_store_item["SalesDate"] = pd.to_datetime(specific_store_item["SalesDate"])
    specific_store_item.set_index("SalesDate", inplace=True)
    weekly_sales = specific_store_item.resample('W').sum().reset_index()

    # Apply log transformation to 'Sales' + 1
    weekly_sales['y'] = np.log(weekly_sales['Sales'] + 1)
    weekly_sales.rename(columns={'SalesDate': 'ds'}, inplace=True)

    # Set the maximum capacity for the logistic growth model
    max_capacity = weekly_sales['y'].max() * 1.5
    weekly_sales['cap'] = max_capacity

    # Initialize Prophet model with logistic growth
    model = Prophet(growth='logistic', interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(weekly_sales)

    # Make future dataframe for predictions, extending 52 weeks into the future
    future = model.make_future_dataframe(periods=52, freq='W')
    future['cap'] = max_capacity

    forecast = model.predict(future)
    
    # Reverse log transformation for forecasted values
    forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.exp(forecast[['yhat', 'yhat_lower', 'yhat_upper']]) - 1
    
    # Bind the 'yhat_upper' to no more than 1.5 times the maximum historical sales value
    max_historical_sales = np.exp(weekly_sales['y']).max() - 1
    bound_upper_limit = 1.5 * max_historical_sales
    forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: min(x, bound_upper_limit))
    
    # Plotting with Plotly
    fig = go.Figure()

    # Actual Sales
    fig.add_trace(go.Scatter(x=weekly_sales['ds'], y=np.exp(weekly_sales['y']) - 1, mode='lines+markers', name='Actual Sales', line=dict(color='black')))
    
    # Forecast Sales
    last_actual_date = weekly_sales['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_actual_date]
    fig.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_upper'], fill=None, mode='lines', name='Upper CI', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_lower'], fill='tonexty', mode='lines', name='Lower CI', line=dict(color='lightblue')))

    # Layout adjustments
    fig.update_layout(title=f'Weekly Sales Forecast (Logistic Growth) for Store {store_number}, {product_description}',
                      xaxis_title='Date',
                      yaxis_title='Sales',
                      legend_title='Legend',
                      hovermode='x unified')

    fig.show()

# Example usage
plot_sales_with_logistic_forecast_and_bounded_ci_plotly(df, option, option2)

# #if selection:
# #st.write("You selected:")
# #st.json(selection["selected_rows"])

