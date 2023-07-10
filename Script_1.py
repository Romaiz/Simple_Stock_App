import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from email.message import EmailMessage  # ! for further functionality

st.write("""
         # Netsol Stock Price
         
         A simple web app to review and predict the stock prices for Netsol.
         """)

tickersymbol = 'NTWK'
tickerdata = yf.Ticker(tickersymbol)
tickerdf = tickerdata.history(period='1y', start='2022-8-10', end='2023-8-10')
# print(tickerdf.head())
tickerdf = tickerdf.dropna()
tickerdf = tickerdf.drop(columns=['Dividends', 'Stock Splits'])
# print(tickerdf.columns)
# print(tickerdf.head())

tickerdf['Price_Change'] = tickerdf['Close'].pct_change()
tickerdf['Price_Change'].fillna(0, inplace=True)
st.write(""" ## Closing Price """)
st.line_chart(tickerdf['Close'])

st.write(""" ## Volume Price """)
st.line_chart(tickerdf['Volume'])

st.write(""" ## Change in price throughout the year """)
st.line_chart(tickerdf['Price_Change'])

scaler = MinMaxScaler()
highest_value = tickerdf['High'].max()
lowest_value = tickerdf['Low'].min()
st.markdown(f"**Highest Value:** {highest_value}")
st.markdown(f"**Lowest Value:** {lowest_value}")
# print(tickerdf.head())
normalized_columns = ['Open', 'Close', 'High', 'Low', 'Volume', 'Price_Change']
tickerdf[normalized_columns] = scaler.fit_transform(
    tickerdf[normalized_columns])
# print(tickerdf.head())
X = tickerdf.drop(columns=['Close', 'Volume'])
y_volume = tickerdf['Volume']
y_close = tickerdf['Close']
X_train, X_test, y_volume_train, y_volume_test, y_close_train, y_close_test = train_test_split(
    X, y_volume, y_close, test_size=0.2, random_state=42)


volume_model = RandomForestRegressor(n_estimators=100, random_state=42)
volume_model.fit(X_train, y_volume_train)
volume_predicted = volume_model.predict(X_test)
volume_mse = mean_squared_error(y_volume_test, volume_predicted)
volume_rmse = np.sqrt(volume_mse)
volume_mae = mean_absolute_error(y_volume_test, volume_predicted)
# print("Volume Model - Mean Squared Error:", volume_mse)
# print("Volume Model - Root Mean Squared Error:", volume_rmse)
# print("Volume Model - Mean Absolute Error:", volume_mae)

close_model = RandomForestRegressor(n_estimators=100, random_state=42)
close_model.fit(X_train, y_close_train)
close_predicted = close_model.predict(X_test)
close_mse = mean_squared_error(y_close_test, close_predicted)
close_rmse = np.sqrt(close_mse)
close_mae = mean_absolute_error(y_close_test, close_predicted)
# print("Close Price Model - Mean Squared Error:", close_mse)
# print("Close Price Model - Root Mean Squared Error:", close_rmse)
# print("Close Price Model - Mean Absolute Error:", close_mae)

future_predictions_volume = volume_model.predict(X)
future_predictions_close = close_model.predict(X)
tickerdf['Predicted_Volume'] = future_predictions_volume
tickerdf['Predicted_Close'] = future_predictions_close

st.write(""" ## Predicted Volume (10th July 2023 - 10th July 2024)""")
st.line_chart(tickerdf['Predicted_Volume'])

st.write(""" ## Predicted Close Price (10th July 2023 - 10th July 2024)""")
st.line_chart(tickerdf['Predicted_Close'])
