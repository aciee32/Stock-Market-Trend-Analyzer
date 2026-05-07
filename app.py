import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Define start and end dates
start = '2014-01-01'
end = '2025-12-31'

st.title('Stock Market Trend Analyzer')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Downloading the data
df = yf.download(user_input, start=start, end=end)

# Describing Data - Updated header to reflect your actual date range
st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

# Visualisations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'green', label='Close Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'orange', label='Close Price')
plt.plot(ma100, 'r', label='100MA')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'purple', label='Close Price')
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.legend()
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

# Load model
model = load_model(r'C:\Users\ARPITA CHOUDHURY\PythonProject\mini project\stock trend prediction\my_model.keras')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Important: Fit the scaler on training data, then transform the final_df
scaler.fit(data_training)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions
y_predicted = model.predict(x_test)

# Inverse scaling factor
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)