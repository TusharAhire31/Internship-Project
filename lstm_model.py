import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense

df = pd.read_csv("data/stock_data_clean.csv")
df['Date'] = pd.to_datetime(df['Date'], format='mixed',dayfirst=False)  

data = df[['Close']].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

predicted = model.predict(X[-30:])
predicted_prices = scaler.inverse_transform(predicted)
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)

pd.DataFrame({
    'Date': future_dates,
    'Forecast_LSTM': predicted_prices.flatten()
}).to_csv("outputs/lstm_forecast.csv", index=False)

plt.plot(df['Close'], label='Actual')
plt.plot(range(len(df), len(df) + 30), predicted_prices, label='LSTM Forecast')
plt.legend()
plt.show()
