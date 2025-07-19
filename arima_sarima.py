import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/stock_data_clean.csv", parse_dates=['Date'])
df['Close'] = df['Close'].replace(r'[\$,]', '', regex=True).astype(float)
df = df[['Date', 'Close']].dropna()
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index ,format = 'mixed')

arima_model = ARIMA(df['Close'], order=(5, 1, 0))
arima_result = arima_model.fit()

arima_forecast = arima_result.forecast(steps=30)
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)

arima_df = pd.DataFrame({'Date': forecast_dates, 'Close': arima_forecast})
arima_df.to_csv("../outputs/arima_forcast.csv", index=False)
plt.plot(df['Close'], label='Actual')
plt.plot(pd.date_range(df.index[-1], periods=30), arima_forecast, label='ARIMA Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.tight_layout()
plt.show()
