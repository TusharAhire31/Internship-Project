import pandas as pd
from prophet import Prophet

df = pd.read_csv("data/stock_data_clean.csv")
df = df[['Date', 'Close']].dropna()
df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)

df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')  # remove invalid/missing dates
df = df.dropna(subset=['ds', 'y'])  

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

forecast[['ds', 'yhat']].tail(30).rename(
    columns={'ds': 'Date', 'yhat': 'Forecast'}
).to_csv("outputs/prophet_forcast.csv", index=False)

print(" Prophet forecast saved to outputs/prophet_forcast.csv")
