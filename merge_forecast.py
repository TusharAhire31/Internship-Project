import pandas as pd

df = pd.read_csv("data/stock_data_clean.csv", parse_dates=["Date"])
df["Close"] = df["Close"].replace(r"[\$,]", "", regex=True).astype(float)
df["Open"] = df["Open"].replace(r"[\$,]", "", regex=True).astype(float)
df["High"] = df["High"].replace(r"[\$,]", "", regex=True).astype(float)
df["Low"] = df["Low"].replace(r"[\$,]", "", regex=True).astype(float)
df = df.rename(columns={"Close": "Actual"})

try:
    arima = pd.read_csv("outputs/arima_forecast.csv", parse_dates=["Date"])
    arima = arima.rename(columns={"Forecast": "ARIMA"})
except Exception as e:
    print(" ARIMA file issue:", e)
    arima = pd.DataFrame(columns=["Date", "ARIMA"])

try:
    prophet = pd.read_csv("outputs/prophet_forecast.csv", parse_dates=["Date"])
    prophet = prophet.rename(columns={"Forecast": "Prophet"})
except Exception as e:
    print(" Prophet file issue:", e)
    prophet = pd.DataFrame(columns=["Date", "Prophet"])

try:
    lstm = pd.read_csv("outputs/lstm_forecast.csv", parse_dates=["Date"])
    lstm = lstm.rename(columns={"Forecast": "LSTM"})
except Exception as e:
    print(" LSTM file issue:", e)
    lstm = pd.DataFrame(columns=["Date", "LSTM"])

final = df.copy()
final["Date"] = pd.to_datetime(final["Date"], format='mixed', errors='coerce')

if not arima.empty:
    arima["Date"] = pd.to_datetime(arima["Date"])
    final = final.merge(arima, on="Date", how="outer")

if not prophet.empty:
    prophet["Date"] = pd.to_datetime(prophet["Date"])
    final = final.merge(prophet, on="Date", how="outer")

if not lstm.empty:
    lstm["Date"] = pd.to_datetime(lstm["Date"])
    final = final.merge(lstm, on="Date", how="outer")

for col in ["ARIMA", "Prophet", "LSTM"]:
    if col in final.columns:
        final[col].fillna(final[col].mean(), inplace=True)

final.fillna(method="ffill", inplace=True)

columns = ["Date", "Open", "High", "Low", "Actual", "Volume"]
columns += [col for col in ["ARIMA", "Prophet", "LSTM"] if col in final.columns]
final = final[columns]

final.to_csv("outputs/final_forecast_tableau_dashboard.csv", index=False)
print("Merged forecast file saved: outputs/final_forecast_tableau_dashboard.csv")
