import pandas as pd
from utils import load_data

df = load_data("data/stock_data_clean.csv")
arima = pd.read_csv("outputs/arima_forecast.csv")

merged = pd.merge(df, arima, on="Date", how="outer")
merged.to_csv("outputs/final_forecast_tableau.csv", index=False)

print(" Forecast saved for Tableau as 'final_forecast_tableau.csv'")
