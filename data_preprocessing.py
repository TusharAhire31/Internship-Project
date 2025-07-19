import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.abspath("data/stock_data.csv")
print("  file path :", file_path)

df = pd.read_csv(file_path, sep="\t", parse_dates=["Date"])
df.rename(columns={"Close/Last": "Close"}, inplace=True)
df["Close"] = df["Close"].str.replace("$", "", regex=False).astype(float)

df = df.sort_values("Date")
df.ffill()
df.to_csv("data/stock_data_clean.csv", index=False)
print("Data preprocessing done and  Cleaned file saved.")
