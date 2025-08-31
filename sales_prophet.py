# sales_prophet.py
# Prepare CSV with columns ds (date) and y (sales)
# Run: python sales_prophet.py data/sales.csv

import sys, pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def main():
    fn = sys.argv[1] if len(sys.argv)>1 else "data/sales.csv"
    df = pd.read_csv(fn, parse_dates=['ds'])
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    m.plot(forecast)
    plt.show()
    forecast.to_csv("output/sales_forecast.csv", index=False)
    print("Saved output/sales_forecast.csv")

if __name__=="__main__":
    main()
