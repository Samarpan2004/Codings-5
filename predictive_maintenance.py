# predictive_maintenance.py
# Usage: sensor CSV with timestamp + multiple sensor columns.
# Run: python predictive_maintenance.py data/sensors.csv

import sys, pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def main():
    fn = sys.argv[1] if len(sys.argv)>1 else "data/sensors.csv"
    df = pd.read_csv(fn, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    sensors = df.columns.drop('timestamp')
    X = df[sensors].fillna(method='ffill').fillna(0)
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = iso.fit_predict(X)
    # anomaly == -1 are outliers
    print("Anomalies detected:", (df['anomaly']==-1).sum())
    # quick plot for first sensor
    plt.plot(df['timestamp'], df[sensors[0]])
    plt.scatter(df.loc[df['anomaly']==-1,'timestamp'], df.loc[df['anomaly']==-1, sensors[0]], color='r')
    plt.title('Sensor 1 with anomalies')
    plt.show()
    df.to_csv("output/sensor_with_anomaly.csv", index=False)
    print("Saved output/sensor_with_anomaly.csv")

if __name__=="__main__":
    main()
