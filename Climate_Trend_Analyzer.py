import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import os

# -------------------------------
# Load dataset
# -------------------------------
DATA_PATH = 'data/GlobalLandTemperaturesByCity.csv'
df = pd.read_csv(DATA_PATH)

print("Raw Data:")
print(df.head())

# -------------------------------
# Data Cleaning
# -------------------------------
df['dt'] = pd.to_datetime(df['dt'], errors='coerce')

df = df.dropna(subset=['AverageTemperature'])

df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month

df = df[df['Year'] > 1900]

print("\nCleaned Data:")
print(df.head())
os.makedirs("outputs/plots", exist_ok=True)
yearly_temp = df.groupby('Year')['AverageTemperature'].mean().reset_index()

plt.figure(figsize=(10,5))
plt.plot(yearly_temp['Year'], yearly_temp['AverageTemperature'])
plt.title("Year vs Temperature Trend")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.grid()

plt.savefig("outputs/plots/temp_trend.png")
plt.show()

pivot = df.pivot_table(index='Month', columns='Year', values='AverageTemperature')

plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap='coolwarm')

plt.title("Monthly Temperature Heatmap")

plt.savefig("outputs/plots/heatmap.png")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(yearly_temp['Year'], yearly_temp['AverageTemperature'])

plt.title("Scatter: Year vs Temperature")
plt.xlabel("Year")
plt.ylabel("Temperature")

plt.savefig("outputs/plots/scatter_year_temp.png")
plt.show()

yearly_temp['RollingAvg'] = yearly_temp['AverageTemperature'].rolling(5).mean()

plt.figure(figsize=(10,5))
plt.plot(yearly_temp['Year'], yearly_temp['AverageTemperature'], alpha=0.5)
plt.plot(yearly_temp['Year'], yearly_temp['RollingAvg'])

plt.title("Smoothed Temperature Trend")

plt.savefig("outputs/plots/rolling_avg.png")
plt.show()

mean = yearly_temp['AverageTemperature'].mean()
std = yearly_temp['AverageTemperature'].std()

anomalies = yearly_temp[
    (yearly_temp['AverageTemperature'] > mean + 2*std) |
    (yearly_temp['AverageTemperature'] < mean - 2*std)
]

plt.figure(figsize=(10,5))
plt.plot(yearly_temp['Year'], yearly_temp['AverageTemperature'])
plt.scatter(anomalies['Year'], anomalies['AverageTemperature'], color='red')

plt.title("Anomaly Detection")

plt.savefig("outputs/plots/anomaly_plot.png")
plt.show()