# Generated from: Energy_Demand_Prediction_Model.ipynb
# Converted at: 2026-03-06T02:09:01.806Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

#Importing the tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Downloading electricity grid data
url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

df = pd.read_csv(url)

df.head()

#Converting the time column

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df.set_index('utc_timestamp', inplace=True)

#Selecting Germany's electricity load

germany = df[['DE_load_actual_entsoe_transparency']].copy()
germany = germany.dropna() #Remove missing values

germany.head()

#Limiting data to 2015 - 2019

germany = germany['2015-01-01':'2019-12-31']
germany.head()

len(germany)

#Converting hourly data to daily data

daily_load = germany.resample('D').sum()
daily_load.columns = ['Daily_MWh']

daily_load.head()

#Visualizing electricity demand

plt.figure(figsize=(15,5))
plt.plot(daily_load)
plt.title("Germany Daily Electricity Demand (2015–2019)")
plt.ylabel("MWh")
plt.show()

import requests
import pandas as pd

#Collecting weather data

latitude = 52.52
longitude = 13.405

start_date = "20150101"
end_date = "20191231"

url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M&community=RE&longitude={longitude}&latitude={latitude}&start={start_date}&end={end_date}&format=JSON"

response = requests.get(url)
data = response.json()

temps = data['properties']['parameter']['T2M']

weather = pd.DataFrame(list(temps.items()), columns=['Date', 'Avg_Temp'])

weather['Date'] = pd.to_datetime(weather['Date'])
weather.set_index('Date', inplace=True)

weather.head()

#Merging electricitty demand and weather

weather.index = weather.index.tz_localize('UTC')
merged = daily_load.merge(weather, left_index=True, right_index=True)

merged.head()

len(merged)

#Creating heating and cooling demand

merged['HDD'] = (18 - merged['Avg_Temp']).clip(lower=0)
merged['CDD'] = (merged['Avg_Temp'] - 18).clip(lower=0)

merged.head()

#Computing correlation

merged[['Daily_MWh','Avg_Temp','HDD','CDD']].corr()

#Creating memory features (Lag variables)

merged['Lag_1'] = merged['Daily_MWh'].shift(1)
merged['Lag_7'] = merged['Daily_MWh'].shift(7)

merged['Rolling_7'] = merged['Daily_MWh'].rolling(window=7).mean()

merged = merged.dropna()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = merged[['HDD','CDD','Lag_1','Lag_7','Rolling_7']]
y = merged['Daily_MWh']

# Time-aware split (no shuffling!)
train_size = int(len(merged) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

mae, rmse

merged['Daily_MWh'].mean()

avg_demand = merged['Daily_MWh'].mean()

mae_percentage = (mae / avg_demand) * 100

mae_percentage

#Calendar features

merged['DayOfWeek'] = merged.index.dayofweek
merged['Month'] = merged.index.month
merged['IsWeekend'] = (merged['DayOfWeek'] >= 5).astype(int)

#Building the model

X = merged[['HDD','CDD','Lag_1','Lag_7','Rolling_7',
            'DayOfWeek','Month','IsWeekend']]
y = merged['Daily_MWh']

train_size = int(len(merged) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

#Measuring error

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

avg_demand = merged['Daily_MWh'].mean()
mae_percentage = (mae / avg_demand) * 100

mae_percentage

#Upgrading the model - Random forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

rf_mae_percentage = (rf_mae / avg_demand) * 100

rf_mae_percentage

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, rf_predictions, label="Predicted")
plt.legend()
plt.title("Random Forest: Actual vs Predicted")
plt.show()

#Feature importance

import pandas as pd

feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feature_importance

import pandas as pd

# Create 2027 date range
future_dates = pd.date_range(start='2027-01-01', end='2027-12-31', freq='D')

future_df = pd.DataFrame(index=future_dates)
future_df['DayOfWeek'] = future_df.index.dayofweek
future_df['Month'] = future_df.index.month
future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)

# Calculate average temp per day-of-year from past data
merged['DayOfYear'] = merged.index.dayofyear
daily_temp_avg = merged.groupby('DayOfYear')['Avg_Temp'].mean()

# Map to future 2027 dates
future_df['DayOfYear'] = future_df.index.dayofyear
future_df['Avg_Temp'] = future_df['DayOfYear'].map(daily_temp_avg)

# Calculate HDD and CDD
future_df['HDD'] = (18 - future_df['Avg_Temp']).clip(lower=0)
future_df['CDD'] = (future_df['Avg_Temp'] - 18).clip(lower=0)

future_df.drop(columns='DayOfYear', inplace=True)

# Take last 7 days from 2026
last_days = merged['Daily_MWh'][-7:].values

# Create Lag_1 and Lag_7 for first week
lag_1 = [last_days[-1]]  # first day lag_1
lag_7 = [last_days[0]]   # first day lag_7

# Fill the rest sequentially after predictions
future_df['Lag_1'] = 0
future_df['Lag_7'] = 0
future_df['Rolling_7'] = 0

import numpy as np

# Make a copy to avoid modifying original
future_sim = future_df.copy()

# Get last 7 actual demand values from 2026
history = list(merged['Daily_MWh'][-7:].values)

predictions_2027 = []

for date in future_sim.index:

    # Lag features
    lag_1 = history[-1]
    lag_7 = history[-7]
    rolling_7 = np.mean(history[-7:])

    # Build feature row
    row = future_sim.loc[date].copy()
    row['Lag_1'] = lag_1
    row['Lag_7'] = lag_7
    row['Rolling_7'] = rolling_7

    # Ensure correct feature order
    row = row[X.columns]

    # Predict
    prediction = rf.predict([row])[0]

    # Store prediction
    predictions_2027.append(prediction)

    # Update history for next day
    history.append(prediction)

# Save results
future_sim['Forecast_MWh'] = predictions_2027

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.plot(future_sim.index, future_sim['Forecast_MWh'])
plt.title("Forecasted Daily Electricity Demand - 2027")
plt.ylabel("Daily MWh")
plt.show()

#Computing total electricity demand for the year

total_2027_mwh = future_sim['Forecast_MWh'].sum()
total_2027_mwh

#Converting the total electricity demand to TWh

total_2027_twh = total_2027_mwh / 1_000_000
total_2027_twh