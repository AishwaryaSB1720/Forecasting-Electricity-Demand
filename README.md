# Forecasting-Electricity-Demand
One of leading electricity Distribution Company would like to understand demand for electricity for the next 1-2 years to manage the production of electricity and managing the vendors for the same. It is one of the important exercises to getting accurate estimation of demand so that they can procure or produce the electricity as per the demand. 
Available Data: 
The data of monthly electricity consumption available starting from January 1973 to December 2019. We need to forecast the demand for next two years. 
1.	Date – Month & Year 
2.	Electricity Consumption – Electricity consumption in Trillion Watts
Business Objective: 
a.	Need to forecast the demand of electricity for next 1-2 years. 
b.	Calculate error metrics (RMSE, RMSPE, MAPE) 

# import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the File 
df = pd.read_csv('Electricity Consumption.csv')
df.head(5)
df.tail(5)
df.shape()

# Check the Date time format and create a new Dataframe for updated one
df.types()
dt_df = pd.date_range(start = '1973-01-01', end='2019-01-09')
dt_df = pd.DataFrame(dt_df,columns=['date'])
dt_df.head()

# Now set the date format by using String format time function
df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
df.head(3)

# Check the null values
print(df.isnull().sum())

# Split data:As the data has 561 records, Split the data in 80:20 ratio into Train:Test
train = df.iloc[:-113]
test = df.iloc[-113:]
from prophet import Prophet

# Initialize and fit the model
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=len(test), freq='MS')
forecast = model.predict(future)

# Extract predicted values for test period
predicted = forecast.set_index('ds').loc[test['ds'].values]
actual = test.set_index('ds')['y']

# Calculate the error using MAE,RMSE,MAPE Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(actual, predicted['yhat']))
mae = mean_absolute_error(actual, predicted['yhat'])
mape = np.mean(np.abs((actual - predicted['yhat']) / actual)) * 100
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# Plot the Graph to showcase the Electricity Demand Estimation for next 2 year
plt.figure(figsize=(10,5))
plt.plot(actual.index, actual.values, label='Actual')
plt.plot(predicted.index, predicted['yhat'].values, label='Predicted')
plt.title("Prophet Forecast – 80:20 Split (114 Months Test)")
plt.xlabel("Date")
plt.ylabel("Electricity Consumption (TW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


