import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ARDL, AutoReg

df = pd.read_csv("assignment_2/data_assign_p2.csv")

quarters = df['obs']
gdp_growth = df['GDP_QGR']
unemployment_rate = df['UN_RATE']

df['GDP_QGR_lag1'] = df['GDP_QGR'].shift(1)
df['GDP_QGR_lag2'] = df['GDP_QGR'].shift(2)
df['GDP_QGR_lag3'] = df['GDP_QGR'].shift(3)
df['GDP_QGR_lag4'] = df['GDP_QGR'].shift(4)
df['UN_RATE_lag1'] = df['UN_RATE'].shift(1)
df['UN_RATE_lag2'] = df['UN_RATE'].shift(2)
df['UN_RATE_lag3'] = df['UN_RATE'].shift(3)
df['UN_RATE_lag4'] = df['UN_RATE'].shift(4)
df.dropna(inplace=True)

alpha = 0.05

# Exercise 1: Plot GDP growth and Unemployment rates
years = quarters.str[:4]

plt.figure(figsize=(11, 6))

# Plot GDP growth
plt.subplot(2, 1, 1)
plt.plot(quarters, gdp_growth, marker='', linestyle='-')
plt.title('Dutch GDP Quarterly Growth Rates')
plt.xlabel('Year')
plt.ylabel('GDP Growth Rate')
plt.grid(True)
plt.xticks(quarters[::4], years[::4], rotation=45)

# Plot Unemployment rates
plt.subplot(2, 1, 2)
plt.plot(quarters, unemployment_rate, marker='', linestyle='-')
plt.title('Dutch Quarterly Unemployment Rates')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.grid(True)
plt.xticks(quarters[::4], years[::4], rotation=45)

plt.tight_layout()
plt.show()

# Exercise 2: Estimate an AR model for GDP growth with 4 lags

X_ar = df[['GDP_QGR_lag1', 'GDP_QGR_lag2', 'GDP_QGR_lag3', 'GDP_QGR_lag4']]
X_ar = sm.add_constant(X_ar)
y_ar = df['GDP_QGR']

ar_model = sm.OLS(y_ar, X_ar).fit()
print(ar_model.summary())

X_ar = df[['GDP_QGR_lag1', 'GDP_QGR_lag3']]
X_ar = sm.add_constant(X_ar)
y_ar = df['GDP_QGR']

ar_model = sm.OLS(y_ar, X_ar).fit()
print("Best AR model:")
print(ar_model.summary())

# Exercise 2: Estimate an ADL model for GDP growth and Unemployment rate with 4 lags
X_adl = df[['GDP_QGR_lag1', 'GDP_QGR_lag2', 'GDP_QGR_lag3', 'GDP_QGR_lag4', 'UN_RATE_lag1', 'UN_RATE_lag2', 'UN_RATE_lag3', 'UN_RATE_lag4']]
X_adl = sm.add_constant(X_adl)
y_adl = df['UN_RATE']

adl_model = sm.OLS(y_adl, X_adl).fit()
print(adl_model.summary())

X_adl = df[['GDP_QGR_lag1',  'UN_RATE_lag1', 'UN_RATE_lag3']]
X_adl = sm.add_constant(X_adl)
y_adl = df['UN_RATE']

adl_model = sm.OLS(y_adl, X_adl).fit()
print("Best ADL model:")
print(adl_model.summary())

# Exercise 3: 8 period forecast for unemployment rate

ar_coeffs = ar_model.params
adl_coeffs = adl_model.params

UN_RATE_values = [unemployment_rate.values[-3], unemployment_rate.values[-2], unemployment_rate.values[-1]]
GDP_QGR_values = [gdp_growth.values[-3], gdp_growth.values[-2], gdp_growth.values[-1]]

print(UN_RATE_values, GDP_QGR_values)

forecast_periods = 8

forecasted_UN_RATE = []
forecasted_GDP_QGR = []

for i in range(forecast_periods):
    next_UN_RATE = (
        adl_coeffs[0]
        + adl_coeffs[1] * GDP_QGR_values[-1]
        + adl_coeffs[2] * UN_RATE_values[-1]
        + adl_coeffs[3] * UN_RATE_values[-3]
    )
    
    next_GDP_QGR = (
        ar_coeffs[0]
        + ar_coeffs[1] * GDP_QGR_values[-1]
        + ar_coeffs[2] * GDP_QGR_values[-3]
    )
    
    UN_RATE_values.append(next_UN_RATE)
    GDP_QGR_values.append(next_GDP_QGR)
    
    forecasted_UN_RATE.append(next_UN_RATE)
    forecasted_GDP_QGR.append(next_GDP_QGR)

time_axis = range(1, forecast_periods + 1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, forecasted_UN_RATE, label='Forecasted Unemployment Rate')
plt.xlabel('Time Period')
plt.ylabel('Unemployment Rate')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_axis, forecasted_GDP_QGR, label='Forecasted GDP Growth Rate', color="orange")
plt.xlabel('Time Period')
plt.ylabel('GDP Growth Rate')
plt.legend()

plt.tight_layout()
plt.show()

print("8-period forecasted unemployment rate(%):")
for index, rate in enumerate(forecasted_UN_RATE): 
  print(f"Period {index + 1}: {rate:.4f}")




