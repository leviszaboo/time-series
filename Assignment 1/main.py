import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

data = pd.read_csv("Assignment 1/data_assign_p1.csv")

quarters = data['obs']
gdp_growth = data['GDP_QGR']

## Exercise 1

# Plot GDP growth

years = quarters.str[:4]

plt.figure(figsize=(12, 6))
plt.plot(quarters, gdp_growth, marker='o', linestyle='-')
plt.title('Dutch GDP Quarterly Growth Rates')
plt.xlabel('Quarter')
plt.ylabel('GDP Growth Rate')
plt.grid(True)

plt.xticks(quarters[::4], years[::4], rotation=45)

plt.show()

# ACF and PACF

periods = 12  

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot ACF
plot_acf(gdp_growth, lags=periods, ax=ax1, marker='o', color='b')
ax1.set_title('Sample ACF')
ax1.set_xlabel('Lags')
ax1.set_ylabel('ACF')
ax1.grid(True)

# Plot PACF
plot_pacf(gdp_growth, lags=periods, ax=ax2, marker='o', color='b')
ax2.set_title('Sample PACF')
ax2.set_xlabel('Lags')
ax2.set_ylabel('PACF')
ax2.grid(True)

plt.tight_layout()
plt.show()


## Exercise 2

p = 4
significance_level = 0.05

significant_lags = []

for lag in range(p, 0, -1):
    model = AutoReg(gdp_growth, lags=lag, old_names=False)
    results = model.fit()
    
    if all(results.pvalues[1:] < significance_level):
        significant_lags.append((lag, results))

for lag, model in significant_lags:
    print(f"Best model with {lag} lag(s):")
    print(model.summary())


## Exercise 3
# Note: since we only have one significant model this for loop is not necessary,
# but it's better for reusability.

acf_plots = {}
resid_lags = 19

for lag, model in significant_lags:
    residuals = model.resid

    acf_plot = plot_acf(residuals, lags=resid_lags, title=f'ACF of Residuals (AR({lag}))')
    acf_plot.axes[0].set_xlabel('Lags')

    acf_plots[lag] = acf_plot

for lag, acf_plot in acf_plots.items():
    plt.show()

# Exercise 4

forecast_horizon = 8

forecast_quarters = ["2009Q2", "2009Q3", "2009Q4", "2010Q1", "2010Q2", "2010Q3", "2010Q4", "2011Q1"]
forecasts = []

for lag, model in significant_lags:
    values = model.predict(start=len(gdp_growth), end=len(gdp_growth) + forecast_horizon - 1, dynamic=False)
    values = values.to_frame(name="Prediction")
    values['Quarter'] = forecast_quarters
    values = values.set_index("Quarter")
    forecasts.append((f"Model with {lag} lag(s)", values))

print("Quarterly GDP Growth Rate Forecast:")
for label, values in forecasts:
  print(f"{label}: \n")
  print(values)

# Exercise 5

def prediction_conf_int(X: float, h: int, alpha: float, var: float, phi: float) -> tuple[float, float]:
    var_hth_resid = var * np.sum([phi ** (2 * i - 1) for i in range(1, h + 1)])
    std_err = np.sqrt(var_hth_resid)
    z = stats.norm.ppf(1 - (alpha / 2))
    return (X - z * std_err, X + z * std_err)

conf_intervals = []

for index, (lag, model) in enumerate(significant_lags):
    phi = model.params[1]
    var = np.var(model.resid)
    model_intervals = []
    forecasted = forecasts[index][1].reset_index()
    X_forecasted = forecasted['Prediction']
    for i in range(1, forecast_horizon + 1):
      X = X_forecasted[i - 1]
      conf_interval = prediction_conf_int(X, i, alpha, var, phi)
      model_intervals.append([i, conf_interval])
    conf_intervals.append((lag, model_intervals))

print("Summary of 95% Confidence Intervals:")
for lag, model_intervals in conf_intervals:
    print(f"Model with {lag} lag(s):")
    for i, conf_interval in model_intervals:
        print(f"  h = {i}: ({conf_interval[0]:.4f}, {conf_interval[1]:.4f})")