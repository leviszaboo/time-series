import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

for lag_order, best_model in significant_lags:
    print(f"Best model with {lag_order} lag(s):")
    print(best_model.summary())


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