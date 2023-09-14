import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

data = pd.read_csv('data_assign_p1.csv')

quarters = data['obs']
gdp_growth = data['GDP_QGR']

# Exercise 1

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

