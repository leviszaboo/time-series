import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from scipy.stats import norm

np.random.seed(0)

# Exercise 1

sigma = 1.0
B = 10000
sample_sizes = [30, 100, 500, 1000, 2000, 5000, 10000]

beta_hat_dict = {}
r_squared_dict = {}
t_test_dict = {}
p_value_dict = {}

for sample_size in sample_sizes:
    beta_hat_dict[sample_size] = np.zeros(B)
    r_squared_dict[sample_size] = np.zeros(B)
    t_test_dict[sample_size] = np.zeros(B)
    p_value_dict[sample_size] = np.zeros(B)

    for i in range(B):
        Y = np.zeros(sample_size)
        X = np.zeros(sample_size)
        for t in range(1, sample_size):
            ut = np.random.normal(0, sigma)
            vt = np.random.normal(0, sigma)
            Y[t] = Y[t - 1] + ut
            X[t] = X[t - 1] + vt
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        beta_hat = model.params[1]
        se_beta_hat = model.bse[1]
        beta_hat_dict[sample_size][i] = beta_hat
        r_squared_dict[sample_size][i] = model.rsquared
        t_test_dict[sample_size][i] = abs(beta_hat) / se_beta_hat
        p_value = model.pvalues[1]
        p_value_dict[sample_size][i] = p_value

fig, axes = plt.subplots(4, 1, figsize=(10, 15))

for i, sample_size in enumerate(sample_sizes):
    sns.kdeplot(beta_hat_dict[sample_size], label=f'Sample Size {sample_size}', ax=axes[0])

for i, sample_size in enumerate(sample_sizes):
    axes[0].hist(beta_hat_dict[sample_size], alpha=0.1, density=True, bins=100)

axes[0].set_xlabel('$\\hat{\\beta}$')
axes[0].set_title('Distribution of $\\hat{\\beta}$ for Different Sample Sizes')
axes[0].legend()
axes[0].grid(True)

for i, sample_size in enumerate(sample_sizes):
    sns.kdeplot(r_squared_dict[sample_size], label=f'Sample Size {sample_size}', ax=axes[1])

for i, sample_size in enumerate(sample_sizes):
    axes[1].hist(r_squared_dict[sample_size], alpha=0.1, density=True, bins=100)

axes[1].set_xlabel('R-squared')
axes[1].set_title('Distribution of R-squared for Different Sample Sizes')
axes[1].set_xlim(0)
axes[1].legend()
axes[1].grid(True)

for i, sample_size in enumerate(sample_sizes):
    axes[2].hist(t_test_dict[sample_size], alpha=0.6, density=True, bins=100, label=f'Sample Size {sample_size}')

axes[2].set_title('Distribution of $| \\hat{\\beta} | / SE(\\hat{\\beta})$ for Different Sample Sizes')
axes[2].set_xlabel('Sample Size')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].set_xlim(0, 60)
axes[2].set_axisbelow(True)
axes[2].grid(True)

p_value_proportions = [np.mean(p_value_dict[sample_size] < 0.05) for sample_size in sample_sizes]
sns.barplot(x=sample_sizes, y=p_value_proportions, ax=axes[3], width=0.1)
axes[3].set_xlabel('Sample Size')
axes[3].set_title('Proportion of P - values < 0.05 for Different Sample Sizes')
axes[3].set_ylim(0, 1.1)
axes[3].axhline(y=1, color='grey', linestyle='--')
axes[3].set_axisbelow(True)
axes[3].grid(True)

plt.tight_layout()
plt.show()

# Exercise 2

data = pd.read_csv("assignment_3/data_assign_p3.csv")

apple = data['APPLE']
netflix = data['NETFLIX']

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(apple, label='Apple Stock Price')
plt.xlabel('t')
plt.title('Apple Stock Price Over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(netflix, label='Netflix Stock Price')
plt.xlabel('t')
plt.title('Netflix Stock Price Over Time')
plt.legend()

plt.tight_layout()
plt.show()


plot_acf(apple, lags=12, title='ACF for Apple Stock')

plot_pacf(apple, lags=12, title='PACF for Apple Stock')

plot_acf(netflix, lags=12, title='ACF for Netflix Stock')

plot_pacf(netflix, lags=12, title='PACF for Netflix Stock')


# Exercise 3

def perform_adf_test(series):
    best_order = None
    best_sic = np.inf
    
    for lag in range(1, 5):  
        result = adfuller(series, maxlag=lag)
        sic = result[5]  
        
        if sic < best_sic:
            best_sic = sic
            best_order = lag
    
    result = adfuller(series, maxlag=best_order)
    
    adf_statistic = result[0]
    critical_values = result[4]

    is_stationary = adf_statistic < critical_values['10%']
    
    return adf_statistic, is_stationary

results = []

for column in data.columns[1:]:
    series = data[column]
    adf_statistic, is_stationary = perform_adf_test(series)
    
    result = {
        "Stock": column,
        "ADF Statistic": adf_statistic,
        "Stationary at 90% Confidence Level": is_stationary
    }
    
    results.append(result)

results_df = pd.DataFrame(results)

print(results_df)

# Exercise 4

microsoft = data['MICROSOFT']

apple_diffs = apple.diff()
microsoft_diffs = microsoft.diff()

mean_diff_apple = apple_diffs.mean()
var_apple = apple_diffs.var()

mean_diff_microsoft = microsoft_diffs.mean()
var_microsoft = microsoft_diffs.var()

forecast_days = 5
alpha = 0.05
z = norm.ppf(1 - (alpha) / 2)

apple_forecast_df = pd.DataFrame(columns=['Forecasted Price', 'Lower Bound', 'Upper Bound'])
microsoft_forecast_df = pd.DataFrame(columns=['Forecasted Price', 'Lower Bound', 'Upper Bound'])

for day in range(forecast_days):
    last_apple_price = apple_forecast_df.iloc[-1]['Forecasted Price'] if day > 0 else apple.iloc[-1]
    random_innovation_apple = np.random.normal(mean_diff_apple, np.sqrt(var_apple))
    last_apple_price += random_innovation_apple
    std_error_apple = np.sqrt(var_apple * (day + 1))
    lower_bound_apple = last_apple_price - z * std_error_apple
    upper_bound_apple = last_apple_price + z * std_error_apple
    last_microsoft_price = microsoft_forecast_df.iloc[-1]['Forecasted Price'] if day > 0 else microsoft.iloc[-1]
    random_innovation_microsoft = np.random.normal(mean_diff_microsoft, np.sqrt(var_microsoft))
    last_microsoft_price += random_innovation_microsoft
    std_error_microsoft = np.sqrt(var_microsoft * (day + 1) )
    lower_bound_microsoft = last_microsoft_price - z * std_error_microsoft 
    upper_bound_microsoft = last_microsoft_price + z * std_error_microsoft

    apple_forecast_df.loc[day] = [last_apple_price, lower_bound_apple, upper_bound_apple]
    microsoft_forecast_df.loc[day] = [last_microsoft_price, lower_bound_microsoft, upper_bound_microsoft]

print("5-Day Stock Price Forecast for Apple:")
apple_forecast_df

print("\n5-Day Stock Price Forecast for Microsoft:")
microsoft_forecast_df

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(1, forecast_days + 1), apple_forecast_df['Forecasted Price'], label='Forecasted Apple Prices', marker='o', linestyle='--')
plt.fill_between(np.arange(1, forecast_days + 1), apple_forecast_df['Lower Bound'], apple_forecast_df['Upper Bound'], alpha=0.2, color='blue')
plt.title('Apple Stock Price Forecast')
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('Day')
plt.ylabel('Price')

plt.subplot(2, 1, 2)
plt.plot(np.arange(1, forecast_days + 1), microsoft_forecast_df['Forecasted Price'], label='Forecasted Microsoft Prices', marker='o', linestyle='--')
plt.fill_between(np.arange(1, forecast_days + 1), microsoft_forecast_df['Lower Bound'], microsoft_forecast_df['Upper Bound'], alpha=0.2, color='orange')
plt.title('Microsoft Stock Price Forecast')
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('Day')
plt.ylabel('Price')

plt.tight_layout()
plt.show()

# Exercise 5

exxon = data['EXXON_MOBIL']

microsoft_returns = microsoft.pct_change()
exxon_returns = exxon.pct_change()

exxon_returns = exxon_returns.replace([np.inf, -np.inf], np.nan).dropna()
microsoft_returns = microsoft_returns.replace([np.inf, -np.inf], np.nan).dropna()

X = exxon
Y = microsoft

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())

X = exxon_returns
Y = microsoft_returns

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary(yname='Microsoft_Returns', xname=['Intercept', 'Exxon_Returns']))


