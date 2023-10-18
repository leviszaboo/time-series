import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import norm

np.random.seed(1234)

# Exercise 1

sigma = 1
phis = [1, 0.6]
B = 1000
sample_sizes = [30, 100, 500, 1000, 2000, 5000]

beta_hat_dict = {}
r_squared_dict = {}
t_test_dict = {}
p_value_dict = {}

for phi in phis:
    beta_hat_dict[phi] = {}
    r_squared_dict[phi] = {}
    t_test_dict[phi] = {}
    p_value_dict[phi] = {}

    for sample_size in sample_sizes:
        beta_hat_dict[phi][sample_size] = np.zeros(B)
        r_squared_dict[phi][sample_size] = np.zeros(B)
        t_test_dict[phi][sample_size] = np.zeros(B)
        p_value_dict[phi][sample_size] = np.zeros(B)

    for sample_size in sample_sizes:
        for i in range(B):
            Y = np.zeros(sample_size)
            X = np.zeros(sample_size)
            for t in range(1, sample_size):
                ut = np.random.normal(0, phi)
                vt = np.random.normal(0, phi)
                X[t] = X[t - 1] + vt
                Y[t] = phi * X[t] + ut
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            beta_hat = model.params[1]
            se_beta_hat = model.bse[1]
            beta_hat_dict[phi][sample_size][i] = beta_hat
            r_squared_dict[phi][sample_size][i] = model.rsquared
            t_test_dict[phi][sample_size][i] = abs(beta_hat) / se_beta_hat
            p_value = model.pvalues[1]
            p_value_dict[phi][sample_size][i] = p_value

# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for phi in phis:
    for sample_size in sample_sizes:
        sns.kdeplot(beta_hat_dict[phi][sample_size], label=f'$\\phi$={phi}, Sample Size {sample_size}', ax=axes[0])

for phi in phis:
    for sample_size in sample_sizes:
        axes[0].hist(beta_hat_dict[phi][sample_size], alpha=0.1, density=True, bins=100)

axes[0].set_xlabel('$\\hat{\\beta}$')
axes[0].set_title('Distribution of $\\hat{\\gamma}$ for Different $\\phi$ values and Sample Sizes')
axes[0].legend()
axes[0].grid(True)

for phi in phis:
    for sample_size in sample_sizes:
        sns.kdeplot(r_squared_dict[phi][sample_size], label=f'$\\phi$={phi}, Sample Size {sample_size}', ax=axes[1])

for phi in phis:
    for sample_size in sample_sizes:
        axes[1].hist(r_squared_dict[phi][sample_size], alpha=0.1, density=True, bins=100)

axes[1].set_xlabel('R-squared')
axes[1].set_title('Distribution of R-squared for Different $\\phi$ values and Sample Sizes')
axes[1].set_xlim(0)
axes[1].legend()
axes[1].grid(True)

for phi in phis:
    for sample_size in sample_sizes:
        axes[2].hist(t_test_dict[phi][sample_size], alpha=0.6, density=True, bins=100, label=f'$\\phi$={phi}, Sample Size {sample_size}')

axes[2].set_title('Distribution of $| \\hat{\\gamma} | / SE(\\hat{\\gamma})$ for Different $\\phi$ values and Sample Sizes')
axes[2].set_xlabel('Sample Size')
axes[2].set_ylabel('Density')
axes[2].set_xlim(0, 1000)
axes[2].legend()
axes[2].set_axisbelow(True)
axes[2].grid(True)

plt.tight_layout()
plt.show()


# Exercise 2

data = pd.read_csv('assignment_4/data_assign_p4.csv')

cons = data['CONS']
inc = data['INC']

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(cons, label='Aggregate Consumption')
plt.title('Aggregate Consumption Time Series')
plt.xlabel('Time')
plt.ylabel('CONS')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(inc, label='Aggregate Income')
plt.title('Aggregate Income Time Series')
plt.xlabel('Time')
plt.ylabel('INC')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plot_acf(cons, lags=12, title='ACF for Consumption')
plot_pacf(cons, lags=12, title='PACF for Consumption')

plot_acf(inc, lags=12, title='ACF for Income')
plot_pacf(inc, lags=12, title='PACF for Income')

# Exercise 3 and 4

def perform_adf_test(series: pd.Series) -> tuple[float, bool, int]:

    result = adfuller(series, autolag="BIC", maxlag=30)

    adf_statistic = result[0]
    critical_values = result[4]

    is_stationary = adf_statistic < critical_values['5%']
    best_lag = result[2]

    return adf_statistic, is_stationary, best_lag

def adf_results(data: pd.DataFrame, difference = False) -> pd.DataFrame:
    results = []

    for column in data.columns[1:3]:
        series = data[column]
        if difference:
          series = series.diff().dropna()
          series = series.diff().dropna()

        adf_statistic, is_stationary, best_lag = perform_adf_test(series)

        result = {
            "Series": column,
            "ADF Statistic": adf_statistic,
            "Lag Order": best_lag,
            "Stationary at 95% Confidence Level": is_stationary
        }

        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df

results = adf_results(data)
results_diff = adf_results(data, difference=True)

print(results)
print(results_diff)

# Exercise 5

X = sm.add_constant(data['INC'])
y = data['CONS']
model = sm.OLS(y, X)
results = model.fit()
beta_hat = results.params['INC']

print(results.summary())

residuals = results.resid
adf_result = adfuller(residuals, autolag='BIC', maxlag=30)

n_lags = adf_result[2]

print("Estimated Regression Coefficients:")
print("Beta_hat (Income coefficient):", beta_hat)

print("\nADF Unit-Root Test on Residuals:")
adf_statistic = adf_result[0]
critical_value = -3.33613 - ( 6.1101 / 97 ) - ( 6.823 / 97 ** 2 )
print("ADF Statistic:", adf_statistic)
print("ADF Critical Value (from MacKinnon, 2010): ", critical_value)
print("Lag Order (chosen by ADF with SIC):", n_lags)

if adf_statistic < critical_value:
    print("Null Hypothesis (Unit Root) is Rejected at 5% Significance Level")
else:
    print("Null Hypothesis (Unit Root) is Not Rejected at 5% Significance Level")

plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Regression Residuals")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# Exercise 5

data['Zt_1'] = residuals.shift(1)

data['diff_Yt'] = data['CONS'].diff()
data['diff_Xt'] = data['INC'].diff()

for i in range(1, 5):
    data[f'diff_Yt_{i}'] = data['diff_Yt'].shift(i)
    data[f'diff_Xt_{i}'] = data['diff_Xt'].shift(i)

data = data.dropna(subset=['Zt_1', 'diff_Yt', 'diff_Xt', 'diff_Yt_1', 'diff_Xt_1', 'diff_Yt_2', 'diff_Xt_2', 'diff_Yt_3', 'diff_Xt_3', 'diff_Yt_4', 'diff_Xt_4'])

model = sm.OLS(data['diff_Yt'], data[['Zt_1', 'diff_Yt_3', 'diff_Yt_4', 'diff_Xt_1']])

results = model.fit()

print(results.summary())

print("Estimated ECM Coefficients:")
print("Error Correction Coefficient:", results.params['Zt_1'])
