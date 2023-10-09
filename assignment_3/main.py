import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

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

