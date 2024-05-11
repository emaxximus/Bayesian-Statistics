
"""
Created on Sun Apr 28 19:14:00 2024

@author: DORIA, EMMANUEL C. | CAS-05-601P
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm

sns.set(style="darkgrid", palette="muted")

start = 0
stop = 10
N = 100
beta_0 = 5
beta_1 = 2
eps_mean = 0
eps_sigma_sq = 3

def simulate_linear_data (start, stop, N, beta_0, beta_1, 
                          eps_mean, eps_sigma_sq):
    
    x = np.linspace(start, stop, N)
    eps = np.random.normal(eps_mean, np.sqrt(eps_sigma_sq), N)
    y = beta_0 + beta_1 * x + eps
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Observed data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Simulated Linear Data')
    plt.show()
    
    return x, y

x, y = simulate_linear_data(start, stop, N, beta_0, beta_1, eps_mean, eps_sigma_sq)

with pm.Model() as model:
    
    slope = pm.Normal('slope', mu=0, sd=10)
    intercept = pm.Normal('intercept', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
      
    mu = slope * x + intercept
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=y)
    
    trace = pm.sample(1000, tune=2000, chains=2) 
    
print(pm.summary(trace))
pm.traceplot(trace)
