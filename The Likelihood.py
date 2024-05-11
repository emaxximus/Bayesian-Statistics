"""
Created on Sat Mar  9 21:39:02 2024

@author: Emmanuel C. Doria
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

def likelihood_func (datum, mu):
   likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
   return likelihood_out/likelihood_out.sum()

mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1
likelihood_out = likelihood_func(1.7, mu)
uniform_dist = uniform_dist/uniform_dist.sum()

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given obeservation 1.7m")
plt.xlabel("Value of $\mu$")
plt.ylabel('Probability Density/Likelihood')
plt.show()

