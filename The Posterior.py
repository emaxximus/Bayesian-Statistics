"""
Created on Sat Mar  9 21:46:30 2024

@author: Emmanuel C. Doria
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def likelihood_func (datum, mu):
   likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
   return likelihood_out/likelihood_out.sum()

mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1
likelihood_out = likelihood_func(1.7, mu)
uniform_dist = uniform_dist/uniform_dist.sum()
unnormalized_posterior = likelihood_out * uniform_dist
unnormalized_posterior.sum() 

plt.plot(mu, unnormalized_posterior) 
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()

