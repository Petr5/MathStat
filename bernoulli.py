# Import the required libraries
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from random import random
# Instance of Bernoulli distribution with parameter theta
# theta = 0.8
# bd=bernoulli(theta)
#
# # Outcome of random variable either 0 or 1
# x=[0,1]
#
# # For the visualization of the bar plot of Bernoulli's distribution
# plt.figure(figsize=(10,10))
# plt.xlim(-2, 2)
# plt.bar(x, bd.pmf(x), color='blue')
#
# # For labelling of Bar plot
# plt.title('Bernoulli distribution (p=0.8)', fontsize='20')
# plt.xlabel('Values of random variable x (0, 1)', fontsize='20')
# plt.ylabel('Probability', fontsize='20')
#
# plt.show()

def generate_bernoulli(theta, size):
    bd = bernoulli(theta)
    # Outcome of random variable either 0 or 1
    x = [0, 1]
    ans = list()
    for i in range(size):
        value = random()
        if value <= theta:
            ans.append(1)
        else:
            ans.append(0)
    return ans
# Объемы выборок
sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
samples_bernoulli = dict()
samples_laplace = dict()
theta_bernoulli = 0.8
size = 10
print(generate_bernoulli(theta_bernoulli, size))
# Генерация выборок
import numpy as np
import matplotlib.pyplot as plt

def generate_laplace_sample(theta_bernoulli, size):
    # Generate Laplace-distributed samples
    samples = np.random.laplace(scale=theta_bernoulli, size=size)
    return samples

theta_laplace = 2.0  # The scale parameter of the Laplace distribution



for sample_size in sample_sizes:
    sample_bernoulli = generate_bernoulli(theta_bernoulli, sample_size)
    print(sample_size)
    print(sample_bernoulli)
    samples_bernoulli[sample_size] = sample_bernoulli
    print(f"Generated {sample_size} samples for Bernoulli and Laplace distributions.")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(sample_bernoulli, bins='auto')
    # plt.hist(sample_bernoulli, bins=30, density=True, alpha=0.6, color='b', label='Bernoulli Samples')

    plt.title(f'Bernoulli Distribution (theta={theta_bernoulli}, sample_size={sample_size})')
    laplace_sample = generate_laplace_sample(theta_laplace, sample_size)
    plt.show()
    plt.subplot(1, 2, 1)
    plt.hist(laplace_sample, bins='auto')
    # plt.hist(laplace_sample, bins=30, density=True, alpha=0.6, color='b', label='Laplace Samples')

    plt.title(f'Laplace Distribution (theta={theta_laplace}, sample_size={sample_size})')
    # plt.grid(True)

    # Show the plot
    plt.show()
