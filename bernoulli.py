# Import the required libraries
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
from random import random
import numpy as np


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


sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]
samples_bernoulli = dict()
samples_laplace = dict()
theta_bernoulli = 0.7
theta_laplace = 8.5  # The scale parameter of the Laplace distribution
mu_laplace = 10.5

def generate_laplace_sample(theta_bernoulli, size):
    # Generate Laplace-distributed samples
    samples = np.random.laplace(scale=theta_bernoulli, size=size)
    return samples.tolist()


for sample_size in sample_sizes:
    sample_bernoulli = generate_bernoulli(theta_bernoulli, sample_size)

    samples_bernoulli[sample_size] = sample_bernoulli
    print(f"Generated {sample_size} samples for Bernoulli and Laplace distributions.")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(sample_bernoulli, bins='auto')
    # plt.hist(sample_bernoulli, bins=30, density=True, alpha=0.6, color='b', label='Bernoulli Samples')

    plt.title(f'Bernoulli Distribution (theta={theta_bernoulli}, sample_size={sample_size})')
    laplace_sample = generate_laplace_sample(theta_laplace, sample_size)
    samples_laplace[sample_size] = laplace_sample
    plt.show()
    plt.subplot(1, 2, 1)
    plt.hist(laplace_sample, bins='auto')
    # plt.hist(laplace_sample, bins=30, density=True, alpha=0.6, color='b', label='Laplace Samples')

    plt.title(f'Laplace Distribution (theta={theta_laplace}, sample_size={sample_size})')
    # plt.grid(True)

    plt.show()

# Эмпирическая функция распределения
def empirical_cdf(sample, t):
    return sum([1 for x in sample if x <= t]) / len(sample)

# Вычисление Дискриминанта
def compute_D(samples_x, samples_y):
    combined_samples = sorted(set(samples_x + samples_y))
    max_diff = 0
    for t in combined_samples:
        diff = abs(empirical_cdf(samples_x, t) - empirical_cdf(samples_y, t))
        max_diff = max(max_diff, diff)
    n, m = len(samples_x), len(samples_y)
    D = np.sqrt(n * m / (n + m)) * max_diff
    return D

sample_sizes = [5, 10, 100, 200, 400, 600, 800, 1000]

# Построение эмпирических функций распределения
for size in sample_sizes:
    t_vals = sorted(set(samples_bernoulli[size] + samples_laplace[size]))
    bernoulli_cdf_vals = [empirical_cdf(samples_bernoulli[size], t) for t in t_vals]
    laplace_cdf_vals = [empirical_cdf(samples_laplace[size], t) for t in t_vals]

    plt.plot(t_vals, bernoulli_cdf_vals, label=f'Bernoulli (n={size})')
    plt.plot(t_vals, laplace_cdf_vals, label=f'Laplace (n={size})')
    plt.title(f'Empirical CDFs for n = {size}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Вычисление D для каждой пары объемов выборок
for n in sample_sizes:
    for m in sample_sizes:
        D = compute_D(samples_bernoulli[n], samples_laplace[m])
        print(f'D for Bernoulli(n={n}) and Laplace(m={m}): {D}')


def laplace_pdf(x, theta, mu):
    return theta / 2 * np.exp(-theta * np.abs(x - mu))


def bernoulli_pmf(x, theta):
    return theta ** x * (1 - theta) ** (1 - x)

for n in sample_sizes:
    plt.figure(figsize=(12, 6))
    plt.hist(np.array(samples_laplace[n]), bins=np.arange(1, max(samples_laplace[n]) + 2) - 0.5, density=True, rwidth=0.8, align='mid', label='Histogram')
    k_vals = np.arange(1, max(samples_laplace[n]) + 1)
    plt.plot(k_vals, [bernoulli_pmf(k, theta_bernoulli) for k in k_vals], 'o-', label='PMF')  # предполагаем, что p=0.5
    plt.title(f'Bernoulli Distribution (n={n})')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(samples_bernoulli[n], bins=30, density=True, label='Histogram', alpha=0.7)
    x_vals = np.linspace(0, max(samples_bernoulli[n]), 400)
    plt.plot(x_vals, laplace_pdf(x_vals, theta_laplace, mu_laplace), label='PDF')  #
    plt.title(f'Laplace Distribution (n={n})')
    plt.legend()
    plt.grid(True)
    plt.show()