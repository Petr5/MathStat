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
theta = 0.8
size = 10
print(generate_bernoulli(theta, size))
# Генерация выборок
for n in sample_sizes:
    sample_bernoulli = generate_bernoulli(theta, n)
    print(n)
    print(sample_bernoulli)
    samples_bernoulli[n] = sample_bernoulli
    # sample_ray = generate_rayleigh(param_rayleigh, n)
    # samples_ray[n] = sample_ray
    # print(sample_ray)
    # (Опционально) Выводим информацию о выборке
    print(f"Generated {n} samples for Bernoulli and Laplace distributions.")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(sample_bernoulli, bins='auto')
    plt.title(f'Bernoulli distribution with {n} samples')

    # plt.subplot(1, 2, 2)
    # plt.hist(sample_ray, bins='auto')
    # plt.title(f'Rayleigh distribution with {n} samples')

    plt.show()
