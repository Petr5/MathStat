# Import the required libraries
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

# Instance of Bernoulli distribution with parameter p = 0.8
bd=bernoulli(0.8)

# Outcome of random variable either 0 or 1
x=[0,1]

# For the visualization of the bar plot of Bernoulli's distribution
plt.figure(figsize=(10,10))
plt.xlim(-2, 2)
plt.bar(x, bd.pmf(x), color='blue')

# For labelling of Bar plot
plt.title('Bernoulli distribution (p=0.8)', fontsize='20')
plt.xlabel('Values of random variable x (0, 1)', fontsize='20')
plt.ylabel('Probability', fontsize='20')

plt.show()