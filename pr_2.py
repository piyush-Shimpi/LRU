import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define class test results
math_scores = np.array([72, 78, 75, 73, 79, 82, 80, 76, 69, 77])

# Fit a normal distribution to the data
mu, sigma = stats.norm.fit(math_scores)

# Generate x values for the normal distribution
x = np.linspace(min(math_scores), max(math_scores), 100)

# Calculate the probability density function (PDF) of the normal distribution
pdf = stats.norm.pdf(x, mu, sigma)

# Plot the normal distribution
plt.plot(x, pdf, label='Normal Distribution')
plt.hist(math_scores, bins=10, edgecolor='black', alpha=0.5, label='Data')
plt.xlabel('Math Scores')
plt.ylabel('Probability Density')
plt.title('Normal Distribution of Math Scores')
plt.legend()
plt.show()

# Calculate skewness and kurtosis
skewness = stats.skew(math_scores)
kurtosis = stats.kurtosis(math_scores)

# Print skewness and kurtosis
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)



