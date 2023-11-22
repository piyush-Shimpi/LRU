import numpy as np
import matplotlib.pyplot as plt

# Sample data
heights = np.array([165, 172, 158, 170, 168, 162, 175, 167, 160, 169])

# Compute statistical measures
mean = np.mean(heights)
variance = np.var(heights)
standard_deviation = np.std(heights)

# Display distribution of heights
plt.hist(heights, bins=10, edgecolor='black')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Heights')
plt.show()

print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", standard_deviation)
