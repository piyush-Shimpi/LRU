import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Load the Iris dataset 
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Fit a Gaussian Mixture Model with three components (assuming there are three species in the dataset)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predict the cluster labels
labels = gmm.predict(X)

# Reduce the data dimensionality using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clustered data using PCA
plt.figure(figsize=(10, 6))

for i in range(3):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i + 1}')

plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', marker='x', s=100, label='Cluster Centers')
plt.title('Clustering of Iris Dataset using Gaussian Mixture Models (GMM) with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
