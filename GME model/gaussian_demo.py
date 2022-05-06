import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import Ghelper as GH
import random
# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=3,cluster_std=1, random_state=0)
X = X[:, ::-1]
from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
from scipy.spatial.distance import cdist
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
from sklearn.mixture import GaussianMixture
X_rand = np.array([[6*random.random() for i in range(2)] for j in range(400)])
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
GH.plot_gmm(gmm, X_rand, label=True)
#plt.legend(loc='upper left')
plt.show()
