
import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(42)

X1 = np.random.normal(loc=5,scale=1.5,size=(100,1))
X2 = np.random.normal(loc=15,scale=2,size=(100,1))
X = np.vstack((X1,X2))


gmm = GaussianMixture(n_components=2,max_iter=100,random_state=42)
gmm.fit(X)

labels = gmm.predict(X)

print(gmm.means_.flatten())
print(gmm.covariances_.flatten())
print(gmm.weights_.flatten())
