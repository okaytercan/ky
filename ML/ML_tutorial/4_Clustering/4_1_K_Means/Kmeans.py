# required library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# data loading
data = pd.read_csv('customers.csv')

X = data.iloc[:,3:].values

# clustering model
kmeans = KMeans ( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print('Cluster Center : \n' ,kmeans.cluster_centers_)
result = []
# finding the optimum value for K
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    result.append(kmeans.inertia_)

plt.plot(range(1,11),result)
plt.show()
