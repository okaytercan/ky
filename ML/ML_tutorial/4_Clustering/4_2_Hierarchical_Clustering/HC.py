# required library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# data loading
veriler = pd.read_csv('customers.csv')

X = veriler.iloc[:,3:].values

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

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_prediction= kmeans.fit_predict(X)
print('Y Prediction : \n',Y_prediction)  
plt.scatter(X[Y_prediction==0,0],X[Y_prediction==0,1],s=100, c='red')
plt.scatter(X[Y_prediction==1,0],X[Y_prediction==1,1],s=100, c='blue')
plt.scatter(X[Y_prediction==2,0],X[Y_prediction==2,1],s=100, c='green')
plt.scatter(X[Y_prediction==3,0],X[Y_prediction==3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()

# HC ---> Agglomerative Clustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_prediction_agglo = ac.fit_predict(X)
print(Y_prediction_agglo)

plt.scatter(X[Y_prediction_agglo==0,0],X[Y_prediction_agglo==0,1],s=100, c='red')
plt.scatter(X[Y_prediction_agglo==1,0],X[Y_prediction_agglo==1,1],s=100, c='blue')
plt.scatter(X[Y_prediction_agglo==2,0],X[Y_prediction_agglo==2,1],s=100, c='green')
plt.scatter(X[Y_prediction_agglo==3,0],X[Y_prediction_agglo==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

# Dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()