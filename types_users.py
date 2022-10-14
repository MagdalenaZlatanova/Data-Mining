import pandas as pd
import pandas as ps
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df=pd.read_csv('courses.csv')
df.drop(['Unnamed: 0'],inplace=True,axis=1)
print(df.columns)
pca=PCA(n_components=2)
data_pca=pca.fit(df).transform(df)
#plt.scatter(x=data_pca[:,0],y=data_pca[:,1])
#plt.show()

def k_means(data, n_clusters):
  k_means = KMeans(n_clusters=n_clusters, max_iter=50, random_state=1)
  k_means.fit(data)
  labels = k_means.labels_
  centroids = k_means.cluster_centers_
  pd.DataFrame(centroids)
  plt.scatter(x=data[:,0], y=data[:,1])
  plt.scatter(x=centroids[:,0], y=centroids[:,1], marker="*", color="r", s=100)
  plt.show()

k_means(data_pca, 8)

db=DBSCAN(eps=0.1,min_samples=1).fit(data_pca)
