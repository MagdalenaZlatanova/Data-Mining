import pandas as pd
import pandas as ps
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from itertools import cycle, islice
from sklearn.metrics import calinski_harabasz_score
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

def spectral(data, n_clusters):
  clustering=MeanShift(cluster_all=False,n_jobs=-1).fit(data)

  y_pred = clustering.labels_.astype(int)
  colors = np.array(
    list(
      islice(
        cycle(
          [
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            "#e41a1c",
            "#dede00",
          ]
        ),
        int(max(y_pred) + 1),
      )
    )
  )
  plt.scatter(x=data[:,0], y=data[:,1],s=10, color=colors[y_pred])
  plt.show()

#spectral(data_pca,7)

#k_means(data_pca, 8)

#db=DBSCAN(eps=0.1,min_samples=1).fit(data_pca)

results = {}
for i in range(2,11):
    print(i)
    clusterx = MeanShift(cluster_all=False,n_jobs=-1)
    labels=clusterx.fit_predict(data_pca)
    #labels = kmeans.fit_predict(X)
    db_index = calinski_harabasz_score(data_pca, labels)
    results.update({i: db_index})

plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.show()