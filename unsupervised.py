# -*- coding: utf-8 -*-
"""Untitled29.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15OArXBSsuwWuIqiWwH5zFNR1g_3cb4Np
"""

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

df=pd.read_csv('/content/courses.csv')
df.drop(['Unnamed: 0'],inplace=True,axis=1)
print(df.columns)
pca=PCA(n_components=2)
data_pca=pca.fit(df).transform(df)

db = DBSCAN(eps=0.1, min_samples=1).fit(data_pca)
sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=db.labels_)
plt.show()

db = DBSCAN(eps=0.05, min_samples=1).fit(data_pca)
sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=db.labels_)
plt.show()

numClusters = 10
SSE = []
silhouette_coefficients = []
for k in range(1, numClusters):
    k_means = KMeans(n_clusters=k)
    k_means.fit(df)
    SSE.append(k_means.inertia_)
    if k<=1:
      continue
    score = silhouette_score(df, k_means.labels_)
    silhouette_coefficients.append(score)
plt.plot(range(1, numClusters), SSE);
plt.xlabel('Number of Clusters');
plt.ylabel('SSE');

from kneed import KneeLocator
kl = KneeLocator(range(1, numClusters), SSE, curve="convex", direction="decreasing")
kl.elbow

plt.plot(range(2, numClusters), silhouette_coefficients)
plt.xticks(range(2, numClusters))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

print(silhouette_coefficients)

ac = AgglomerativeClustering(n_clusters=7).fit(data_pca)
sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=ac.labels_)
plt.show()

from scipy.spatial import distance

fig1 = plt.figure(figsize=(9, 9))
wcv = {}
bcv = {}


for k in range(1, 10):
    kmeans = KMeans(n_clusters=k,max_iter=1000).fit(df)
    wcv[k] = kmeans.inertia_ 
    centers = kmeans.cluster_centers_
    BCV = 0
    for i in range(len(centers)):
        for j in range(len(centers)):
            BCV += distance.euclidean(centers[i], centers[j])**2
    if(k==1):
        bcv[1] = 0
    else:
        bcv[k] = BCV/(k*(k-1))*100
plt.plot(list(wcv.keys()), list(wcv.values()), label="Within Cluster Distance (WCV)")
plt.plot(list(bcv.keys()), list(bcv.values()), label="Between Cluster Distance (BCV)")
plt.xlabel("Number of clusters K")
plt.legend()
plt.show()

estimator = KMeans(init='k-means++', n_clusters=8, random_state=170)
y_pred = estimator.fit_predict(df)
print(y_pred)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(df)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.2)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.2, c=y_pred)

np.unique(y_pred)

cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='average')
y_pred=cluster.fit_predict(df)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.2, c=y_pred)

import scipy.cluster.hierarchy as shc

import sys
sys.setrecursionlimit(10000)

plt.figure(figsize=(10, 7))
plt.title("Users dendograms")
dend = shc.dendrogram(shc.linkage(df[:100], method='ward'))

from sklearn.metrics import calinski_harabasz_score

results = {}
for i in range(2,11):
    print(i)
    clusterx = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
    labels=clusterx.fit_predict(df)
    db_index = calinski_harabasz_score(df, labels)
    results.update({i: db_index})

plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.show()