from sklearn.mixture import GaussianMixture
import pandas as pd
import pandas as ps
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from pyclustering.cluster import xmeans
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from itertools import cycle, islice
df=pd.read_csv('courses.csv')
df.drop(['Unnamed: 0'],inplace=True,axis=1)
print(df.columns)
pca=PCA(n_components=2)
data_pca=pca.fit(df).transform(df)
gmm = GaussianMixture(n_components=10)
gmm.fit(data_pca)
labels=gmm.predict(data_pca)
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
            int(max(labels) + 1),
        )
    )
)
plt.scatter(x=data_pca[:, 0], y=data_pca[:, 1], s=10, color=colors[labels])
plt.show()