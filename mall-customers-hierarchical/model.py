import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy as sc

# Read files
df = pd.read_csv("hierarchy_clustering1/Mall_Customers.csv")
print(df.head())

# defining data we work on
data = df[["Annual Income (k$)","Spending Score (1-100)"]]

# visulizing dendrogram
plt.style.use("dark_background")
dnd_plot = sc.cluster.hierarchy.dendrogram(sc.cluster.hierarchy.linkage(data, method="ward"))
plt.title("Dendrogram")
plt.show()

# visulizing customer clusters
plot_lab = sklearn.cluster.AgglomerativeClustering(n_clusters=5, linkage="ward", metric="euclidean").fit_predict(data)

plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"], c=plot_lab)
plt.style.use("dark_background")
plt.title("Hierarchical Clustering")
plt.show()
