import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# file 
df = pd.read_csv("kmeans_clustering/Wholesale customers data.csv")

# defining data for model
data = df[["Milk","Grocery"]]

# modeling
model = sklearn.cluster.KMeans(n_clusters=3, random_state=20).fit(data)

# calculate center of each cluster
centers = model.cluster_centers_

# label and visualizations 
labels = model.labels_

plt.style.use("dark_background")
plt.scatter(df["Milk"], df["Grocery"], c=labels, label="Original Data")
plt.scatter(centers[:,0], centers[:,1], marker="*", c="white", label="Center of Cluster") # Visualizing Cluster Centers in Plot
plt.title("K-Means Clustering for Mike vs Grocery")
plt.legend()
plt.show()
