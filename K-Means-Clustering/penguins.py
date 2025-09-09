import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Data
penguins = sns.load_dataset("penguins").dropna()
features = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

penguins["cluster"] = labels

centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
centers_unscaled = pd.DataFrame(scaler.inverse_transform(centers), columns=features.columns)
print("Cluster Centers (unscaled):\n", centers_unscaled)

# Visualize
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="bill_length_mm",
    y="flipper_length_mm",
    hue="cluster",
    palette="Set1",
    data=penguins,
    s=60,
    alpha=0.8
)
plt.title("K-Means Clusters on Palmer Penguins")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Flipper Length (mm)")
plt.legend(title="Cluster")
plt.show()

# Eval
print("\nCluster Counts:")
print(penguins["cluster"].value_counts().sort_index())
